#!/usr/bin/env python3
import re
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from transformers import pipeline as hf_pipeline
from torchaudio import AudioMetaData
from pyannote.audio import Pipeline
from pyannote.database.loader import RTTMLoader
from pytube import YouTube
from pydub import AudioSegment
from dotenv import load_dotenv
import gradio as gr
import moviepy.editor as mp
import datetime
import logging
import os
import shutil

# Load .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Tokens, etc
# Hugging Face token: https://huggingface.co/docs/hub/security-tokens#user-access-tokens
HUGGINGFACE_AUTH_TOKEN = os.getenv("HUGGINGFACE_AUTH_TOKEN")
if HUGGINGFACE_AUTH_TOKEN is None:
    logging.error("HUGGINGFACE_AUTH_TOKEN is not set")
    exit(1)
logging.info(f"Hugging Face token: {HUGGINGFACE_AUTH_TOKEN}")

TEMP_VIDEO_FILE = "temp/input.mp4"
TEMP_AUDIO_FILE = "temp/input.wav"
TEMP_DIARIZATION_FILE = "temp/diarization.rttm"


def ensure_dir(path):
    """Make sure director from the given path exists"""
    dir = os.path.dirname(path)
    if dir:
        os.makedirs(dir, exist_ok=True)


def fetch_youtube(url, output_video_file, output_audio_file):
    """Fetch WAV audio from given youtube URL"""

    logging.info(f"Fetching audio from Youtube URL: {url}")

    ensure_dir(output_video_file)
    ensure_dir(output_audio_file)

    video_stream = YouTube(url).streams.first()
    video_stream.download(filename=output_video_file)

    video = mp.VideoFileClip(output_video_file)
    video.audio.write_audiofile(output_audio_file, codec='pcm_s16le')

    logging.info("Done fetching audio form YouTube")


def extract_wav_from_video(video_file, output_audio_file):
    """Extract WAV audio from given video file"""

    logging.info(f"Extracting audio from video file: {video_file}")

    ensure_dir(output_audio_file)
    
    try:
        video = mp.VideoFileClip(video_file)
        video.audio.write_audiofile(output_audio_file, codec='pcm_s16le')
        logging.info("Done extracting audio from video file")
    except (OSError, IOError) as e:
        logging.error(f"Error processing video file: {e}")
        raise gr.Error("El archivo de video parece estar dañado o no es compatible. Por favor, intente con otro archivo o URL de YouTube.")


TIMESTAMP_FORMAT = "%H:%M:%S.%f"
base_time = datetime.datetime(1970, 1, 1)


def format_timestamp(seconds):
    """Format timestamp in SubViewer format: https://wiki.videolan.org/SubViewer/"""

    date = base_time + datetime.timedelta(seconds=seconds)
    return date.strftime(TIMESTAMP_FORMAT)[:-4]


def extract_audio_track(input_file, start_time, end_time, track_file):
    """Extract and save part of given audio file"""

    # Load the WAV file
    audio = AudioSegment.from_wav(input_file)

    # Calculate the start and end positions in milliseconds
    start_ms = start_time * 1000
    end_ms = end_time * 1000

    # Extract the desired segment
    track = audio[start_ms:end_ms]

    track.export(track_file, format="mp3")


def generate_speaker_diarization(audio_file):
    """Generate speaker diarization for given audio file"""

    logging.info(f"Generating speaker diarization... audio file: {audio_file}")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token=HUGGINGFACE_AUTH_TOKEN)

    result = pipeline(audio_file)

    logging.info("Done generating spearer diarization")

    with open(TEMP_DIARIZATION_FILE, "w") as rttm:
        result.write_rttm(rttm)

    logging.info(f"Wrote diarization file: {TEMP_DIARIZATION_FILE}", )

    return result


def generate_transcription(diarization, model, collar, language):
    """Generate transcription from given diarization object"""

    logging.info(f"Generating transcription... model: {model}, language: {language}")

    # For Spanish, we should use a model that's not language-specific
    # For English, we can use the .en models which are optimized for English
    model_name = model
    if language == "es" and ".en" in model:
        # If user selected an English-specific model but wants Spanish output,
        # switch to the base model version
        model_name = model.replace(".en", "")
        logging.info(f"Switching to base model {model_name} for Spanish transcription")

    # Initialize the pipeline with the appropriate model
    pipe = pipeline(
        "automatic-speech-recognition",
        model=f"openai/whisper-{model_name}",
        chunk_length_s=30,
        device="cuda"
    )

    # Create directory for tracks
    shutil.rmtree("output-tracks", ignore_errors=True)
    os.mkdir("output-tracks")

    result = []
    for turn, _, speaker in diarization.support(collar).itertracks(yield_label=True):
        part_file = f"output-tracks/{round(turn.start, 2)}-{speaker}.mp3"
        part_path = os.path.join(os.curdir, part_file)
        extract_audio_track(TEMP_AUDIO_FILE, turn.start, turn.end, part_file)

        part_data = None
        with open(part_path, "rb") as audio_content:
            part_data = audio_content.read()

        # Call the pipeline with basic parameters
        output = pipe(part_data, batch_size=8, return_timestamps=False)
        text = output['text']

        result.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker,
            'text': text.strip(),
            'track_path': part_path
        })

    logging.info(f"Done generating transcription tracks: {len(result)}")
    return result


def format_transcription(transcription):
    """Format transcription in SubViewer format: https://wiki.videolan.org/SubViewer/"""

    result = ""
    for t in transcription:
        result += f"{format_timestamp(t['start'])},{format_timestamp(t['end'])}\n{t['speaker']}: {t['text']}\n\n"
    return result


def save_transcription(transcription):
    """Save trainscription in SubViewer format to file."""

    logging.info("Saving transcripion... to file: output.sub")

    f = open("output.sub", "w")
    for t in transcription:
        # Format in SubViewer format: https://wiki.videolan.org/SubViewer/
        f.write(
            f"{format_timestamp(t['start'])},{format_timestamp(t['end'])}\n{t['speaker']}: {t['text']}\n\n")
    f.close()

    logging.info("Done saving transcripion")

def generate_meeting_summary(transcription, max_length=1024):
    """Generate a summary of the meeting from the transcription using an extractive approach"""
    
    logging.info("Generating meeting summary using extractive approach...")
    
    # Step 1: Combine all transcription segments into a single text
    full_text = ""
    for t in transcription:
        full_text += f"{t['speaker']}: {t['text']}\n"
    
    # Step 2: Split into sentences and clean them
    import re
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    
    # Step 3: Clean and filter sentences
    clean_sentences = []
    for sentence in sentences:
        # Remove speaker tags
        cleaned = re.sub(r'SPEAKER_\d+:', '', sentence).strip()
        # Remove sentences that are too short or contain problematic patterns
        if (len(cleaned.split()) > 5 and 
            not cleaned.startswith("Ahead of") and
            not re.search(r'^\d+\.\d+\s*-', cleaned) and
            not re.search(r'^\d+\.\d+\s*billion', cleaned) and
            "€" not in cleaned and
            "billion euros" not in cleaned.lower()):
            clean_sentences.append(cleaned)
    
    # Step 4: Score sentences by importance (simple heuristic)
    # We'll use term frequency as a simple metric
    from collections import Counter
    
    # Get word frequencies
    words = " ".join(clean_sentences).lower().split()
    # Remove common Spanish stopwords
    stopwords = ["el", "la", "los", "las", "you", "un", "una", "unos", "unas", "y", "o", "pero", 
                "porque", "que", "de", "a", "en", "por", "para", "con", "sin", "sobre",
                "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "aquel",
                "aquella", "aquellos", "aquellas", "mi", "tu", "su", "nuestro", "vuestro"]
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    word_freq = Counter(filtered_words)
    
    # Score sentences based on word importance
    sentence_scores = []
    for sentence in clean_sentences:
        score = 0
        for word in sentence.lower().split():
            if word in word_freq:
                score += word_freq[word]
        # Normalize by sentence length to avoid bias toward longer sentences
        score = score / max(5, len(sentence.split()))
        sentence_scores.append((score, sentence))
    
    # Step 5: Select top sentences
    sentence_scores.sort(reverse=True)  # Sort by score, highest first
    top_sentences = [s[1] for s in sentence_scores[:15]]  # Take top 15 sentences
    
    # Step 6: Reorder sentences to maintain chronological order
    # This preserves the flow of the meeting
    ordered_top_sentences = []
    for sentence in clean_sentences:
        if sentence in [s for s in top_sentences]:
            ordered_top_sentences.append(sentence)
            if len(ordered_top_sentences) >= 15:  # Limit to 15 sentences
                break
    
    # Step 7: Format as key points in Spanish
    key_points = "Puntos clave de la reunión:\n\n"
    
    # Remove duplicates or very similar sentences
    unique_sentences = []
    seen_content = set()
    
    for sentence in ordered_top_sentences:
        # Create a simplified version for duplicate detection
        simplified = re.sub(r'\s+', ' ', sentence.lower())
        simplified = re.sub(r'[^\w\s]', '', simplified)
        
        # Check if we've seen something very similar
        is_duplicate = False
        for seen in seen_content:
            # If more than 70% of words are the same, consider it a duplicate
            words1 = set(simplified.split())
            words2 = set(seen.split())
            common_words = words1.intersection(words2)
            if len(common_words) > 0.7 * min(len(words1), len(words2)):
                is_duplicate = True
                break
        
        if not is_duplicate:
            seen_content.add(simplified)
            unique_sentences.append(sentence)
    
    # Format the final output
    for i, sentence in enumerate(unique_sentences, 1):
        key_points += f"{i}. {sentence}\n\n"
    
    logging.info("Done generating meeting summary")
    return key_points


def format_key_points_in_spanish(summary_text):
    """Format the summary as key points in Spanish, avoiding repetition"""
    
    # Use a GPT-2 model which is supported for text-generation
    generator = hf_pipeline(
        "text-generation",
        model="gpt2",  # Using GPT-2 which is supported
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Break the summary into smaller chunks to avoid exceeding max length
    max_chunk_size = 400  # Keeping well under the 512 token limit
    summary_chunks = [summary_text[i:i+max_chunk_size] for i in range(0, len(summary_text), max_chunk_size)]
    
    # Process each chunk and extract key points
    all_key_points = []
    
    for i, chunk in enumerate(summary_chunks):
        # Create a prompt for extracting key points in Spanish
        prompt = f"""
        Extrae los puntos clave más importantes del siguiente texto en español:
        
        {chunk}
        
        Puntos clave en español:
        1."""
        
        try:
            result = generator(
                prompt,
                max_length=len(prompt) + 200,  # Keep generated text relatively short
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=50256  # GPT-2's pad token ID
            )
            
            # Extract the generated text and format it
            generated_text = result[0]['generated_text']
            # Extract only the part after "Puntos clave en español:"
            if "Puntos clave en español:" in generated_text:
                points_text = generated_text.split("Puntos clave en español:")[1].strip()
                # Clean up the text to make it more readable
                points_text = points_text.replace("\n\n", "\n").strip()
                all_key_points.append(points_text)
            else:
                # If the model didn't follow the format, extract what we can
                points_text = generated_text[len(prompt)-2:].strip()
                all_key_points.append(points_text)
                
        except Exception as e:
            logging.error(f"Error generating key points for chunk {i}: {e}")
            all_key_points.append(f"Error al procesar esta sección: {str(e)}")
    
    # Combine all key points
    combined_points = "\n".join(all_key_points)
    
    # Post-process to ensure we have a clean numbered list and remove duplicates
    lines = combined_points.split("\n")
    unique_points = []
    seen_content = set()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Remove numbering and leading characters to check content
        content = re.sub(r'^\d+[\.\)\-]\s*', '', line).strip().lower()
        
        # Skip very short points or points we've already seen
        if len(content) < 10 or content in seen_content:
            continue
            
        seen_content.add(content)
        unique_points.append(line)
    
    # Format the final output with proper numbering
    final_points = "Puntos clave:\n"
    for i, point in enumerate(unique_points, 1):
        # If the point already starts with a number, replace it
        point = re.sub(r'^\d+[\.\)\-]\s*', '', point)
        final_points += f"{i}. {point}\n"
    
    return final_points

# Alternative approach for Spanish summaries
def format_key_points_spanish_alternative(summary_text):
    """Alternative approach to extract key points in Spanish using sentence importance"""
    
    # Split the summary into sentences
    import re
    sentences = re.split(r'(?<=[.!?])\s+', summary_text)
    
    # Filter out very short sentences
    sentences = [s for s in sentences if len(s.split()) > 5]
    
    # Remove duplicate sentences (case insensitive comparison)
    unique_sentences = []
    seen_sentences = set()
    for sentence in sentences:
        lower_sentence = sentence.lower()
        if lower_sentence not in seen_sentences:
            seen_sentences.add(lower_sentence)
            unique_sentences.append(sentence)
    
    # If we have too many sentences, select a subset
    if len(unique_sentences) > 10:
        # Simple approach: take sentences from beginning, middle and end
        selected = unique_sentences[:3] + unique_sentences[len(unique_sentences)//2-1:len(unique_sentences)//2+2] + unique_sentences[-3:]
    else:
        selected = unique_sentences
    
    # Format as key points in Spanish
    key_points = "Puntos clave:\n"
    for i, sentence in enumerate(selected, 1):
        key_points += f"{i}. {sentence.strip()}\n"
    
    return key_points

# If GPT-2 doesn't work well with Spanish, use this function instead
def translate_and_format_key_points(summary_text):
    """Translate the summary to Spanish and format as key points"""
    
    # First, extract key points in English (which GPT-2 handles better)
    generator = hf_pipeline(
        "text-generation",
        model="gpt2",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Create a prompt for extracting key points
    prompt = f"Extract key points from this text:\n\n{summary_text}\n\nKey points:\n1."
    
    result = generator(
        prompt,
        max_length=len(prompt) + 200,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    # Extract the generated text
    generated_text = result[0]['generated_text']
    if "Key points:" in generated_text:
        points_text = generated_text.split("Key points:")[1].strip()
    else:
        points_text = generated_text[len(prompt)-2:].strip()
    
    # Then translate to Spanish using a translation model
    translator = hf_pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-en-es",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Translate each point separately for better results
    points = points_text.split("\n")
    translated_points = []
    
    for point in points:
        if point.strip():
            # Remove numbering before translation
            clean_point = re.sub(r'^\d+[\.\)\-]\s*', '', point).strip()
            if clean_point:
                translation = translator(clean_point)
                translated_points.append(translation[0]['translation_text'])
    
    # Format the final output with proper numbering and remove duplicates
    seen_content = set()
    final_points = "Puntos clave:\n"
    point_number = 1
    
    for point in translated_points:
        # Check for duplicates (case insensitive)
        lower_point = point.lower()
        if lower_point not in seen_content and len(point.split()) > 3:
            seen_content.add(lower_point)
            final_points += f"{point_number}. {point}\n"
            point_number += 1
    
    return final_points


def format_key_points_with_gpt(summary_text):
    """Format the summary as key points using GPT-2 model"""
    
    # Use a GPT-2 model which is supported for text-generation
    generator = hf_pipeline(
        "text-generation",
        model="gpt2",  # Using GPT-2 which is supported
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Break the summary into smaller chunks to avoid exceeding max length
    max_chunk_size = 400  # Keeping well under the 512 token limit
    summary_chunks = [summary_text[i:i+max_chunk_size] for i in range(0, len(summary_text), max_chunk_size)]
    
    # Process each chunk and extract key points
    all_key_points = []
    
    for i, chunk in enumerate(summary_chunks):
        # Create a prompt for extracting key points
        prompt = f"Extracting key points from this text in Spanish:\n\n{chunk}\n\nPuntos clave:\n1."
        
        try:
            result = generator(
                prompt,
                max_length=len(prompt) + 200,  # Keep generated text relatively short
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=50256  # GPT-2's pad token ID
            )
            
            # Extract the generated text and format it
            generated_text = result[0]['generated_text']
            # Extract only the part after "Puntos clave:"
            if "Puntos clave:" in generated_text:
                points_text = generated_text.split("Puntos clave:")[1].strip()
                # Clean up the text to make it more readable
                points_text = points_text.replace("\n\n", "\n").strip()
                all_key_points.append(points_text)
            else:
                # If the model didn't follow the format, extract what we can
                points_text = generated_text[len(prompt)-2:].strip()
                all_key_points.append(points_text)
                
        except Exception as e:
            logging.error(f"Error generating key points for chunk {i}: {e}")
            all_key_points.append(f"Error processing this section: {str(e)}")
    
    # Combine all key points
    combined_points = "\n".join(all_key_points)
    
    # Post-process to ensure we have a clean numbered list
    # This is a simple approach - you might need more sophisticated processing
    final_points = "Puntos clave:\n"
    point_number = 1
    
    for line in combined_points.split("\n"):
        line = line.strip()
        if line and not line.startswith(str(point_number) + "."):
            # Add numbering if it's not already there
            final_points += f"{point_number}. {line}\n"
            point_number += 1
        elif line:
            final_points += f"{line}\n"
            # Try to extract the next point number
            try:
                if "." in line:
                    potential_number = int(line.split(".")[0])
                    point_number = potential_number + 1
            except:
                point_number += 1
    
    return final_points


# Alternative approach using a text-classification model for key points
def format_key_points_alternative(summary_text):
    """Alternative approach to extract key points using sentence importance"""
    
    # Split the summary into sentences
    import re
    sentences = re.split(r'(?<=[.!?])\s+', summary_text)
    
    # Filter out very short sentences
    sentences = [s for s in sentences if len(s.split()) > 5]
    
    # If we have too many sentences, select a subset
    if len(sentences) > 10:
        # Simple approach: take sentences from beginning, middle and end
        selected = sentences[:3] + sentences[len(sentences)//2-1:len(sentences)//2+2] + sentences[-3:]
    else:
        selected = sentences
    
    # Format as key points
    key_points = "Puntos clave:\n"
    for i, sentence in enumerate(selected, 1):
        key_points += f"{i}. {sentence.strip()}\n"
    
    return key_points


# Modify the process_video function to include summarization
def process_video(youtube_url, video_file, model, collar, skip, language, generate_summary=False, progress=gr.Progress()):
    """Main function to run the whole procesessing pipeline."""

    try:
        if "Extract audio" not in skip:
            if video_file:
                progress(0.1, desc="Processing video file...")
                extract_wav_from_video(
                    video_file,
                    output_audio_file=TEMP_AUDIO_FILE,
                )
            elif youtube_url:
                progress(0.1, desc="Downloading video...")
                try:
                    fetch_youtube(
                        youtube_url,
                        output_audio_file=TEMP_AUDIO_FILE,
                        output_video_file=TEMP_VIDEO_FILE
                    )
                except Exception as e:
                    logging.error(f"Error downloading YouTube video: {e}")
                    raise gr.Error("No se pudo descargar el video de YouTube. Verifique la URL e intente nuevamente.")
            else:
                raise gr.Error("Proporcione una URL de YouTube o un archivo de video")
        else:
            progress(0.1, desc=f"Reusing local file... {TEMP_AUDIO_FILE}")
            logging.debug(f"Reusing local file {TEMP_AUDIO_FILE}")
            if not os.path.exists(TEMP_AUDIO_FILE):
                raise gr.Error(f"El archivo de audio temporal {TEMP_AUDIO_FILE} no existe. No se puede omitir la extracción de audio.")

        if "Speaker diarization" not in skip:
            progress(
                0.5, desc="Generating speaker diarization... (this may take a while)")
            diarization = generate_speaker_diarization(TEMP_AUDIO_FILE)
        else:
            progress(0.5, desc="Reusing local dirization file...")
            logging.info(
                f"Reusing local dirization file... {TEMP_DIARIZATION_FILE}")
            if not os.path.exists(TEMP_DIARIZATION_FILE):
                raise gr.Error(f"El archivo de diarización temporal {TEMP_DIARIZATION_FILE} no existe. No se puede omitir la diarización.")
            rttm = RTTMLoader(TEMP_DIARIZATION_FILE).loaded_
            diarization = rttm['input']

        progress(0.8, desc="Generating transcription... (this may take a while)")
        transcription = generate_transcription(diarization, model, collar, language)

        output = format_transcription(transcription)
        save_transcription(transcription)
        
        # Generate summary if requested
        summary = ""
        if generate_summary:
            progress(0.9, desc="Generando resumen de la reunión...")
            summary = generate_meeting_summary(transcription)
            
            # Save summary to file
            with open("output_summary.txt", "w") as f:
                f.write(summary)
        
        progress(1.0, desc="Done!")
        return output, summary
    except gr.Error as e:
        # Re-raise Gradio errors to display them properly
        raise e
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise gr.Error(f"Error inesperado: {str(e)}")

# Update the UI to include the summary functionality
with gr.Blocks() as ui:
    gr.Markdown(
        """
        # Herramienta de Transcripción y Resumen de Video
        Sube un archivo de video o pega una URL de YouTube. Luego presiona "Iniciar" para generar una transcripción y un resumen.
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            video_file = gr.Video()

            youtube_url = gr.Textbox(
                label="URL de YouTube",
                placeholder="https://www.youtube.com/watch?v=...",
                value="https://www.youtube.com/watch?v=4V2C0X4qqLY",
            )

            # Reset youtube URL on video upload
            video_file.upload(lambda: None, [], youtube_url)
            youtube_url.input(lambda: None, [], video_file)

            start_btn = gr.Button("Iniciar")

        with gr.Column(scale=1):
            pass

        with gr.Column(scale=3, variant="panel"):
            gr.Markdown(
                """
                ### Parámetros
                """
            )
            collar = gr.Number(
                label="Collar (segundos)",
                info="Unirá dos pistas consecutivas si están más cerca que este valor",
                value=0.5,
                minimum=0,
                maximum=30,
                step=0.1)

            language = gr.Dropdown([
                "auto", "es", "en"],
                value="es",
                label="Idioma",
                info="'auto' detectará el idioma, 'es' para español, 'en' para inglés"
            )

            model = gr.Dropdown([
                "tiny", "tiny.en",
                "base", "base.en",
                "small", "small.en",
                "medium", "medium.en",
                "large", "large-v3"],
                value="base",
                label="Modelo Whisper",
                info="Para español, use modelos sin '.en'. Para inglés, los modelos con '.en' funcionan mejor."
            )

            # Update model selection based on language
            def update_model_info(lang):
                if lang == "es":
                    return gr.Dropdown.update(
                        info="Para español, recomendamos usar modelos sin '.en' (base, small, medium, large)"
                    )
                elif lang == "en":
                    return gr.Dropdown.update(
                        info="Para inglés, recomendamos usar modelos con '.en' (tiny.en, base.en, small.en, medium.en)"
                    )
                else:
                    return gr.Dropdown.update(
                        info="Para detección automática, recomendamos modelos sin '.en'"
                    )
            
            language.change(update_model_info, inputs=[language], outputs=[model])

            skip_group = gr.CheckboxGroup(
                ["Extract audio", "Speaker diarization"],
                label="Omitir Pasos",
                info="Omitir pasos ya realizados para acelerar el procesamiento."
            )
            
            # Add checkbox for summary generation
            generate_summary = gr.Checkbox(
                label="Generar resumen",
                value=True,
                info="Genera un resumen con los puntos clave de la reunión"
            )

    with gr.Tabs():
        with gr.TabItem("Transcripción"):
            gr.Markdown(
                """
                ## Transcripción
                La transcripción completa aparecerá aquí. También se guardará en el archivo `output.sub`.
                """
            )
            
            output_text = gr.Textbox(
                label="Texto de salida",
                max_lines=25,
                show_copy_button=True,
            )
            
        with gr.TabItem("Resumen"):
            gr.Markdown(
                """
                ## Resumen de la Reunión
                Los puntos clave de la reunión aparecerán aquí. También se guardarán en el archivo `output_summary.txt`.
                """
            )
            
            summary_text = gr.Textbox(
                label="Puntos clave",
                max_lines=25,
                show_copy_button=True,
            )

    start_btn.click(
        fn=process_video,
        inputs=[youtube_url, video_file, model, collar, skip_group, language, generate_summary],
        outputs=[output_text, summary_text]
    )

ui.queue()
ui.launch(inbrowser=True)
