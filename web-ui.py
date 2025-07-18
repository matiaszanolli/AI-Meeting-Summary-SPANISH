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

    # Make sure we're exporting in a format that Whisper can handle
    # WAV is more reliable than MP3 for this purpose
    ensure_dir(track_file)
    track.export(track_file, format="wav")  # Changed from mp3 to wav


def generate_speaker_diarization(audio_file, num_speakers=None):
    """Generate speaker diarization for given audio file
    
    Parameters:
    audio_file (str): Path to the audio file
    num_speakers (int, optional): Number of speakers in the audio. If provided, improves diarization accuracy and speed.
    """

    logging.info(f"Generating speaker diarization... audio file: {audio_file}, num_speakers: {num_speakers}")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGINGFACE_AUTH_TOKEN)
    
    pipeline.to(torch.device("cuda"))  # Use GPU if available
    
    # Apply speaker count parameter if provided
    if num_speakers is not None:
        logging.info(f"Using fixed number of speakers: {num_speakers}")
        result = pipeline(audio_file, num_speakers=num_speakers)
    else:
        result = pipeline(audio_file)

    logging.info("Done generating speaker diarization")

    with open(TEMP_DIARIZATION_FILE, "w") as rttm:
        result.write_rttm(rttm)

    logging.info(f"Wrote diarization file: {TEMP_DIARIZATION_FILE}")

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
    os.makedirs("output-tracks", exist_ok=True)

    result = []
    for turn, _, speaker in diarization.support(collar).itertracks(yield_label=True):
        # Use WAV format instead of MP3
        part_file = f"output-tracks/{round(turn.start, 2)}-{speaker}.wav"  # Changed from mp3 to wav
        part_path = os.path.join(os.curdir, part_file)
        extract_audio_track(TEMP_AUDIO_FILE, turn.start, turn.end, part_file)

        try:
            # Read the audio file
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
        except Exception as e:
            logging.error(f"Error processing audio segment {part_file}: {e}")
            # Add a placeholder for the failed segment
            result.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker,
                'text': "[Error al transcribir este segmento]",
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


def generate_meeting_summary(transcription, max_points=10):
    """Generate a summary of the meeting with improved formatting and context"""
    
    logging.info("Generating meeting summary with improved formatting...")
    
    # Step 1: Extract meaningful segments with better filtering
    meaningful_segments = []
    
    for t in transcription:
        text = t['text'].strip()
        speaker = t['speaker']
        
        # Skip if text is too short
        if len(text) < 40:  # Increased minimum length for better context
            continue
            
        # Skip segments with excessive repetition
        words = text.split()
        if len(words) > 5:
            unique_words = set(words)
            repetition_ratio = len(unique_words) / len(words)
            if repetition_ratio < 0.5:  # Increased uniqueness requirement
                continue
        
        # Skip segments that are just repetitions of the same word/phrase
        repeated_pattern = False
        if len(words) > 10:
            # Check for repeating patterns
            for i in range(1, 6):  # Check for patterns of length 1-5 words
                if len(words) >= i*3:  # Need at least 3 repetitions to detect a pattern
                    pattern = ' '.join(words[:i])
                    pattern_count = 0
                    for j in range(0, len(words), i):
                        if j + i <= len(words) and ' '.join(words[j:j+i]) == pattern:
                            pattern_count += 1
                    
                    if pattern_count >= 3 and pattern_count * i > len(words) * 0.5:
                        repeated_pattern = True
                        break
        
        if repeated_pattern:
            continue
        
        # Improved scoring for sentence completeness
        if text.endswith((".", "!", "?")):
            sentence_score = 1.5  # Complete sentence gets higher score
        else:
            # Check if it's a substantial fragment
            if len(text) < 60:  # Short fragments are likely incomplete thoughts
                continue
            sentence_score = 0.7  # Longer fragment, might be useful
        
        # Skip segments that are likely not meaningful content
        filler_phrases = ["la idea sería", "bueno", "este", "pues", "dale", "eh", "o sea", "entonces", 
                          "hmm", "yeah", "de repente", "vamos a"]
        has_filler = False
        for phrase in filler_phrases:
            if text.lower().startswith(phrase) and len(text) < 70:
                has_filler = True
                break
        
        if has_filler:
            continue
        
        # Check for contextual relevance - segments should contain keywords related to the project/meeting
        relevant_keywords = ["proyecto", "desarrollo", "cliente", "usuario", "producto", "pictograma",
                            "tecnología", "aplicación", "sistema", "paciente", "terapia", "picto",
                            "educación", "aprendizaje", "interfaz", "diseño", "innovación", "android",
                            "iOS", "móvil", "web", "software", "hardware", "empresa", "negocio", 
                            "mercado", "negocio", "estrategia", "plan", "meta", "objetivo", "resultado",
                            "beneficio", "ventaja", "competencia", "mercado", "cliente", "usuario",
                            "comunicación", "empresa", "inversión", "inversor", "estrategia", "tablet"]
        
        relevance_score = 0
        for keyword in relevant_keywords:
            if keyword in text.lower():
                relevance_score += 0.5
        
        # Add the segment if it passed all filters
        meaningful_segments.append({
            'speaker': speaker,
            'text': text,
            'score': (sentence_score + relevance_score) * len(text) / 100  # Score based on length, completeness and relevance
        })
    
    # Step 2: Sort by score and select top segments
    meaningful_segments.sort(key=lambda x: x['score'], reverse=True)
    
    # Make sure we have enough segments
    if len(meaningful_segments) < max_points:
        max_points = len(meaningful_segments)
    
    if max_points == 0:
        return "No se pudieron extraer puntos clave significativos de la transcripción."
    
    # Step 3: Ensure diversity by selecting from different parts of the meeting
    final_segments = []
    
    # If we have enough segments, select from beginning, middle, and end
    if len(meaningful_segments) > max_points:
        # Divide into three parts
        part_size = len(meaningful_segments) // 3
        beginning = meaningful_segments[:part_size]
        middle = meaningful_segments[part_size:2*part_size]
        end = meaningful_segments[2*part_size:]
        
        # Take top segments from each part
        points_per_part = max(1, max_points // 3)
        final_segments.extend(beginning[:points_per_part])
        final_segments.extend(middle[:points_per_part])
        final_segments.extend(end[:points_per_part])
        
        # Add any remaining points from the highest scored segments
        remaining = max_points - len(final_segments)
        if remaining > 0:
            # Get segments not already selected
            unused = [s for s in meaningful_segments if s not in final_segments]
            final_segments.extend(unused[:remaining])
    else:
        final_segments = meaningful_segments[:max_points]
    
    # Step 4: Format as key points with improved context - REMOVE SPEAKER TAGS
    key_points = "Puntos clave de la reunión:\n\n"
    
    for i, segment in enumerate(final_segments, 1):
        # Clean up the text
        text = segment['text']
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        # Ensure first letter is capitalized
        if text and len(text) > 0:
            text = text[0].upper() + text[1:]
        
        # Make sure the text ends with proper punctuation
        if not text.endswith((".", "!", "?")):
            text += "."
        
        # Add the point WITHOUT speaker information
        key_points += f"{i}. {text}\n\n"
    
    logging.info("Done generating improved meeting summary")
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


def process_video(youtube_url, video_file, model, collar, skip, language, generate_summary=False, num_speakers=None, progress=gr.Progress()):
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
            diarization = generate_speaker_diarization(TEMP_AUDIO_FILE, num_speakers)
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


# Update the UI to include the number of speakers input
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

            # Add number of speakers input
            num_speakers = gr.Number(
                label="Número de hablantes",
                info="Si conoce el número exacto de hablantes, especificarlo mejorará la velocidad y precisión de la diarización",
                value=None,
                minimum=1,
                maximum=20,
                step=1
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
        inputs=[youtube_url, video_file, model, collar, skip_group, language, generate_summary, num_speakers],
        outputs=[output_text, summary_text]
    )

ui.queue()
ui.launch(inbrowser=True)
