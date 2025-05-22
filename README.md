# AI Meeting Transcription and Summarization

A tool for transcribing and summarizing meetings from video files or YouTube URLs. This application uses state-of-the-art AI models to perform speaker diarization (identifying who is speaking) and speech-to-text transcription, followed by automatic summarization of key points.

## Features

- Process video files or YouTube URLs
- Extract audio automatically
- Identify different speakers (diarization)
- Transcribe speech to text with Whisper models
- Generate meeting summaries with key points
- Support for Spanish and English languages
- Export transcriptions in SubViewer format (.sub)

## Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended for faster processing)
- Hugging Face account with API token

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/matiaszanolli/AI-Meeting-Summary-SPANISH.git
   cd AI-Meeting-Summary-SPANISH
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a .env file in the project root with your Hugging Face token:
   ```bash
   HUGGINGFACE_AUTH_TOKEN=your_token_here
   ```

## Usage

1. Start the web interface:
   ```bash
   python web-ui.py
   ```

2. Open your browser at http://localhost:7860

3. Upload a video file or enter a YouTube URL

4. Configure the parameters:
   - Select language (Spanish, English, or auto-detect)
   - Choose Whisper model size (larger models are more accurate but slower)
   - Adjust collar value for speaker diarization
   - Enable/disable summary generation

5. Click "Iniciar" to start processing

6. View the transcription and summary in the respective tabs

## Models Used

- Speaker Diarization: pyannote/speaker-diarization-3.0
- Speech Recognition: OpenAI Whisper (various sizes)
- Summarization: Custom extractive summarization algorithm

## Output Files

- output.sub: Transcription in SubViewer format
- output_summary.txt: Meeting summary with key points
- output-tracks/: Directory containing audio segments for each speaker turn

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Whisper (https://github.com/openai/whisper)
- Pyannote Audio (https://github.com/pyannote/pyannote-audio)
- Gradio (https://www.gradio.app/)
- PyTube (https://github.com/pytube/pytube)