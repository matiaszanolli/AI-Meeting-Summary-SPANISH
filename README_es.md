# Transcripción y Resumen de Reuniones con IA

Una herramienta para transcribir y resumir reuniones a partir de archivos de video o URLs de YouTube. Esta aplicación utiliza modelos de IA de última generación para realizar diarización de hablantes (identificar quién está hablando) y transcripción de voz a texto, seguido de un resumen automático de los puntos clave.

## Características

- Procesa archivos de video o URLs de YouTube
- Extrae audio automáticamente
- Identifica diferentes hablantes (diarización)
- Transcribe voz a texto con modelos Whisper
- Genera resúmenes de reuniones con puntos clave
- Soporte para idiomas español e inglés
- Exporta transcripciones en formato SubViewer (.sub)

## Requisitos

- Python 3.8+
- GPU compatible con CUDA (recomendado para procesamiento más rápido)
- Cuenta de Hugging Face con token de API

## Instalación

1. Clona este repositorio:
   git clone https://github.com/tunombre/ai-meeting-transcription.git
   cd ai-meeting-transcription

2. Crea un entorno virtual e instala las dependencias:
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt

3. Crea un archivo .env en la raíz del proyecto con tu token de Hugging Face:
   HUGGINGFACE_AUTH_TOKEN=tu_token_aquí

## Uso

1. Inicia la interfaz web:
   python web-ui.py

2. Abre tu navegador en http://localhost:7860

3. Sube un archivo de video o ingresa una URL de YouTube

4. Configura los parámetros:
   - Selecciona el idioma (español, inglés o detección automática)
   - Elige el tamaño del modelo Whisper (los modelos más grandes son más precisos pero más lentos)
   - Ajusta el valor de collar para la diarización de hablantes
   - Activa/desactiva la generación de resumen

5. Haz clic en "Iniciar" para comenzar el procesamiento

6. Visualiza la transcripción y el resumen en las pestañas respectivas

## Modelos Utilizados

- Diarización de Hablantes: pyannote/speaker-diarization-3.0
- Reconocimiento de Voz: OpenAI Whisper (varios tamaños)
- Resumen: Algoritmo personalizado de resumen extractivo

## Archivos de Salida

- output.sub: Transcripción en formato SubViewer
- output_summary.txt: Resumen de la reunión con puntos clave
- output-tracks/: Directorio que contiene segmentos de audio para cada turno de hablante

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - consulta el archivo LICENSE para más detalles.

## Agradecimientos

- OpenAI Whisper (https://github.com/openai/whisper)
- Pyannote Audio (https://github.com/pyannote/pyannote-audio)
- Gradio (https://www.gradio.app/)
- PyTube (https://github.com/pytube/pytube)