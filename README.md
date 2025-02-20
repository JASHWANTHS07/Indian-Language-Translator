# Indian Language Translation System

## Overview
The **Indian Language Translation System** is a pipeline that enables real-time speech-to-text conversion, machine translation, and text-to-speech synthesis for multiple Indian languages. This project leverages **Whisper (for STT)**, **IndicTrans (for translation)**, and **gTTS (for TTS)** to achieve seamless multilingual communication.

## Features
- **Speech-to-Text (STT)**: Converts spoken words into text using OpenAI Whisper.
- **Machine Translation (MT)**: Translates extracted text into various Indian languages using IndicTrans.
- **Text-to-Speech (TTS)**: Converts translated text into speech using gTTS.
- **Real-time Processing**: Optimized for efficient execution on Jetson Nano and local machines.
- **Modular Design**: The system is structured into separate classes for STT, MT, and TTS.

## Installation
Ensure you have **Python 3.8+** installed. Run the following commands to install the necessary dependencies:

```bash
pip install sounddevice scipy transformers librosa soundfile indic-nlp-library gTTS
```

### Clone IndicTrans for Machine Translation
```bash
git clone https://github.com/VarunGumma/IndicTransToolkit
cd IndicTransToolkit
pip install --editable ./
```

## Usage
### Speech-to-Text (STT)
Use the `AudioTranscriber` class to convert an audio file into text.

```python
from AudioTranscriber import AudioTranscriber

transcriber = AudioTranscriber(model_name="openai/whisper-large-v2")
text = transcriber.transcribe("audio_sample.wav")
print("Transcribed Text:", text)
```

### Machine Translation (MT)
Use the `MachineTranslator` class to translate text between supported languages.

```python
from MachineTranslator import MachineTranslator

translator = MachineTranslator()
translated_text = translator.translate("Hello, how are you?", src_lang="en", tgt_lang="hi")
print("Translated Text:", translated_text)
```

### Text-to-Speech (TTS)
Use the `TextToSpeech` class to convert text into speech.

```python
from TextToSpeech import TextToSpeech

tts = TextToSpeech()
tts.speak("\u0928\u092e\u0938\u094d\u0924\u0947, \u0906\u092a \u0915\u0948\u0938\u0947 \u0939\u0948\u0902?", lang="hi")
```

## Project Structure
```
Indian_Language_Translation_System/
│── IndicTransToolkit/       # Cloned IndicTrans for translation
│── models/                  # Model files for STT, MT, and TTS
│── scripts/
│   ├── AudioTranscriber.py  # Speech-to-Text module
│   ├── MachineTranslator.py # Text Translation module
│   ├── TextToSpeech.py      # Text-to-Speech module
│── data/                    # Sample audio and text data
│── app.py                   # Main execution script
│── requirements.txt         # Dependencies list
│── README.md                # Documentation
```

## Model Details
- **Speech-to-Text:** Whisper (Large-v2)
- **Machine Translation:** IndicTrans Toolkit (Fine-tuned on Indian languages)
- **Text-to-Speech:** gTTS (Google Text-to-Speech API)

## Future Improvements
- Support for more Indian languages.
- Optimization for real-time performance on Jetson Nano.
- Improve TTS naturalness with alternatives like VITS.
- Integrate with a web-based UI for accessibility.

## Contributions
Contributions are welcome! Feel free to fork the repository, create an issue, or submit a pull request.

## License
This project is open-source and licensed under the **MIT License**.

