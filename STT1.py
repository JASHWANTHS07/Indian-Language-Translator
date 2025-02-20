import torch
import librosa
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


class AudioTranscriber:
    def __init__(self, model_name="openai/whisper-large-v2", device=None):
        """
        Initialize the AudioTranscriber with specified model.

        Args:
            model_name (str): Name or path of the Whisper model to use
            device (str): Device to run the model on ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        # self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        # Initialize processor and model
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Whisper model and processor."""
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name)
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _load_audio(self, file_path, sample_rate=16000):
        """
        Load and preprocess audio file.

        Args:
            file_path (str): Path to the audio file
            sample_rate (int): Target sample rate for the audio

        Returns:
            numpy.ndarray: Loaded and resampled audio data
        """
        try:
            audio, _ = librosa.load(file_path, sr=sample_rate)
            return audio
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {str(e)}")

    def transcribe(self, audio_path, return_timestamps=False):
        """
        Transcribe audio file to text.

        Args:
            audio_path (str): Path to the audio file
            return_timestamps (bool): Whether to return timestamps with the transcription

        Returns:
            str: Transcribed text
        """
        try:
            # Load and preprocess audio
            audio = self._load_audio(audio_path)

            # Prepare input features
            inputs = self.processor(
                audio,
                return_tensors="pt",
                sampling_rate=16000
            ).to(self.device)

            # Generate transcription
            with torch.no_grad():
                if return_timestamps:
                    generated = self.model.generate(
                        inputs.input_features,
                        return_timestamps=True
                    )
                else:
                    generated = self.model.generate(
                        inputs.input_features
                    )

            # Decode the output
            transcription = self.processor.batch_decode(
                generated,
                skip_special_tokens=True
            )[0]

            return transcription

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}")

    def transcribe_batch(self, audio_paths):
        """
        Transcribe multiple audio files.

        Args:
            audio_paths (list): List of paths to audio files

        Returns:
            list: List of transcribed texts
        """
        return [self.transcribe(path) for path in audio_paths]

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        try:
            del self.model
            del self.processor
            torch.cuda.empty_cache()
        except:
            pass