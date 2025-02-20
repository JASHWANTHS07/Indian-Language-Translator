import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from IndicTransToolkit import IndicProcessor


class IndicTranslator:
    def __init__(self, checkpoint_dir="ai4bharat/indictrans2-indic-indic-1B", batch_size=4, quantization=None):
        """
        Initialize the IndicTranslator with model configuration.

        Args:
            checkpoint_dir (str): Path to the model checkpoint
            batch_size (int): Size of batches for translation
            quantization (str): Quantization type ('4-bit', '8-bit', or None)
        """
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.processor = IndicProcessor(inference=True)
        self.tokenizer, self.model = self._initialize_model_and_tokenizer(quantization)

    def _initialize_model_and_tokenizer(self, quantization):
        """Initialize and configure the model and tokenizer."""
        if quantization == "4-bit":
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8-bit":
            qconfig = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
        else:
            qconfig = None

        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.checkpoint_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=qconfig,
        )

        if qconfig is None:
            model = model.to(self.device)
            if self.device == "cuda":
                model.half()

        model.eval()
        return tokenizer, model

    def translate(self, texts, source_lang, target_lang):
        """
        Translate a list of texts from source language to target language.

        Args:
            texts (list): List of input texts to translate
            source_lang (str): Source language code (e.g., 'tam_Taml')
            target_lang (str): Target language code (e.g., 'hin_Deva')

        Returns:
            list: List of translated texts
        """
        translations = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]

            # Preprocess the batch
            batch = self.processor.preprocess_batch(batch, src_lang=source_lang, tgt_lang=target_lang)

            # Tokenize
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)

            # Generate translations
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )

            # Decode translations
            with self.tokenizer.as_target_tokenizer():
                generated_tokens = self.tokenizer.batch_decode(
                    generated_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

            # Postprocess translations
            translations.extend(self.processor.postprocess_batch(generated_tokens, lang=target_lang))

            del inputs
            torch.cuda.empty_cache()

        return translations

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        try:
            del self.tokenizer
            del self.model
            torch.cuda.empty_cache()
        except:
            pass