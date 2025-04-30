import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PegasusSummarizer:
    def __init__(self, model_name: str = 'tuner007/pegasus_summarizer'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.device)
        logger.info(f"Model '{model_name}' loaded successfully.")

    def summarize(self, text: str, max_input_length: int = 1024, max_summary_length: int = 128,
                  num_beams: int = 8, temperature: float = 1.0) -> str:
        if not text.strip():
            logger.warning("Input text is empty.")
            return ""

        logger.info("Tokenizing input...")
        inputs = self.tokenizer(
            [text],
            truncation=True,
            padding='longest',
            max_length=max_input_length,
            return_tensors="pt"
        ).to(self.device)   

        logger.info("Generating summary...")
        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                max_length=max_summary_length,
                num_beams=num_beams,
                temperature=temperature,
                num_return_sequences=1,
                length_penalty=2.0,
                early_stopping=True
            )

        summary = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
        logger.info("Summary generation complete.")
        return summary

