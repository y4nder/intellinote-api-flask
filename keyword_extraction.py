import logging
from typing import List
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from app_types import Keyword
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeywordExtractor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading KeyBERT model on device: {device}...")
        
        embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device="cpu")  
        self.model = KeyBERT(model=embed_model)
        
        logger.info("KeyBERT model loaded successfully.")

    def extract_keywords(
        self,
        document: str,
        top_n: int = 10,
        ngram_range: tuple = (1, 2),
        use_mmr: bool = True,
        diversity: float = 0.7,
        show_progress: bool = True,
    ) -> List[Keyword]:
        logger.info("Extracting keywords from summarized text...")

        if not document.strip():
            logger.warning("Document is empty. Skipping keyword extraction.")
            return []

        try:
            with tqdm(total=1, desc="Extracting Keywords", disable=not show_progress) as pbar:
                results = self.model.extract_keywords(
                    document,
                    top_n=top_n,
                    keyphrase_ngram_range=ngram_range,
                    use_mmr=use_mmr,
                    diversity=diversity,
                    stop_words="english"
                )
                pbar.update(1)

            keywords = [Keyword(keyword=k, score=s) for k, s in results]
            logger.info(f"✅ Extracted {len(keywords)} keywords.")
            return keywords

        except Exception as e:
            logger.exception(f"❌ Keyword extraction failed: {e}")
            return []
