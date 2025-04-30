import logging
from typing import List
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from app_types import Keyword

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeywordExtractor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info("Loading KeyBERT model...")
        embed_model = SentenceTransformer(model_name, device="cpu")
        self.model = KeyBERT(model=embed_model)
        logger.info(f"keybert loaded successfully using")

    def extract_keywords(
        self,
        document: str,
        top_n: int = 6,
        ngram_range: tuple = (1, 1),
        use_mmr: bool = True,
        diversity: float = 0.7,
        show_progress: bool = True,
    ) -> List[Keyword]:
        logger.info("Starting keyword extraction...")
        try:
            with tqdm(total=1, desc="Processing Document", disable=not show_progress) as pbar:
                results = self.model.extract_keywords(
                    document,
                    top_n=top_n,
                    keyphrase_ngram_range=ngram_range,
                    use_mmr=use_mmr,
                    diversity=diversity,
                    stop_words="english",
                )
                pbar.update(1)
            keywords = [Keyword(keyword=k, score=s) for k, s in results]
            logger.info(f"Extracted {len(keywords)} keywords successfully.")
            return keywords

        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
