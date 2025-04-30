from dataclasses import asdict
import logging
from app_types import GeneratedResponse
from keyword_extraction import KeywordExtractor
from summarizer import PegasusSummarizer

logger = logging.getLogger(__name__)


def run_generate(document: str, extractor: KeywordExtractor, summarizer: PegasusSummarizer):
    keywords = extractor.extract_keywords(document=document)
    keyword_dicts = [asdict(k) for k in keywords]
    summarization = summarize_large_document(document=document, summarizer=summarizer)
    return GeneratedResponse(keywords=keyword_dicts, summary=summarization)


def chunk_text(document: str, tokenizer, max_tokens=1024):
    sentences = document.split(". ")  
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        prospective_chunk = current_chunk + " " + sentence if current_chunk else sentence
        tokenized = tokenizer(prospective_chunk, return_tensors="pt", truncation=False)
        num_tokens = tokenized.input_ids.shape[1]

        if num_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = prospective_chunk

    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def summarize_large_document(document: str, summarizer: PegasusSummarizer, max_input_length: int = 1024, max_summary_length: int = 128):
    """
    Summarizes a large document by chunking it into smaller parts and summarizing each chunk.
    Then, it summarizes the concatenation of those summaries to get the final result.
    """
    logger.info("Starting the document summarization process.")

    # Step 1: Split the text into chunks
    logger.info("Splitting the document into chunks...")
    chunks = chunk_text(document, summarizer.tokenizer, max_input_length)
    logger.info(f"Document split into {len(chunks)} chunks.")

    # Step 2: Summarize each chunk
    summaries = []
    logger.info("Summarizing each chunk...")
    for i, chunk in enumerate(chunks):
        logger.info(f"Summarizing chunk {i + 1}/{len(chunks)}...")
        summary = summarizer.summarize(chunk, max_input_length=max_input_length, max_summary_length=max_summary_length)
        summaries.append(summary)
        logger.info(f"Chunk {i + 1} summarized: {summary}")  # Log the first 100 characters of the summary

    # Step 3: Concatenate all chunk summaries into a final text
    final_summary_input = " ".join(summaries)
    logger.info(f"Concatenated chunk summaries into one text of length {len(final_summary_input)} characters.")

    # Step 4: Summarize the concatenated summaries (optional, for hierarchical summarization)
    logger.info("Summarizing the concatenated chunk summaries...")
    final_summary = summarizer.summarize(final_summary_input, max_input_length=max_input_length, max_summary_length=max_summary_length)
    logger.info("Final summary generated.")

    return final_summary