from dataclasses import asdict
import logging

from openai import OpenAI
from app_types import GeneratedResponse, Keyword
from keyword_extraction import KeywordExtractor
from summarizer import PegasusSummarizer
from dotenv import load_dotenv
import os


load_dotenv(override=True)

logger = logging.getLogger(__name__)

API_KEY = os.getenv("OPEN_AI_API_KEY")
client = OpenAI(api_key=API_KEY)


def run_generate(document: str, extractor: KeywordExtractor, summarizer: PegasusSummarizer):
    summarization = summarize_large_document(document=document, summarizer=summarizer)
    keywords = extractor.extract_keywords(document=summarization + " " + document)
    keyword_dicts = [asdict(k) for k in keywords]
    generated_topics = generate_topics(keywords=keywords, summary=summarization)
    return GeneratedResponse(keywords=keyword_dicts, summary=summarization, topics=generated_topics)


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

def generate_topics(keywords: list[Keyword], summary: str) -> list[str]:
    """
    Generates concise topics based on the keywords and summary.
    """
    logger.info("Generating concise topics from keywords and summary...")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that generates concise topics. "
                        "Return a list of topics delimited by new lines, without bullet points or dashes."
                    )
                },
                {
                    "role": "user",
                    "content": f"Keywords: {[k.keyword for k in keywords]}\nSummary: {summary}"
                }
            ],
            max_tokens=100,
            temperature=0.7
        )
        raw_topics = response.choices[0].message.content.strip().split("\n")
        topics = [t.lstrip("-• ").strip() for t in raw_topics if t.strip()]
        logger.info(f"Generated {len(topics)} concise topics.")
        return topics
    except Exception as e:
        logger.error(f"OpenAI topic generation failed: {e}")
        return []


def summarize_with_openai(text: str, model: str = "gpt-3.5-turbo", max_tokens: int = 300) -> str:
    logger.info("Sending text to OpenAI for final summarization...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        summary = response.choices[0].message.content.strip()
        logger.info("OpenAI summarization successful.")
        return summary
    except Exception as e:
        logger.error(f"OpenAI summarization failed: {e}")
        return ""

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
    final_summary = summarize_with_openai(final_summary_input)
    logger.info("Final summary generated.")

    return final_summary

