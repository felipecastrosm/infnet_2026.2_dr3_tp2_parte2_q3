import re
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_huggingface import HuggingFacePipeline

MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"
BATCH_SIZE = 8
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

prompt = PromptTemplate.from_template("{text}")


@lru_cache
def get_chain() -> Runnable:
    llm = HuggingFacePipeline.from_model_id(
        model_id=MODEL_NAME,
        task="translation",
        batch_size=BATCH_SIZE,
        pipeline_kwargs={"truncation": True},
    )
    return prompt | llm | StrOutputParser()


def split_into_sentences(paragraph: str) -> list[str]:
    return [sentence for sentence in SENTENCE_BOUNDARY.split(paragraph.strip()) if sentence]


def translate(text: str) -> str:
    paragraphs = [split_into_sentences(line) for line in text.splitlines() if line.strip()]
    sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
    results = iter(get_chain().batch([{"text": sentence} for sentence in sentences]))
    return "\n".join(" ".join(next(results) for _ in paragraph) for paragraph in paragraphs)
