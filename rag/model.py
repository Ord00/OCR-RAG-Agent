"""
RAG over video subtitles.
  embeddings : BAAI/bge-m3 (local, sentence-transformers)
  vector DB  : Chroma (persistent, on disk)
  LLM        : local via Ollama (default) or any API provider, chosen by LLM_MODEL
"""
from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda")  # "cpu" if VRAM is tight
# "ollama:<tag>" = local; "qwen:<model>" = Qwen API; "anthropic:<model>", "openai:<model>" = other APIs
LLM_MODEL = os.getenv("LLM_MODEL", "ollama:qwen3.5:9b")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHUNK_SECONDS = float(os.getenv("CHUNK_SECONDS", "40"))  # max video time per chunk
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "800"))  # max text length per chunk
OVERLAP_LINES = int(os.getenv("OVERLAP_LINES", "1"))  # subtitle lines shared by neighbours
TOP_K = int(os.getenv("TOP_K", "5"))


def clean_lines(lines: list[dict]) -> list[dict]:
    """
    lines: [{"start": sec, "end": sec, "text": str}, ...]
    OCR reads the same subtitle on many consecutive frames (often with small
    differences), so merge near-identical neighbours into one line with a longer time span.
    """
    out: list[dict] = []
    for l in sorted(lines, key=lambda x: x["start"]):
        text = re.sub(r"\s+", " ", l["text"]).strip()
        if not text:
            continue
        if out:
            prev = out[-1]
            similar = SequenceMatcher(None, prev["text"].lower(), text.lower()).ratio() >= 0.9
            if similar and l["start"] - prev["end"] <= 2.0:
                prev["end"] = max(prev["end"], float(l["end"]))
                if len(text) > len(prev["text"]):  # keep the fuller OCR variant
                    prev["text"] = text
                continue
        out.append({"start": float(l["start"]), "end": float(l["end"]), "text": text})
    return out


def make_chunks(video_id: str, lines: list[dict]) -> list[Document]:
    """Group consecutive subtitle lines; every chunk remembers its start/end time."""
    docs: list[Document] = []
    i = 0
    while i < len(lines):
        j, chars = i, 0
        while j < len(lines):
            chars += len(lines[j]["text"]) + 1
            too_long = lines[j]["end"] - lines[i]["start"] > CHUNK_SECONDS or chars > CHUNK_CHARS
            if j > i and too_long:
                break
            j += 1
        group = lines[i:j]
        docs.append(
            Document(
                page_content=" ".join(g["text"] for g in group),
                metadata={
                    "video_id": video_id,
                    "start": group[0]["start"],
                    "end": group[-1]["end"],
                    "chunk": len(docs),
                    # same text with a timestamp on every line; shown to the LLM, not embedded
                    "timed_text": "\n".join(f"[{_ts(g['start'])}] {g['text']}" for g in group),
                },
            )
        )
        if j >= len(lines):
            break
        i = max(j - OVERLAP_LINES, i + 1)
    return docs


# ----------------------------- 3. embeddings + vector store -----------------------------
@lru_cache(maxsize=1)
def get_embeddings():
    # imported lazily so the module can be imported without torch (tests, API-only setups)
    from langchain_huggingface import HuggingFaceEmbeddings

    st_kwargs: dict = {"device": EMBED_DEVICE}
    if EMBED_DEVICE.startswith("cuda"):
        import torch

        # fp16 halves VRAM (~1.2 GB instead of ~2.3 GB); Turing GPUs support fp16, not bf16
        st_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs=st_kwargs,
        # bge-m3 needs no query/passage prefixes; normalized vectors -> cosine-equivalent ranking
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )


@lru_cache(maxsize=1)
def get_store() -> Chroma:
    return Chroma(
        collection_name="subtitles",
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def index_video(video_id: str, lines: list[dict]) -> int:
    """(Re)index one video. Returns the number of chunks stored."""
    store = get_store()
    store.delete(where={"video_id": video_id})  # safe re-indexing
    docs = make_chunks(video_id, clean_lines(lines))
    if docs:
        store.add_documents(docs, ids=[f"{video_id}:{d.metadata['chunk']}" for d in docs])
    return len(docs)


# ----------------------------- 4. LLM + answer chain -----------------------------
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты помощник, который отвечает на вопросы по видео. Ниже даны фрагменты субтитров. "
            "Каждая строка начинается с таймкода [мм:сс] — момента, когда она появилась в видео. "
            "Субтитры получены распознаванием текста (OCR), поэтому в них возможны "
            "опечатки и обрывки слов.\n\n"
            "Правила:\n"
            "1. Отвечай только на основе фрагментов. Не добавляй сведений из своих знаний.\n"
            "2. Если во фрагментах нет ответа, напиши: «В субтитрах этого видео нет информации "
            "по этому вопросу». Если есть хотя бы частичный ответ, дай его, а не отказывай.\n"
            "3. Отвечай на языке вопроса. Пиши только итоговый ответ: без черновиков, "
            "примечаний и исправлений.\n"
            "4. В конце ответа укажи таймкоды строк, на которых основан ответ: скопируй их из "
            "фрагментов без изменений, в формате [мм:сс]. Никогда не придумывай таймкоды.\n\n"
            "Фрагменты субтитров:\n{context}\n\n"
        ),
        ("human", "{question}"),
    ]
)


@lru_cache(maxsize=1)
def get_llm():
    from langchain.chat_models import init_chat_model

    if LLM_MODEL.startswith("ollama:"):
        # num_ctx caps the KV cache so the model stays inside 8 GB VRAM;
        # reasoning=False turns off Qwen "thinking" (much faster, enough for RAG)
        return init_chat_model(LLM_MODEL, temperature=0.1, num_ctx=4096, reasoning=False)
    if LLM_MODEL.startswith("qwen:"):
        # Qwen via Alibaba Cloud Model Studio (OpenAI-compatible endpoint), e.g. "qwen:qwen-plus"
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=LLM_MODEL.split(":", 1)[1],
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.getenv(
                "QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            ),
            temperature=0.1,
        )
    return init_chat_model(LLM_MODEL, temperature=0.1)  # other API providers


def _ts(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def ask(video_id: str, question: str, k: int = TOP_K) -> dict:
    docs = get_store().similarity_search(question, k=k, filter={"video_id": video_id})
    if not docs:
        return {"answer": "Это видео ещё не проиндексировано.", "sources": []}

    docs.sort(key=lambda d: d.metadata["start"])  # chronological order for the LLM
    context = "\n\n".join(d.metadata.get("timed_text", d.page_content) for d in docs)
    answer = (PROMPT | get_llm() | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    return {
        "answer": answer.strip(),
        "sources": [
            {"start": d.metadata["start"], "end": d.metadata["end"], "text": d.page_content}
            for d in docs
        ],
    }
