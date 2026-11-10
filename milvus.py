from langchain_milvus import Milvus
from langchain_milvus.function import BM25BuiltInFunction
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class VideoVectorStore:
    """Управляет коллекцией Milvus для одного пользователя (tenant)"""

    def __init__(
        self,
        user_id: int,
        embeddings: HuggingFaceEmbeddings,
        host: str = "standalone",
        port: str = "19530",
    ):
        self.user_id = user_id
        self.collection_name = f"video_chunks_{user_id}"
        self.embeddings = embeddings
        self.connection_args = {"host": host, "port": port}

    def _get_store(self, drop_old: bool = False) -> Milvus:
        """
        Создаёт/подключается к коллекции с гибридной схемой
        При первом вызове (drop_old=False и коллекции нет) она создастся
        """
        bm25_func = BM25BuiltInFunction(
            input_field_names="text",
            output_field_names="sparse",
            # Для русского и английского текста используем стандартный анализатор
            # Можно указать analyzer_params={"type": "russian"} если нужно
        )

        return Milvus(
            embedding_function=self.embeddings,
            builtin_function=bm25_func,
            vector_field=["dense", "sparse"],  # два векторных поля
            collection_name=self.collection_name,
            connection_args=self.connection_args,
            auto_id=True,
            drop_old=drop_old,
            index_params=[
                {"metric_type": "IP", "index_type": "AUTOINDEX"},  # dense
                {"metric_type": "BM25", "index_type": "AUTOINDEX"},  # sparse
            ],
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Записывает чанки в Milvus
        chunks — список словарей с ключами text, timecode, start_sec, confidence
        """
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "timecode": c["timecode"],
                "start_sec": c["start_sec"],
                "confidence": c["confidence"],
            }
            for c in chunks
        ]

        store = self._get_store()
        store.add_texts(texts=texts, metadatas=metadatas)
        logger.info(f"Added {len(chunks)} chunks to {self.collection_name}")

    def search(self, question: str, k: int = 3) -> List[Document]:
        """
        Гибридный поиск (dense + sparse) с RRF-слиянием.
        Возвращает документы с метаданными (timecode и т.д.)
        """
        store = self._get_store()
        # similarity_search с указанием ранкера RRF включает гибридный поиск
        docs = store.similarity_search(
            question,
            k=k,
            ranker_type="rrf",  # Reciprocal Rank Fusion
        )
        return docs