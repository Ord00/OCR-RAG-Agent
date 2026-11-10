import cv2
import easyocr
import hashlib
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class OcrChunk:
    text: str
    timecode: str  # "MM:SS"
    start_sec: float
    confidence: float


def extract_text_from_video(
        video_path: str,
        ocr_reader: easyocr.Reader,
        sample_interval_sec: float = 1.0,  # как часто сэмплировать кадры
        min_confidence: float = 0.5,  # отбрасывать ненадёжные распознавания
        dedup_threshold: float = 0.8,  # насколько похожим должен быть текст, чтобы считать дубликатом
) -> List[OcrChunk]:
    """
    Извлекает уникальные текстовые фрагменты из видео.

    Логика:
    1. Читаем видео через OpenCV
    2. Каждые `sample_interval_sec` секунд берём кадр
    3. Прогоняем через EasyOCR, фильтруем по confidence
    4. Сравниваем с предыдущим сохранённым текстом: если похож — пропускаем (это тот же слайд)
    5. Сохраняем уникальные фрагменты с таймкодом
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    chunks: List[OcrChunk] = []
    last_saved_text = ""

    # Шаг в кадрах, соответствующий sample_interval_sec
    frame_step = max(1, int(fps * sample_interval_sec))
    current_frame = 0

    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        # EasyOCR ожидает RGB, OpenCV даёт BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # readtext возвращает список кортежей: (bbox, text, confidence)
        results = ocr_reader.readtext(rgb_frame)

        # Собираем весь текст с кадра, отфильтровав по уверенности
        frame_texts = []
        frame_confidences = []
        for (bbox, text, conf) in results:
            if conf >= min_confidence and text.strip():
                frame_texts.append(text.strip())
                frame_confidences.append(conf)

        if frame_texts:
            combined_text = " ".join(frame_texts)
            avg_conf = sum(frame_confidences) / len(frame_confidences)

            # Дедупликация: если текст очень похож на предыдущий сохранённый — пропускаем
            if not _is_duplicate(combined_text, last_saved_text, dedup_threshold):
                time_sec = current_frame / fps
                timecode = _format_timecode(time_sec)
                chunks.append(OcrChunk(
                    text=combined_text,
                    timecode=timecode,
                    start_sec=time_sec,
                    confidence=avg_conf,
                ))
                last_saved_text = combined_text

        current_frame += frame_step

    cap.release()
    return chunks


def _is_duplicate(new_text: str, last_text: str, threshold: float) -> bool:
    """Простое сравнение: если новый текст содержит старый или они очень похожи — дубликат"""
    if not last_text:
        return False
    # Нормализуем: убираем лишние пробелы, приводим к нижнему регистру
    a = " ".join(new_text.lower().split())
    b = " ".join(last_text.lower().split())
    if a == b:
        return True
    # Если один текст является подстрокой другого (типично для слайдов с добавлением строк)
    if a in b or b in a:
        return True
    # Можно добавить difflib.SequenceMatcher для более гибкого сравнения
    import difflib
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio >= threshold


def _format_timecode(seconds: float) -> str:
    """Преобразует секунды в 'MM:SS'."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"