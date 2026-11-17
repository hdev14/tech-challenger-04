import hashlib
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from fastapi import UploadFile


class InvalidUpload(ValueError):
    pass


class UploadTooLarge(InvalidUpload):
    pass


@dataclass(frozen=True)
class StoredUpload:
    path: Path
    sha256: str
    size_bytes: int


async def store_upload(
    upload: UploadFile, directory: Path, max_bytes: int, suffix: str
) -> StoredUpload:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, filename = tempfile.mkstemp(prefix="upload-", suffix=suffix, dir=directory)
    path = Path(filename)
    hasher = hashlib.sha256()
    size = 0
    success = False
    try:
        with os.fdopen(descriptor, "wb") as target:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise UploadTooLarge("O arquivo excede o limite configurado.")
                hasher.update(chunk)
                target.write(chunk)
        if size == 0:
            raise InvalidUpload("O arquivo está vazio.")
        success = True
        return StoredUpload(path=path, sha256=hasher.hexdigest(), size_bytes=size)
    finally:
        try:
            await upload.close()
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            if not success:
                path.unlink(missing_ok=True)


def validate_audio(path: Path, max_duration_seconds: float) -> dict[str, float | int]:
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            if (
                audio.getcomptype() != "NONE"
                or channels != 1
                or sample_width != 2
                or sample_rate not in (8_000, 16_000)
            ):
                raise InvalidUpload("Use WAV PCM mono de 16 bits, com 8 ou 16 kHz.")
            if frame_count == 0:
                raise InvalidUpload("O áudio não contém amostras.")
            duration = frame_count / sample_rate
            if duration > max_duration_seconds:
                raise InvalidUpload("O áudio excede a duração máxima configurada.")
            remaining = frame_count
            while remaining:
                frames_to_read = min(remaining, 32_768)
                data = audio.readframes(frames_to_read)
                if len(data) != frames_to_read * channels * sample_width:
                    raise InvalidUpload("O arquivo de áudio está incompleto.")
                remaining -= frames_to_read
    except (wave.Error, EOFError) as exc:
        raise InvalidUpload("O arquivo não é um WAV válido.") from exc
    return {"duration_seconds": duration, "sample_rate": sample_rate}


def validate_pdf(path: Path, max_pages: int) -> dict[str, int]:
    try:
        with pymupdf.open(path) as document:
            if not document.is_pdf:
                raise InvalidUpload("O arquivo não é um PDF válido.")
            if document.needs_pass or document.is_encrypted:
                raise InvalidUpload("PDFs criptografados não são aceitos.")
            pages = document.page_count
            if pages == 0:
                raise InvalidUpload("O PDF não contém páginas.")
            if pages > max_pages:
                raise InvalidUpload("O PDF excede o limite de páginas configurado.")
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
        raise InvalidUpload("O arquivo não é um PDF válido.") from exc
    return {"page_count": pages}
