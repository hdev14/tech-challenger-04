import asyncio
import io
import wave

import pymupdf
import pytest
from fastapi import UploadFile

from src.core.errors import ReviewRequired
from src.services.pdf_extractor import (
    PdfPage,
    chunk_pdf_pages,
    classify_pdf_pages,
    extract_pdf_pages,
    normalize_whitespace,
)
from src.services.upload import InvalidUpload, UploadTooLarge, store_upload, validate_audio, validate_pdf


def make_wav(path, *, rate=16_000, channels=1, frames=16_000):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\0\0" * frames * channels)


def make_pdf(path, texts):
    document = pymupdf.open()
    for text in texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_streamed_upload_enforces_limit_and_cleans_up(tmp_path):
    upload = UploadFile(file=io.BytesIO(b"abcdef"), filename="audio.wav")
    with pytest.raises(UploadTooLarge):
        asyncio.run(store_upload(upload, tmp_path, 3, ".wav"))
    assert list(tmp_path.iterdir()) == []

    upload = UploadFile(file=io.BytesIO(b"abc"), filename="audio.wav")
    stored = asyncio.run(store_upload(upload, tmp_path, 3, ".wav"))
    assert stored.size_bytes == 3
    assert stored.sha256 == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert stored.path.read_bytes() == b"abc"


def test_audio_validation_rejects_bad_formats_duration_and_truncated_files(tmp_path):
    valid = tmp_path / "valid.wav"
    make_wav(valid)
    assert validate_audio(valid, 1)["duration_seconds"] == 1
    with pytest.raises(InvalidUpload):
        validate_audio(valid, 0.9)
    invalid = tmp_path / "invalid.wav"
    make_wav(invalid, rate=44_100, frames=44_100)
    with pytest.raises(InvalidUpload):
        validate_audio(invalid, 10)
    truncated = tmp_path / "truncated.wav"
    make_wav(truncated)
    truncated.write_bytes(truncated.read_bytes()[:-100])
    with pytest.raises(InvalidUpload):
        validate_audio(truncated, 10)


def test_pdf_extraction_preserves_pages_and_marks_empty_pages(tmp_path):
    path = tmp_path / "example.pdf"
    make_pdf(path, ["Prescrição: não alterar Med-A 10,5 mg.", "", "Med-B 20 mg"])
    assert validate_pdf(path, 3) == {"page_count": 3}
    pages = extract_pdf_pages(path, 1_000)
    assert [page.page for page in pages] == [1, 2, 3]
    assert "Prescrição: não alterar Med-A 10,5 mg." in pages[0].text
    assert classify_pdf_pages(pages) == ([1, 3], [2])
    with pytest.raises(InvalidUpload):
        validate_pdf(path, 2)
    with pytest.raises(ReviewRequired):
        extract_pdf_pages(path, 5)


def test_pdf_rejects_encrypted_and_invalid_files(tmp_path):
    invalid = tmp_path / "invalid.pdf"
    invalid.write_bytes(b"not a pdf")
    with pytest.raises(InvalidUpload):
        validate_pdf(invalid, 10)
    encrypted = tmp_path / "encrypted.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(
        encrypted,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    document.close()
    with pytest.raises(InvalidUpload):
        validate_pdf(encrypted, 10)


def test_pdf_chunking_preserves_all_text_and_page_provenance():
    pages = [PdfPage(1, "Primeira linha\ncom 10,5 mg e não alterar."), PdfPage(2, "Second page.")]
    chunks = chunk_pdf_pages(pages, 15)
    assert all(sum(len(item.text) for item in chunk) <= 15 for chunk in chunks)
    for page in pages:
        rebuilt = "".join(item.text for chunk in chunks for item in chunk if item.page == page.page)
        assert rebuilt == normalize_whitespace(page.text)
