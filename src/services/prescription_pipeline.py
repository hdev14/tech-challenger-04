from pathlib import Path

from src.core.config import Settings
from src.core.errors import ProcessingError, ReviewRequired
from src.schemas.prescription import PrescriptionAnalysis
from src.services.pdf_extractor import PdfPage, classify_pdf_pages, extract_pdf_pages
from src.services.prescription_analyzer import analyze_prescription, create_azure_openai_client


def process_prescription(path: Path, settings: Settings) -> PrescriptionAnalysis:
    pages: list[PdfPage] = []
    try:
        pages = extract_pdf_pages(path, settings.max_pdf_text_chars)
        with_text, _ = classify_pdf_pages(pages)
        if not with_text:
            return analyze_prescription(pages, None, "", settings.pdf_model_chunk_chars)
        client = create_azure_openai_client(settings)
        try:
            if not settings.azure_openai_deployment:
                raise ProcessingError(
                    "AZURE_OPENAI_NOT_CONFIGURED", "Azure OpenAI não está configurado."
                )
            return analyze_prescription(
                pages, client, settings.azure_openai_deployment, settings.pdf_model_chunk_chars
            )
        finally:
            client.close()
    except ReviewRequired as exc:
        with_text, without_text = classify_pdf_pages(pages)
        return PrescriptionAnalysis(
            status="needs_review",
            summary="",
            pages_with_text=with_text,
            pages_without_text=without_text,
            uncertainties=[exc.message],
        )
