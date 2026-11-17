from pathlib import Path

from src.core.config import Settings
from src.schemas.audio import AudioAnalysis, LanguageStep
from src.schemas.common import Locale
from src.services.alert_service import find_critical_terms
from src.services.language_client import (
    analyze_healthcare,
    analyze_sentiment,
    create_language_client,
)
from src.services.speech_client import transcribe_audio


def process_audio(path: Path, language: Locale, settings: Settings) -> AudioAnalysis:
    segments = transcribe_audio(path, language, settings)
    client = create_language_client(settings)
    try:
        healthcare = analyze_healthcare(
            segments, language, client, settings.azure_request_timeout_seconds
        )
        sentiment = analyze_sentiment(
            segments, language, client, enabled=settings.sentiment_enabled
        )
    finally:
        if client is not None:
            client.close()

    uncertainties: list[str] = []
    if not segments:
        uncertainties.append("Nenhuma fala foi reconhecida no áudio.")
    if healthcare.step.status != "available":
        uncertainties.append("A análise de menções clínicas não está completa.")
    if sentiment.status in {"unavailable", "partial"}:
        uncertainties.append("A análise textual de sentimento não está completa.")
    return AudioAnalysis(
        status="needs_review" if uncertainties else "completed",
        language=language,
        transcript=" ".join(segment.text for segment in segments),
        segments=segments,
        clinical_analysis=healthcare.step,
        clinical_entities=healthcare.entities,
        relations=healthcare.relations,
        relevant_terms=find_critical_terms(segments, settings.critical_terms, healthcare.entities),
        text_sentiment=sentiment,
        acoustic_features=LanguageStep(
            status="disabled", reason="Indicadores acústicos não foram implementados."
        ),
        uncertainties=uncertainties,
    )
