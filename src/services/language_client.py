from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlparse

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from src.core.config import Settings
from src.core.errors import ProcessingError
from src.schemas.audio import (
    Assertion,
    AudioEvidence,
    AudioSegment,
    ClinicalMention,
    ClinicalRelation,
    LanguageStep,
    MentionType,
    RelationRole,
    SentimentSegment,
    SentimentStep,
)
from src.schemas.common import Locale


@dataclass(frozen=True)
class TextPiece:
    id: str
    segment_index: int
    offset: int
    text: str


@dataclass
class HealthcareResult:
    step: LanguageStep
    entities: list[ClinicalMention]
    relations: list[ClinicalRelation]


def create_language_client(settings: Settings) -> TextAnalyticsClient | None:
    if not settings.azure_language_endpoint or not settings.azure_language_key:
        return None
    try:
        endpoint = urlparse(settings.azure_language_endpoint)
        host = endpoint.hostname or ""
        valid = (
            endpoint.scheme == "https"
            and host.endswith(
                (".cognitiveservices.azure.com", ".api.cognitive.microsoft.com", ".services.ai.azure.com")
            )
            and endpoint.port in (None, 443)
            and not endpoint.username
            and not endpoint.password
            and endpoint.path in ("", "/")
            and not endpoint.query
            and not endpoint.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise ProcessingError("INVALID_AZURE_LANGUAGE_ENDPOINT", "O endpoint Azure Language é inválido.")
    return TextAnalyticsClient(
        endpoint=settings.azure_language_endpoint,
        credential=AzureKeyCredential(settings.azure_language_key.get_secret_value()),
        connection_timeout=settings.azure_request_timeout_seconds,
        read_timeout=settings.azure_request_timeout_seconds,
        retry_total=2,
        retry_backoff_factor=0.5,
    )


def _pieces(segments: list[AudioSegment], limit: int = 4_000) -> list[TextPiece]:
    pieces: list[TextPiece] = []
    for index, segment in enumerate(segments):
        position = 0
        while position < len(segment.text):
            end = min(position + limit, len(segment.text))
            if end < len(segment.text):
                last_space = segment.text.rfind(" ", position, end)
                if last_space > position:
                    end = last_space + 1
            text = segment.text[position:end]
            if text.strip():
                pieces.append(TextPiece(str(len(pieces)), index, position, text))
            position = end
    return pieces


def _batches(pieces: list[TextPiece]) -> Iterable[list[TextPiece]]:
    for index in range(0, len(pieces), 10):
        yield pieces[index : index + 10]


def _documents(pieces: list[TextPiece], language_code: str) -> list[dict[str, str]]:
    return [{"id": piece.id, "language": language_code, "text": piece.text} for piece in pieces]


def _unsupported(error: Any) -> bool:
    for _ in range(4):
        if error is None:
            break
        code = str(getattr(error, "code", "")).lower().replace("_", "")
        if any(
            term in code
            for term in (
                "unsupportedlanguage",
                "languagenotsupported",
                "unsupportedfeature",
                "featurenotsupported",
            )
        ):
            return True
        error = getattr(error, "innererror", None)
    return False


def _evidence(
    piece: TextPiece, segments: list[AudioSegment], offset: int | None = None, length: int | None = None
) -> AudioEvidence:
    segment = segments[piece.segment_index]
    return AudioEvidence(
        segment_index=piece.segment_index,
        quote=segment.text,
        start_seconds=segment.start_seconds,
        end_seconds=segment.end_seconds,
        text_offset=piece.offset + offset if offset is not None else None,
        text_length=length,
    )


def _assertion(entity: Any) -> Assertion | None:
    assertion = entity.assertion
    if assertion is None:
        return None
    return Assertion(
        certainty=assertion.certainty,
        conditionality=assertion.conditionality,
        association=assertion.association,
    )


def _mention(entity: Any, piece: TextPiece, segments: list[AudioSegment]) -> ClinicalMention:
    offset = entity.offset
    length = entity.length
    if (
        offset < 0
        or length <= 0
        or offset + length > len(piece.text)
        or piece.text[offset : offset + length] != entity.text
    ):
        raise ProcessingError("INVALID_LANGUAGE_EVIDENCE", "A evidência clínica retornada é inválida.")
    types: dict[str, MentionType] = {
        "SymptomOrSign": "symptom_mentioned",
        "Diagnosis": "condition_mentioned",
        "MedicationName": "medication_mentioned",
    }
    return ClinicalMention(
        type=types.get(entity.category, "clinical_mention"),
        text=entity.text,
        category=entity.category,
        normalized_text=entity.normalized_text,
        confidence=entity.confidence_score,
        assertion=_assertion(entity),
        evidence=_evidence(piece, segments, offset, length),
    )


def _relation(relation: Any, piece: TextPiece, segments: list[AudioSegment]) -> ClinicalRelation:
    roles: list[RelationRole] = []
    for role in relation.roles:
        entity = role.entity
        _mention(entity, piece, segments)
        roles.append(
            RelationRole(
                name=role.name,
                text=entity.text,
                category=entity.category,
                text_offset=piece.offset + entity.offset,
            )
        )
    return ClinicalRelation(
        type=relation.relation_type,
        roles=roles,
        confidence=relation.confidence_score,
        evidence=_evidence(piece, segments),
    )


def _check_result_ids(results: list[Any], pieces: list[TextPiece]) -> dict[str, TextPiece]:
    mapping = {piece.id: piece for piece in pieces}
    ids = [str(result.id) for result in results]
    if len(ids) != len(mapping) or set(ids) != set(mapping):
        raise ProcessingError("INVALID_LANGUAGE_RESPONSE", "A resposta do Azure Language está incompleta.")
    return mapping


def analyze_healthcare(
    segments: list[AudioSegment],
    language: Locale,
    client: TextAnalyticsClient | None,
    timeout: float,
) -> HealthcareResult:
    pieces = _pieces(segments)
    if not pieces:
        return HealthcareResult(
            LanguageStep(status="unavailable", reason="Não há transcrição para análise."), [], []
        )
    if client is None:
        return HealthcareResult(
            LanguageStep(status="unavailable", reason="Azure Language não está configurado."), [], []
        )

    entities: list[ClinicalMention] = []
    relations: list[ClinicalRelation] = []
    successful = 0
    unavailable = 0
    warned = False
    try:
        for batch in _batches(pieces):
            poller = client.begin_analyze_healthcare_entities(
                _documents(batch, "pt" if language == "pt-BR" else "en"),
                string_index_type="UnicodeCodePoint",
                disable_service_logs=True,
                polling_interval=2,
            )
            poller.wait(timeout)
            if not poller.done():
                poller.cancel()
                raise ProcessingError("AZURE_LANGUAGE_TIMEOUT", "O Azure Language excedeu o tempo limite.")
            results = list(poller.result())
            mapping = _check_result_ids(results, batch)
            for result in results:
                if result.is_error:
                    if _unsupported(result.error):
                        unavailable += 1
                        continue
                    raise ProcessingError("AZURE_LANGUAGE_ERROR", "O Azure Language não analisou a transcrição.")
                successful += 1
                warned = warned or bool(result.warnings)
                piece = mapping[str(result.id)]
                entities.extend(_mention(entity, piece, segments) for entity in result.entities)
                relations.extend(_relation(relation, piece, segments) for relation in result.entity_relations)
    except HttpResponseError as exc:
        if _unsupported(exc.error):
            return HealthcareResult(
                LanguageStep(
                    status="partial" if successful else "unavailable",
                    reason="Análise clínica não suportada pelo recurso ou idioma.",
                ),
                entities,
                relations,
            )
        raise
    if unavailable or warned:
        return HealthcareResult(
            LanguageStep(
                status="partial" if successful else "unavailable",
                reason="Análise clínica indisponível ou com avisos em parte da transcrição.",
            ),
            entities,
            relations,
        )
    return HealthcareResult(LanguageStep(status="available"), entities, relations)


def analyze_sentiment(
    segments: list[AudioSegment],
    language: Locale,
    client: TextAnalyticsClient | None,
    *,
    enabled: bool,
) -> SentimentStep:
    if not enabled:
        return SentimentStep(status="disabled", reason="Análise textual de sentimento desabilitada.")
    pieces = _pieces(segments)
    if not pieces:
        return SentimentStep(status="unavailable", reason="Não há transcrição para análise.")
    if client is None:
        return SentimentStep(status="unavailable", reason="Azure Language não está configurado.")
    findings: list[SentimentSegment] = []
    unavailable = 0
    warned = False
    try:
        for batch in _batches(pieces):
            results = client.analyze_sentiment(
                _documents(batch, "pt-BR" if language == "pt-BR" else "en"),
                string_index_type="UnicodeCodePoint",
                disable_service_logs=True,
            )
            mapping = _check_result_ids(results, batch)
            for result in results:
                if result.is_error:
                    if _unsupported(result.error):
                        unavailable += 1
                        continue
                    raise ProcessingError("AZURE_SENTIMENT_ERROR", "O serviço não analisou o sentimento textual.")
                warned = warned or bool(result.warnings)
                scores = result.confidence_scores
                findings.append(
                    SentimentSegment(
                        sentiment=result.sentiment,
                        confidence={
                            "positive": scores.positive,
                            "neutral": scores.neutral,
                            "negative": scores.negative,
                        },
                        evidence=_evidence(mapping[str(result.id)], segments),
                    )
                )
    except HttpResponseError as exc:
        if _unsupported(exc.error):
            return SentimentStep(
                status="partial" if findings else "unavailable",
                reason="Sentimento não suportado pelo recurso ou idioma.",
                segments=findings,
            )
        raise
    if unavailable or warned:
        return SentimentStep(
            status="partial" if findings else "unavailable",
            reason="Sentimento indisponível ou com avisos em parte da transcrição.",
            segments=findings,
        )
    return SentimentStep(status="available", segments=findings)
