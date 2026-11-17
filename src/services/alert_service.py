import re

from src.schemas.audio import AudioAnalysis, AudioEvidence, AudioSegment, ClinicalMention, CriticalTerm
from src.schemas.common import AlertDraft
from src.schemas.prescription import PrescriptionAnalysis


def find_critical_terms(
    segments: list[AudioSegment], terms: list[str], entities: list[ClinicalMention]
) -> list[CriticalTerm]:
    findings: list[CriticalTerm] = []
    for index, segment in enumerate(segments):
        for term in terms:
            pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
            for match in re.finditer(pattern, segment.text, flags=re.IGNORECASE):
                matching_entity = next(
                    (
                        entity
                        for entity in entities
                        if entity.evidence.segment_index == index
                        and entity.evidence.text_offset == match.start()
                        and entity.evidence.text_length == len(match.group())
                    ),
                    None,
                )
                findings.append(
                    CriticalTerm(
                        text=match.group(),
                        assertion=matching_entity.assertion if matching_entity else None,
                        confidence=matching_entity.confidence if matching_entity else None,
                        evidence=AudioEvidence(
                            segment_index=index,
                            quote=segment.text,
                            start_seconds=segment.start_seconds,
                            end_seconds=segment.end_seconds,
                            text_offset=match.start(),
                            text_length=len(match.group()),
                        ),
                    )
                )
    return findings


def audio_alerts(analysis: AudioAnalysis) -> list[AlertDraft]:
    alerts: list[AlertDraft] = []
    for entity in analysis.clinical_entities:
        if entity.type in {"symptom_mentioned", "condition_mentioned"}:
            alerts.append(
                AlertDraft(
                    type=entity.type,
                    source="audio",
                    message="Menção para revisão; não indica presença ou diagnóstico.",
                    evidence=entity.model_dump(mode="json"),
                )
            )
    for term in analysis.relevant_terms:
        alerts.append(
            AlertDraft(
                type="critical_term",
                source="audio",
                message="Termo mencionado no contexto transcrito; verificar negação e incerteza.",
                evidence=term.model_dump(mode="json"),
            )
        )
    return alerts


def prescription_alerts(analysis: PrescriptionAnalysis) -> list[AlertDraft]:
    alerts = [
        AlertDraft(
            type="prescription_page_without_text",
            source="prescription",
            message="Página sem texto selecionável; revisão manual necessária.",
            evidence={"page": page, "quote": None},
        )
        for page in analysis.pages_without_text
    ]
    for medication in analysis.medications:
        if medication.uncertainties:
            alerts.append(
                AlertDraft(
                    type="prescription_uncertainty",
                    source="prescription",
                    message="Há campos ausentes ou incertos; revisar a prescrição original.",
                    evidence={
                        "page": medication.name_as_written.evidence.page,
                        "quote": medication.name_as_written.evidence.quote,
                        "uncertainties": medication.uncertainties,
                    },
                )
            )
    if analysis.status == "needs_review" and not alerts:
        alerts.append(
            AlertDraft(
                type="prescription_review_required",
                source="prescription",
                message="A análise da prescrição requer revisão manual.",
                evidence={"uncertainties": analysis.uncertainties},
            )
        )
    return alerts
