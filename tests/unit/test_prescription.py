import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.core.config import Settings
from src.core.errors import ProcessingError, ReviewRequired
from src.schemas.common import Evidence
from src.schemas.prescription import ModelPrescription
from src.services.pdf_extractor import PdfPage
from src.services.prescription_analyzer import analyze_prescription, create_azure_openai_client
from src.services.prescription_pipeline import process_prescription


def field(value, quote, page=1):
    return {"value": value, "evidence": {"page": page, "quote": quote}}


def response(payload):
    return SimpleNamespace(status="completed", output_text=json.dumps(payload, ensure_ascii=False))


@pytest.mark.parametrize(
    ("text", "name", "dose"),
    [
        ("Med-A 10 mg, via oral, uma vez ao dia.", "Med-A", "10 mg"),
        ("Med-B 20 mg, oral, once daily.", "Med-B", "20 mg"),
    ],
)
def test_analyzer_preserves_original_fields_and_evidence(text, name, dose):
    payload = {
        "summary_items": [field(name, text)],
        "medications": [
            {
                "name_as_written": field(name, text),
                "normalized_name": field(name.lower(), text),
                "dose": field(dose, text),
            }
        ],
        "uncertainties": [],
    }
    client = MagicMock()
    client.responses.create.return_value = response(payload)
    analysis = analyze_prescription([PdfPage(1, text)], client, "deployment", 1_000)

    assert analysis.summary == name
    medication = analysis.medications[0]
    assert medication.name_as_written.value == name
    assert medication.dose.evidence.quote == text
    assert any("frequency" in item for item in medication.uncertainties)
    call = client.responses.create.call_args.kwargs
    assert call["model"] == "deployment"
    assert json.loads(call["input"])["pages"][0]["text"] == text
    assert "json" in call["input"].lower()
    assert call["tools"] == []
    assert call["store"] is False
    assert "untrusted" in call["instructions"]


@pytest.mark.parametrize(
    "payload",
    [
        {"summary_items": [], "medications": []},
        {"summary_items": [{"value": "Med-A"}], "medications": []},
        {"summary_items": [field("Med-A", "Med-A", 0)], "medications": []},
        {"summary_items": [field("Med-A", " ")], "medications": []},
    ],
)
def test_model_schema_rejects_missing_or_invalid_evidence(payload):
    with pytest.raises(ValidationError):
        ModelPrescription.model_validate(payload)
    with pytest.raises(ValidationError):
        Evidence(page=-1, quote="x")


@pytest.mark.parametrize(
    "payload",
    [
        {"summary_items": [field("Med-A", "invented quote")], "medications": []},
        {"summary_items": [field("Med-A", "Med-A", 2)], "medications": []},
        {"summary_items": [field("invented value", "Med-A")], "medications": []},
        {
            "summary_items": [field("Med-A", "Med-A")],
            "medications": [
                {"name_as_written": field("Med-A", "Med-A"), "dose": field("100 mg", "Med-A")}
            ],
        },
        {
            "summary_items": [field("Med-A", "Med-A")],
            "medications": [{"name_as_written": field("Med-A", "Med-A"), "uncertainties": ["take more"]}],
        },
    ],
)
def test_analyzer_rejects_invented_or_unsafe_model_output(payload):
    client = MagicMock()
    client.responses.create.return_value = response(payload)
    with pytest.raises(ReviewRequired):
        analyze_prescription([PdfPage(1, "Med-A")], client, "deployment", 1_000)


def test_scanned_pdf_does_not_call_model():
    client = MagicMock()
    analysis = analyze_prescription([PdfPage(1, ""), PdfPage(2, " ")], client, "unused", 1_000)
    assert analysis.status == "needs_review"
    assert analysis.pages_without_text == [1, 2]
    client.responses.create.assert_not_called()


def test_chunked_pdf_is_explicitly_flagged_for_review():
    client = MagicMock()

    def make_response(**kwargs):
        assert "json" in kwargs["input"].lower()
        page = json.loads(kwargs["input"])["pages"][0]
        text = page["text"].strip()
        return response({"summary_items": [field(text, text, page["page"])], "medications": []})

    client.responses.create.side_effect = make_response
    analysis = analyze_prescription(
        [PdfPage(1, "Med-A 10 mg."), PdfPage(2, "Med-B 20 mg.")], client, "deployment", 12
    )
    assert client.responses.create.call_count == 2
    assert analysis.status == "needs_review"
    assert any("blocos" in item for item in analysis.uncertainties)


def test_invalid_azure_openai_endpoint_is_never_used():
    settings = Settings(
        _env_file=None,
        azure_openai_endpoint="https://other.example/openai/v1/",
        azure_openai_api_key="not-a-real-key",
        azure_openai_deployment="deployment",
    )
    with pytest.raises(ProcessingError) as error:
        create_azure_openai_client(settings)
    assert error.value.code == "INVALID_AZURE_OPENAI_ENDPOINT"


def test_invalid_model_json_requires_review_without_exposing_raw_response(monkeypatch, tmp_path):
    client = MagicMock()
    client.responses.create.return_value = SimpleNamespace(
        status="completed", output_text="invalid private model response"
    )
    monkeypatch.setattr(
        "src.services.prescription_pipeline.extract_pdf_pages",
        lambda *_args: [PdfPage(1, "Med-A")],
    )
    monkeypatch.setattr(
        "src.services.prescription_pipeline.create_azure_openai_client", lambda _settings: client
    )
    settings = Settings(_env_file=None, azure_openai_deployment="deployment")
    analysis = process_prescription(tmp_path / "document.pdf", settings)
    assert analysis.status == "needs_review"
    assert analysis.summary == ""
    assert "private model response" not in analysis.model_dump_json()
    client.close.assert_called_once()
