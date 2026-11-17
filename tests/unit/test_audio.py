import json
from types import SimpleNamespace

import pytest

from src.core.config import Settings
from src.core.errors import ProcessingError
from src.schemas.audio import AudioAnalysis, AudioSegment, LanguageStep, SentimentStep
from src.services.alert_service import audio_alerts, find_critical_terms
from src.services.audio_pipeline import process_audio
from src.services.language_client import analyze_healthcare, analyze_sentiment, create_language_client
from src.services.speech_client import transcribe_audio


class Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, event):
        for callback in self.callbacks:
            callback(event)


class Future:
    def __init__(self, callback=lambda: None):
        self.callback = callback

    def get(self):
        self.callback()


class FakeSpeechSDK:
    ResultReason = SimpleNamespace(RecognizedSpeech="recognized")
    CancellationReason = SimpleNamespace(Error="error", EndOfStream="end")
    CancellationErrorCode = SimpleNamespace(
        TooManyRequests="throttled",
        ConnectionFailure="connection",
        ServiceTimeout="timeout",
        ServiceError="service",
        ServiceUnavailable="unavailable",
    )
    OutputFormat = SimpleNamespace(Detailed="detailed")

    def __init__(self, scenarios):
        self.scenarios = list(scenarios)
        self.configs = []
        self.recognizers = []
        self.audio = SimpleNamespace(AudioConfig=lambda filename: SimpleNamespace(filename=filename))

    def SpeechConfig(self, subscription, region):
        config = SimpleNamespace(subscription=subscription, region=region)
        self.configs.append(config)
        return config

    def SpeechRecognizer(self, speech_config, audio_config):
        scenario = self.scenarios.pop(0)
        recognizer = SimpleNamespace(
            recognized=Signal(),
            canceled=Signal(),
            session_stopped=Signal(),
        )

        def start():
            for text in scenario.get("texts", []):
                result = SimpleNamespace(
                    reason="recognized",
                    text=text,
                    offset=12_000_000,
                    duration=25_000_000,
                    json=json.dumps({"NBest": [{"Confidence": 0.9}]}),
                )
                recognizer.recognized.emit(SimpleNamespace(result=result))
            if "error" in scenario:
                details = SimpleNamespace(reason="error", code=scenario["error"])
                recognizer.canceled.emit(SimpleNamespace(cancellation_details=details))
            else:
                recognizer.session_stopped.emit(SimpleNamespace())

        recognizer.start_continuous_recognition_async = lambda: Future(start)
        recognizer.stop_continuous_recognition_async = lambda: Future()
        self.recognizers.append(recognizer)
        return recognizer


@pytest.mark.parametrize("language", ["pt-BR", "en-US"])
def test_speech_uses_requested_locale_and_preserves_segments(tmp_path, language):
    sdk = FakeSpeechSDK([{"texts": ["First segment.", "Second segment."]}])
    settings = Settings(
        _env_file=None, azure_speech_key="not-a-real-key", azure_speech_region="eastus"
    )
    segments = transcribe_audio(tmp_path / "synthetic.wav", language, settings, sdk=sdk)
    assert sdk.configs[0].speech_recognition_language == language
    assert [segment.text for segment in segments] == ["First segment.", "Second segment."]
    assert segments[0].start_seconds == 1.2
    assert segments[0].end_seconds == 3.7
    assert segments[0].confidence == 0.9


def test_speech_retries_transient_errors_without_returning_partial_attempt(tmp_path, monkeypatch):
    sdk = FakeSpeechSDK(
        [{"texts": ["partial"], "error": "throttled"}, {"texts": ["complete"]}]
    )
    monkeypatch.setattr("src.services.speech_client.time.sleep", lambda _seconds: None)
    settings = Settings(
        _env_file=None, azure_speech_key="not-a-real-key", azure_speech_region="eastus"
    )
    segments = transcribe_audio(tmp_path / "synthetic.wav", "en-US", settings, sdk=sdk)
    assert [segment.text for segment in segments] == ["complete"]
    assert len(sdk.recognizers) == 2


def test_speech_does_not_retry_permanent_errors(tmp_path):
    sdk = FakeSpeechSDK([{"error": "authentication"}])
    settings = Settings(
        _env_file=None, azure_speech_key="not-a-real-key", azure_speech_region="eastus"
    )
    with pytest.raises(ProcessingError):
        transcribe_audio(tmp_path / "synthetic.wav", "en-US", settings, sdk=sdk)
    assert len(sdk.recognizers) == 1


class Poller:
    def __init__(self, result):
        self._result = result

    def wait(self, _timeout):
        return None

    def done(self):
        return True

    def result(self):
        return self._result


def entity(text, phrase, category, *, assertion=None):
    return SimpleNamespace(
        text=phrase,
        category=category,
        length=len(phrase),
        offset=text.index(phrase),
        confidence_score=0.93,
        normalized_text=None,
        assertion=assertion,
    )


class FakeLanguageClient:
    def __init__(self, *, unsupported=False, bad_offset=False):
        self.unsupported = unsupported
        self.bad_offset = bad_offset
        self.health_documents = []
        self.sentiment_documents = []

    def begin_analyze_healthcare_entities(self, documents, **kwargs):
        self.health_documents.extend(documents)
        results = []
        for document in documents:
            if self.unsupported:
                results.append(
                    SimpleNamespace(
                        id=document["id"],
                        is_error=True,
                        error=SimpleNamespace(code="UnsupportedLanguageCode"),
                    )
                )
                continue
            text = document["text"]
            phrase = "falta de ar" if "falta de ar" in text else "shortness of breath"
            time_phrase = "ontem" if "ontem" in text else "yesterday"
            assertion = SimpleNamespace(
                certainty="negative", conditionality="hypothetical", association="subject"
            )
            symptom = entity(text, phrase, "SymptomOrSign", assertion=assertion)
            temporal = entity(text, time_phrase, "Time")
            if self.bad_offset:
                symptom.offset = 0
            relation = SimpleNamespace(
                relation_type="TimeOfCondition",
                confidence_score=0.8,
                roles=[
                    SimpleNamespace(name="Condition", entity=symptom),
                    SimpleNamespace(name="Time", entity=temporal),
                ],
            )
            results.append(
                SimpleNamespace(
                    id=document["id"],
                    is_error=False,
                    entities=[symptom, temporal],
                    entity_relations=[relation],
                    warnings=[],
                )
            )
        return Poller(results)

    def analyze_sentiment(self, documents, **kwargs):
        self.sentiment_documents.extend(documents)
        return [
            SimpleNamespace(
                id=document["id"],
                is_error=False,
                sentiment="negative",
                confidence_scores=SimpleNamespace(positive=0.1, neutral=0.2, negative=0.7),
                warnings=[],
            )
            for document in documents
        ]


@pytest.mark.parametrize(
    ("language", "text", "phrase", "health_language", "sentiment_language"),
    [
        ("pt-BR", "Paciente nega falta de ar desde ontem.", "falta de ar", "pt", "pt-BR"),
        ("en-US", "Patient denies shortness of breath since yesterday.", "shortness of breath", "en", "en"),
    ],
)
def test_language_preserves_negation_temporality_and_locale(
    language, text, phrase, health_language, sentiment_language
):
    segments = [AudioSegment(text=text, start_seconds=1, end_seconds=5)]
    client = FakeLanguageClient()
    health = analyze_healthcare(segments, language, client, 10)
    sentiment = analyze_sentiment(segments, language, client, enabled=True)
    assert client.health_documents[0]["language"] == health_language
    assert client.sentiment_documents[0]["language"] == sentiment_language
    symptom = health.entities[0]
    assert symptom.assertion.certainty == "negative"
    assert symptom.assertion.conditionality == "hypothetical"
    assert symptom.evidence.quote == text
    assert health.relations[0].type == "TimeOfCondition"
    assert health.relations[0].roles[1].category == "Time"
    terms = find_critical_terms(segments, [phrase], health.entities)
    assert terms[0].assertion.certainty == "negative"
    analysis = AudioAnalysis(
        status="completed",
        language=language,
        transcript=text,
        segments=segments,
        clinical_analysis=health.step,
        clinical_entities=health.entities,
        relations=health.relations,
        relevant_terms=terms,
        text_sentiment=sentiment,
    )
    alerts = audio_alerts(analysis)
    assert alerts[0].evidence["assertion"]["certainty"] == "negative"
    assert alerts[-1].evidence["evidence"]["quote"] == text
    assert sentiment.segments[0].sentiment == "negative"
    assert all(alert.type != "sentiment" for alert in alerts)


def test_unsupported_health_is_unavailable_not_an_empty_success():
    segments = [AudioSegment(text="Paciente nega falta de ar desde ontem.")]
    result = analyze_healthcare(segments, "pt-BR", FakeLanguageClient(unsupported=True), 10)
    assert result.step.status == "unavailable"
    assert result.entities == []
    assert analyze_sentiment(segments, "pt-BR", None, enabled=True).status == "unavailable"
    assert analyze_sentiment(segments, "pt-BR", None, enabled=False).status == "disabled"


def test_incorrect_health_offsets_are_rejected():
    segments = [AudioSegment(text="Paciente nega falta de ar desde ontem.")]
    with pytest.raises(ProcessingError) as error:
        analyze_healthcare(segments, "pt-BR", FakeLanguageClient(bad_offset=True), 10)
    assert error.value.code == "INVALID_LANGUAGE_EVIDENCE"


def test_invalid_language_endpoint_is_rejected_before_client_creation():
    settings = Settings(
        _env_file=None,
        azure_language_endpoint="http://other.example/",
        azure_language_key="not-a-real-key",
    )
    with pytest.raises(ProcessingError) as error:
        create_language_client(settings)
    assert error.value.code == "INVALID_AZURE_LANGUAGE_ENDPOINT"


def test_unrecognized_audio_requires_review(monkeypatch, tmp_path):
    monkeypatch.setattr("src.services.audio_pipeline.transcribe_audio", lambda *_args: [])
    result = process_audio(tmp_path / "audio.wav", "en-US", Settings(_env_file=None))
    assert result.status == "needs_review"
    assert result.transcript == ""
    assert result.clinical_analysis.status == "unavailable"
    assert result.text_sentiment.status == "disabled"
