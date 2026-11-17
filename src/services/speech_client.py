import json
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import azure.cognitiveservices.speech as speechsdk

from src.core.config import Settings
from src.core.errors import ProcessingError
from src.schemas.audio import AudioSegment
from src.schemas.common import Locale

TICKS_PER_SECOND = 10_000_000


class SpeechServiceError(ProcessingError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(code, message)
        self.retryable = retryable


def _wait_future(future: Any, timeout: float) -> None:
    finished = Event()
    errors: list[Exception] = []

    def wait() -> None:
        try:
            future.get()
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()

    Thread(target=wait, daemon=True).start()
    if not finished.wait(timeout):
        raise SpeechServiceError(
            "AZURE_SPEECH_TIMEOUT", "O Azure Speech excedeu o tempo limite.", retryable=True
        )
    if errors:
        raise SpeechServiceError("AZURE_SPEECH_ERROR", "Falha ao iniciar ou encerrar o Azure Speech.") from errors[0]


def _confidence(result: Any) -> float | None:
    try:
        payload = json.loads(result.json)
        value = payload.get("NBest", [{}])[0].get("Confidence")
    except (TypeError, ValueError, IndexError, AttributeError):
        return None
    if isinstance(value, (int, float)) and 0 <= value <= 1:
        return float(value)
    return None


def _transcribe_once(
    path: Path, language: Locale, settings: Settings, sdk: Any
) -> list[AudioSegment]:
    if not settings.azure_speech_key or not settings.azure_speech_region:
        raise ProcessingError("AZURE_SPEECH_NOT_CONFIGURED", "Azure Speech não está configurado.")
    config = sdk.SpeechConfig(
        subscription=settings.azure_speech_key.get_secret_value(),
        region=settings.azure_speech_region,
    )
    config.speech_recognition_language = language
    config.output_format = sdk.OutputFormat.Detailed
    audio = sdk.audio.AudioConfig(filename=str(path))
    recognizer = sdk.SpeechRecognizer(speech_config=config, audio_config=audio)
    done = Event()
    lock = Lock()
    segments: list[AudioSegment] = []
    errors: list[SpeechServiceError] = []
    transient_codes = {
        sdk.CancellationErrorCode.TooManyRequests,
        sdk.CancellationErrorCode.ConnectionFailure,
        sdk.CancellationErrorCode.ServiceTimeout,
        sdk.CancellationErrorCode.ServiceError,
        sdk.CancellationErrorCode.ServiceUnavailable,
    }

    def recognized(event: Any) -> None:
        try:
            result = event.result
            if result.reason != sdk.ResultReason.RecognizedSpeech or not result.text.strip():
                return
            start = result.offset / TICKS_PER_SECOND
            duration = result.duration / TICKS_PER_SECOND
            with lock:
                segments.append(
                    AudioSegment(
                        text=result.text,
                        start_seconds=start,
                        end_seconds=start + duration,
                        confidence=_confidence(result),
                    )
                )
        except Exception:
            with lock:
                errors.append(
                    SpeechServiceError("INVALID_SPEECH_RESPONSE", "A resposta do Azure Speech é inválida.")
                )
            done.set()

    def cancelled(event: Any) -> None:
        try:
            details = event.cancellation_details
            if details and details.reason == sdk.CancellationReason.Error:
                with lock:
                    errors.append(
                        SpeechServiceError(
                            "AZURE_SPEECH_ERROR",
                            "O Azure Speech não conseguiu transcrever o áudio.",
                            retryable=details.code in transient_codes,
                        )
                    )
            elif not details or details.reason != sdk.CancellationReason.EndOfStream:
                with lock:
                    errors.append(
                        SpeechServiceError("AZURE_SPEECH_CANCELED", "A transcrição foi cancelada.")
                    )
        except Exception:
            with lock:
                errors.append(
                    SpeechServiceError("INVALID_SPEECH_RESPONSE", "A resposta do Azure Speech é inválida.")
                )
        finally:
            done.set()

    recognizer.recognized.connect(recognized)
    recognizer.canceled.connect(cancelled)
    recognizer.session_stopped.connect(lambda _event: done.set())
    deadline = time.monotonic() + settings.speech_timeout_seconds
    try:
        _wait_future(
            recognizer.start_continuous_recognition_async(),
            min(30.0, settings.speech_timeout_seconds),
        )
        if not done.wait(max(0, deadline - time.monotonic())):
            raise SpeechServiceError(
                "AZURE_SPEECH_TIMEOUT", "O Azure Speech excedeu o tempo limite.", retryable=True
            )
    finally:
        _wait_future(recognizer.stop_continuous_recognition_async(), 10)
    if errors:
        raise errors[0]
    return sorted(segments, key=lambda segment: segment.start_seconds or 0)


def transcribe_audio(
    path: Path, language: Locale, settings: Settings, *, sdk: Any = speechsdk
) -> list[AudioSegment]:
    if not settings.azure_speech_key or not settings.azure_speech_region:
        raise ProcessingError("AZURE_SPEECH_NOT_CONFIGURED", "Azure Speech não está configurado.")
    for attempt in range(3):
        try:
            return _transcribe_once(path, language, settings, sdk)
        except SpeechServiceError as exc:
            if not exc.retryable or attempt == 2:
                raise
            time.sleep(0.5 * (2**attempt))
    raise AssertionError("Tentativas de transcrição esgotadas.")
