import argparse
import shutil
import subprocess
import wave
from pathlib import Path

import pymupdf

PRESCRIPTIONS = {
    "prescricao-pt.pdf": [
        "EXEMPLO FICTÍCIO - NÃO É PRESCRIÇÃO",
        "Medicamento fictício: MED-ALFA",
        "Concentração: 10 mg.",
        "Dose: 1 unidade.",
        "Via: oral. Frequência: uma vez ao dia.",
        "Duração: 5 dias.",
        "Documento apenas para demonstração, sem uso clínico.",
    ],
    "prescription-en.pdf": [
        "FICTIONAL EXAMPLE - NOT A PRESCRIPTION",
        "Fictional medication: MED-BETA",
        "Strength: 20 mg.",
        "Dose: 1 unit.",
        "Route: oral. Frequency: once daily.",
        "Duration: 5 days.",
        "Demonstration document only, not for clinical use.",
    ],
    "sem-texto.pdf": [],
}
AUDIO = {
    "consulta-pt.wav": (
        "pt",
        "Exemplo fictício para demonstração. O paciente nega falta de ar desde ontem. "
        "Ele menciona dor no peito no passado. Este áudio não é uma conclusão clínica.",
    ),
    "consultation-en.wav": (
        "en",
        "This is a fictional example. The patient denies shortness of breath since yesterday. "
        "They mention chest pain in the past. This audio is not a clinical conclusion.",
    ),
}


def write_pdf(path: Path, lines: list[str]) -> None:
    with pymupdf.open() as document:
        page = document.new_page()
        for index, line in enumerate(lines):
            page.insert_text((72, 72 + 24 * index), line, fontsize=11)
        with path.open("xb") as target:
            target.write(document.tobytes())


def write_audio(path: Path, text: str, voice: str) -> None:
    with path.open("xb"):
        pass
    subprocess.run(
        [
            "say",
            "-v",
            voice,
            "--file-format=WAVE",
            "--data-format=LEI16@16000",
            "--channels=1",
            "-o",
            str(path),
            text,
        ],
        check=True,
    )
    with wave.open(str(path), "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getsampwidth() != 2
            or audio.getframerate() != 16_000
            or audio.getnframes() == 0
        ):
            raise ValueError("O sintetizador não produziu WAV PCM mono de 16 bits a 16 kHz.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera exemplos fictícios para a API.")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "generated")
    parser.add_argument("--audio", action="store_true", help="Gera também áudios com o say do macOS.")
    parser.add_argument("--voice-pt", default="Luciana")
    parser.add_argument("--voice-en", default="Samantha")
    args = parser.parse_args()
    if args.audio and not shutil.which("say"):
        parser.error("--audio requer o comando say do macOS.")
    names = list(PRESCRIPTIONS) + (list(AUDIO) if args.audio else [])
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any((args.output / name).exists() for name in names):
        parser.error("Uma amostra já existe; escolha outro diretório com --output.")

    created: list[Path] = []
    try:
        for name, lines in PRESCRIPTIONS.items():
            path = args.output / name
            write_pdf(path, lines)
            created.append(path)
        if args.audio:
            voices = {"pt": args.voice_pt, "en": args.voice_en}
            for name, (language, text) in AUDIO.items():
                path = args.output / name
                created.append(path)
                write_audio(path, text, voices[language])
    except (OSError, ValueError, subprocess.CalledProcessError):
        for path in created:
            path.unlink(missing_ok=True)
        raise
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
