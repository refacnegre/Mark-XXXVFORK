import io
import os
import tarfile
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import sherpa_onnx


def _log(msg: str) -> None:
    print(f"[LocalVoice] {msg}")


class ModelManager:
    ASR_URL = (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
        "sherpa-onnx-whisper-small.tar.bz2"
    )
    TTS_URL = (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
        "vits-piper-tr_TR-fahrettin-medium.tar.bz2"
    )

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir

    def _download_and_extract(self, url: str, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        archive = target_dir / Path(url).name
        if not archive.exists():
            _log(f"⬇️ Downloading model: {url}")
            urllib.request.urlretrieve(url, archive)
        else:
            _log(f"📦 Archive already exists: {archive}")

        _log(f"📂 Extracting: {archive} -> {target_dir}")
        with tarfile.open(archive, mode="r:bz2") as tf:
            tf.extractall(path=target_dir)
        _log("✅ Extract complete")

    def ensure_models(self) -> tuple[Path, Path]:
        asr_root = self.models_dir / "asr"
        tts_root = self.models_dir / "tts"

        asr_model_dir = asr_root / "sherpa-onnx-whisper-small"
        if not (asr_model_dir / "small-encoder.int8.onnx").exists():
            _log("ASR model missing, preparing download")
            self._download_and_extract(self.ASR_URL, asr_root)
        else:
            _log(f"ASR model ready: {asr_model_dir}")

        tts_model_dir = tts_root / "vits-piper-tr_TR-fahrettin-medium"
        if not (tts_model_dir / "tr_TR-fahrettin-medium.onnx").exists():
            _log("TTS model missing, preparing download")
            self._download_and_extract(self.TTS_URL, tts_root)
        else:
            _log(f"TTS model ready: {tts_model_dir}")

        return asr_model_dir, tts_model_dir


class LocalSTT:
    def __init__(
        self,
        model_dir: Path,
        sample_rate: int = 16000,
        language: str | None = None,
    ):
        self.sample_rate = sample_rate
        self.language = (language or os.getenv("JARVIS_STT_LANGUAGE", "tr")).strip() or "tr"

        _log(f"Initializing STT from: {model_dir}")
        model_config = sherpa_onnx.OfflineModelConfig(
            whisper=sherpa_onnx.OfflineWhisperModelConfig(
                encoder=str(model_dir / "small-encoder.int8.onnx"),
                decoder=str(model_dir / "small-decoder.int8.onnx"),
                language=self.language,
                task="transcribe",
                tail_paddings=64,
            ),
            tokens=str(model_dir / "small-tokens.txt"),
            num_threads=2,
            debug=False,
        )

        _ = model_config  # token/model path validation by explicit constructor below
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(model_dir / "small-encoder.int8.onnx"),
            decoder=str(model_dir / "small-decoder.int8.onnx"),
            tokens=str(model_dir / "small-tokens.txt"),
            language=self.language,
            task="transcribe",
            num_threads=2,
            provider="cpu",
            debug=False,
        )
        _log("✅ STT initialized")

    def _record_until_silence(
        self,
        silence_seconds: float = 1.0,
        max_seconds: float = 18.0,
        voice_threshold: float = 0.012,
    ) -> np.ndarray:
        frames: list[np.ndarray] = []
        started = False
        silence_for = 0.0
        chunk_ms = 0.2
        chunk_size = int(self.sample_rate * chunk_ms)

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
        ) as stream:
            t0 = time.time()
            while True:
                data, _ = stream.read(chunk_size)
                chunk = data[:, 0].copy()

                amp = float(np.abs(chunk).mean())
                if amp > voice_threshold:
                    started = True
                    silence_for = 0.0
                elif started:
                    silence_for += chunk_ms

                if started:
                    frames.append(chunk)

                elapsed = time.time() - t0
                if started and silence_for >= silence_seconds:
                    break
                if elapsed >= max_seconds:
                    break

        if not started:
            _log("🎙️ No voice activity detected, skipping")
            return np.zeros((0,), dtype=np.float32)

        return np.concatenate(frames, axis=0) if frames else np.zeros((0,), dtype=np.float32)

    def listen_once(self) -> str:
        samples = self._record_until_silence()
        if samples.size < int(self.sample_rate * 0.4):
            _log("🎙️ Input too short, skipping")
            return ""

        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, samples)
        self.recognizer.decode_stream(stream)
        text = (stream.result.text or "").strip()
        _log(f"📝 STT text: {text!r}")
        return text


class LocalTTS:
    def __init__(self, model_dir: Path):
        _log(f"Initializing TTS from: {model_dir}")
        tts_model_config = sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(model_dir / "tr_TR-fahrettin-medium.onnx"),
                lexicon="",
                tokens=str(model_dir / "tokens.txt"),
                data_dir=str(model_dir / "espeak-ng-data"),
            ),
            num_threads=2,
            debug=False,
        )

        config = sherpa_onnx.OfflineTtsConfig(
            model=tts_model_config,
            max_num_sentences=2,
        )
        self.tts = sherpa_onnx.OfflineTts(config)
        _log("✅ TTS initialized")

    def synthesize(self, text: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        audio = self.tts.generate(text=text, sid=0, speed=speed)
        return np.asarray(audio.samples, dtype=np.float32), int(audio.sample_rate)

    def speak(self, text: str, speed: float = 1.0) -> None:
        _log(f"🔊 TTS speak request ({len(text)} chars)")
        wav, sr = self.synthesize(text, speed=speed)
        if wav.size == 0:
            _log("⚠️ Empty TTS output")
            return
        sd.play(wav, sr, blocking=True)

    @staticmethod
    def as_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
        data = np.clip(samples, -1.0, 1.0)
        int16 = (data * 32767.0).astype(np.int16)
        with io.BytesIO() as buf:
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(int16.tobytes())
            return buf.getvalue()
