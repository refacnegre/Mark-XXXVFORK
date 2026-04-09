import io
import tarfile
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import sherpa_onnx


class ModelManager:
    ASR_URL = (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
        "sherpa-onnx-whisper-tiny.tar.bz2"
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
            urllib.request.urlretrieve(url, archive)

        with tarfile.open(archive, mode="r:bz2") as tf:
            tf.extractall(path=target_dir)

    def ensure_models(self) -> tuple[Path, Path]:
        asr_root = self.models_dir / "asr"
        tts_root = self.models_dir / "tts"

        asr_model_dir = asr_root / "sherpa-onnx-whisper-tiny"
        if not (asr_model_dir / "tiny-encoder.int8.onnx").exists():
            self._download_and_extract(self.ASR_URL, asr_root)

        tts_model_dir = tts_root / "vits-piper-tr_TR-fahrettin-medium"
        if not (tts_model_dir / "tr_TR-fahrettin-medium.onnx").exists():
            self._download_and_extract(self.TTS_URL, tts_root)

        return asr_model_dir, tts_model_dir


class LocalSTT:
    def __init__(self, model_dir: Path, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        config = sherpa_onnx.OfflineRecognizerConfig(
            model=sherpa_onnx.OfflineModelConfig(
                whisper=sherpa_onnx.OfflineWhisperModelConfig(
                    encoder=str(model_dir / "tiny-encoder.int8.onnx"),
                    decoder=str(model_dir / "tiny-decoder.int8.onnx"),
                    language="tr",
                    task="transcribe",
                    tail_paddings=64,
                ),
                tokens=str(model_dir / "tiny-tokens.txt"),
                num_threads=2,
                debug=False,
            ),
            decoding_method="greedy_search",
        )
        if not config.validate():
            raise ValueError("STT model config invalid")

        self.recognizer = sherpa_onnx.OfflineRecognizer(config)

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
                frames.append(chunk)

                amp = float(np.abs(chunk).mean())
                if amp > voice_threshold:
                    started = True
                    silence_for = 0.0
                elif started:
                    silence_for += chunk_ms

                elapsed = time.time() - t0
                if started and silence_for >= silence_seconds:
                    break
                if elapsed >= max_seconds:
                    break

        return np.concatenate(frames, axis=0) if frames else np.zeros((0,), dtype=np.float32)

    def listen_once(self) -> str:
        samples = self._record_until_silence()
        if samples.size < int(self.sample_rate * 0.4):
            return ""

        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, samples)
        self.recognizer.decode_stream(stream)
        return (stream.result.text or "").strip()


class LocalTTS:
    def __init__(self, model_dir: Path):
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model_dir / "tr_TR-fahrettin-medium.onnx"),
                    lexicon="",
                    tokens=str(model_dir / "tokens.txt"),
                    data_dir=str(model_dir / "espeak-ng-data"),
                ),
                num_threads=2,
                debug=False,
            ),
            max_num_sentences=2,
        )
        if not config.validate():
            raise ValueError("TTS model config invalid")

        self.tts = sherpa_onnx.OfflineTts(config)

    def synthesize(self, text: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        audio = self.tts.generate(text=text, sid=0, speed=speed)
        return np.asarray(audio.samples, dtype=np.float32), int(audio.sample_rate)

    def speak(self, text: str, speed: float = 1.0) -> None:
        wav, sr = self.synthesize(text, speed=speed)
        if wav.size == 0:
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
