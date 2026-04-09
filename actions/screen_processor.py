import io
import json
import sys
import cv2
import mss
import mss.tools
from pathlib import Path

try:
    import PIL.Image
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

from core.llm_adapter import complete_text


IMG_MAX_W = 640
IMG_MAX_H = 360
JPEG_Q = 55

SYSTEM_PROMPT = (
    "You are JARVIS from Iron Man movies. "
    "You are running in text-only mode for screen processing. "
    "Give concise, practical guidance in max 2 short sentences and address user as sir."
)


def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"


def _get_camera_index() -> int:
    try:
        with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "camera_index" in cfg:
            return int(cfg["camera_index"])
    except Exception:
        pass
    return 0


def _to_jpeg(img_bytes: bytes) -> bytes:
    if not _PIL_OK:
        return img_bytes
    img = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.thumbnail([IMG_MAX_W, IMG_MAX_H], PIL.Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_Q, optimize=False)
    return buf.getvalue()


def _capture_screenshot() -> bytes:
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[1])
        png_bytes = mss.tools.to_png(shot.rgb, shot.size)
    return _to_jpeg(png_bytes)


def _capture_camera() -> bytes:
    camera_index = _get_camera_index()
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Camera could not be opened: index {camera_index}")
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Could not capture camera frame.")
    if _PIL_OK:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(rgb)
        img.thumbnail([IMG_MAX_W, IMG_MAX_H], PIL.Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_Q, optimize=False)
        return buf.getvalue()
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    return buf.tobytes()


def screen_process(parameters: dict, response: str | None = None, player=None, session_memory=None) -> bool:
    user_text = (parameters or {}).get("text") or (parameters or {}).get("user_text", "")
    user_text = (user_text or "").strip()
    if not user_text:
        print("[ScreenProcess] ⚠️ No user_text provided.")
        return False

    angle = (parameters or {}).get("angle", "screen").lower().strip()
    print(f"[ScreenProcess] angle={angle!r} text={user_text!r}")

    try:
        if angle == "camera":
            _capture_camera()
            print("[ScreenProcess] 📷 Camera captured")
        else:
            _capture_screenshot()
            print("[ScreenProcess] 🖥️ Screen captured")
    except Exception as e:
        print(f"[ScreenProcess] ❌ Capture error: {e}")
        return False

    try:
        answer = complete_text(
            f"User request: {user_text}\n"
            "Captured image is available locally but vision endpoint is disabled in this build. "
            "Give the best possible next-step guidance based only on the request text. "
            "Be explicit about this limitation.",
            system_instruction=SYSTEM_PROMPT,
            max_tokens=220,
        )
        if player:
            player.write_log(f"Jarvis: {answer}")
        print(f"[ScreenProcess] 💬 {answer}")
        return True
    except Exception as e:
        print(f"[ScreenProcess] ❌ MiniMax error: {e}")
        raise


def warmup_session(player=None):
    return None


if __name__ == "__main__":
    print("[TEST] screen_processor minimax text-only mode")
