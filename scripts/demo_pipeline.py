"""
demo_pipeline.py — Slovo + GAD (MediaPipe) + YandexGPT + Silero TTS.

Архитектура:
  MediaPipe Hands → GAD → буфер кадров
  → MViTv2 (uniform sample 32 кадров, как в офлайн)
  → YandexGPT нормализация → Silero TTS

Запуск:
    python demo_pipeline.py -p config_example.yaml -v
    python demo_pipeline.py -p config_example.yaml -v -f test_continuous.mp4
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
del _os, _sys


import argparse
import json
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
import wave
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw, ImageFont
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf

import onnxruntime as ort
from constants import classes

ort.set_default_logger_severity(4)
logger.remove(0)
logger.add(sys.stdout, format="{level} | {message}")

load_dotenv()

YANDEX_API_KEY = os.getenv("api")
YANDEX_FOLDER  = os.getenv("folder")

# ── GAD ──────────────────────────────────────────────────────────────────────
VELOCITY_THRESHOLD  = 0.012
# Default (webcam) behavior: start threshold equals motion threshold
# and no multi-frame confirmation is required.
START_VELOCITY_THRESHOLD = VELOCITY_THRESHOLD
START_MOTION_FRAMES = 1
START_ON_HAND_PRESENCE = False
START_PRESENCE_FRAMES = 2
STILL_FRAMES_TO_END = 8
MIN_GESTURE_FRAMES  = 4
MAX_GESTURE_FRAMES  = 180

# MediaPipe hand detection/tracking thresholds.
MP_MIN_HAND_DET_CONF = 0.5
MP_MIN_HAND_PRESENCE_CONF = 0.5
MP_MIN_HAND_TRACK_CONF = 0.5

# Optional debug dumping of GAD segments (off by default).
SAVE_GAD_SEGMENTS = False
GAD_SEGMENTS_DIR = "gad_segments"
CLEAR_GAD_SEGMENTS_ON_START = False

# ── LLM ──────────────────────────────────────────────────────────────────────
MIN_GESTURES    = 2
PAUSE_THRESHOLD = 3
NOISE_GLOSS_TOKENS = {
    "---", "первый", "второй", "третий", "четвертый", "пятый",
    "правый", "левый", "рука", "нога", "вторник", "две тысячи",
}

# ── UI colors (RGB for PIL; convert with _bgr() for OpenCV) ──────────────────
PANEL_H        = 162
_DISPLAY_MAX_W = 960   # larger display surface for sharper text/UI

_P_BG     = ( 18,  20,  26)
_P_PANEL  = ( 26,  29,  38)
_P_CARD   = ( 36,  40,  52)
_P_CHIP   = ( 58,  64,  86)
_P_LINE   = ( 56,  61,  78)
_P_HDR    = ( 15,  17,  24)
_P_REC    = (210,  55,  55)
_P_INFER  = (242, 176,  52)
_P_OK     = ( 62, 200,  68)
_P_WHITE  = (240, 243, 252)
_P_MUTED  = (130, 138, 158)
_P_ACCENT = ( 92, 154, 235)
_P_FOOTER = ( 88,  96, 118)


def _bgr(rgb: tuple) -> tuple:
    return (rgb[2], rgb[1], rgb[0])


# ── Font system ───────────────────────────────────────────────────────────────
_FONT_PATH: Optional[str] = None
_FONT_CACHE: dict = {}

for _p in [
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]:
    if os.path.exists(_p):
        _FONT_PATH = _p
        break


def _font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _FONT_CACHE:
        try:
            kw = {"index": 0} if _FONT_PATH and _FONT_PATH.endswith(".ttc") else {}
            _FONT_CACHE[size] = (
                ImageFont.truetype(_FONT_PATH, size, **kw)
                if _FONT_PATH else ImageFont.load_default()
            )
        except Exception:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _tsz(text: str, size: int) -> Tuple[int, int]:
    bb = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
        (0, 0), text, font=_font(size))
    return bb[2] - bb[0], bb[3] - bb[1]


def _rr(draw: ImageDraw.ImageDraw, x1, y1, x2, y2, fill, r: int = 6):
    try:
        draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=r, fill=fill)
    except AttributeError:
        draw.rectangle([(x1 + r, y1), (x2 - r, y2)], fill=fill)
        draw.rectangle([(x1, y1 + r), (x2, y2 - r)], fill=fill)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r),
                       (x1+r, y2-r), (x2-r, y2-r)]:
            draw.ellipse([(cx-r, cy-r), (cx+r, cy+r)], fill=fill)


# ===========================================================================
# GAD — Gesture Activity Detector
# ===========================================================================

class GestureActivityDetector:
    STATE_IDLE      = "idle"
    STATE_ACTIVE    = "active"
    LANDMARKER_PATH = "hand_landmarker.task"

    def __init__(self, fps_hint: float = 30.0):
        base_options = mp_python.BaseOptions(
            model_asset_path=self.LANDMARKER_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=MP_MIN_HAND_DET_CONF,
            min_hand_presence_confidence=MP_MIN_HAND_PRESENCE_CONF,
            min_tracking_confidence=MP_MIN_HAND_TRACK_CONF,
        )
        self.landmarker  = mp_vision.HandLandmarker.create_from_options(options)
        self.state       = self.STATE_IDLE
        self.buffer: list = []
        self.still_count = 0
        self.start_motion_count = 0
        self.start_presence_count = 0
        self.prev_centroid: Optional[np.ndarray] = None
        self.fps_hint = fps_hint if fps_hint and fps_hint > 0 else 30.0

    def _centroid(self, result) -> Optional[np.ndarray]:
        if not result.hand_landmarks:
            return None
        # Use all detected hands to reduce jitter from hand index swaps.
        pts = np.array(
            [[lm.x, lm.y] for hand in result.hand_landmarks for lm in hand],
            dtype=np.float32,
        )
        return pts.mean(axis=0)

    def process(self, frame_bgr: np.ndarray) -> Optional[list]:
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_img)

        centroid = self._centroid(result)
        velocity = 0.0
        if centroid is not None and self.prev_centroid is not None:
            velocity = float(np.linalg.norm(centroid - self.prev_centroid))
        self.prev_centroid = centroid
        # Normalize per-frame motion by FPS so behavior is closer between
        # webcam/live stream and prerecorded files with different frame rates.
        velocity_norm = velocity * (self.fps_hint / 30.0)
        moving = (centroid is not None) and velocity_norm > VELOCITY_THRESHOLD
        moving_start = (centroid is not None) and velocity_norm > START_VELOCITY_THRESHOLD

        if self.state == self.STATE_IDLE:
            if moving_start:
                self.start_motion_count += 1
            else:
                self.start_motion_count = 0

            if centroid is not None:
                self.start_presence_count += 1
            else:
                self.start_presence_count = 0

            start_by_motion = self.start_motion_count >= START_MOTION_FRAMES
            start_by_presence = START_ON_HAND_PRESENCE and (
                self.start_presence_count >= START_PRESENCE_FRAMES
            )

            if start_by_motion or start_by_presence:
                self.state       = self.STATE_ACTIVE
                self.buffer      = [rgb]
                self.still_count = 0
                self.start_motion_count = 0
                self.start_presence_count = 0

        elif self.state == self.STATE_ACTIVE:
            self.buffer.append(rgb)
            if moving:
                self.still_count = 0
            else:
                self.still_count += 1

            ended = (self.still_count >= STILL_FRAMES_TO_END
                     or len(self.buffer) >= MAX_GESTURE_FRAMES)
            if ended:
                frames           = self.buffer.copy()
                self.buffer      = []
                self.still_count = 0
                self.state       = self.STATE_IDLE
                if len(frames) >= MIN_GESTURE_FRAMES:
                    return frames

        return None

    def get_state(self)      -> str: return self.state
    def get_buffer_len(self) -> int: return len(self.buffer)

    def flush(self) -> Optional[list]:
        """
        Flush unfinished ACTIVE segment (e.g. when video file ends).
        """
        if self.state != self.STATE_ACTIVE:
            return None
        frames = self.buffer.copy()
        self.buffer = []
        self.still_count = 0
        self.start_motion_count = 0
        self.start_presence_count = 0
        self.state = self.STATE_IDLE
        if len(frames) >= MIN_GESTURE_FRAMES:
            return frames
        return None


# ===========================================================================
# Preprocessing & Recognizer
# ===========================================================================

MEAN      = np.array([123.675, 116.28,  103.53], dtype=np.float32)
STD       = np.array([58.395,  57.12,   57.375],  dtype=np.float32)
CROP_SIZE = 224


def letterbox(frame: np.ndarray) -> np.ndarray:
    h, w   = frame.shape[:2]
    r      = min(CROP_SIZE / h, CROP_SIZE / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    frame  = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (CROP_SIZE - nw) / 2, (CROP_SIZE - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(frame, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))


def frames_to_tensor(frames: list, window_size: int) -> np.ndarray:
    n       = len(frames)
    indices = np.linspace(0, n - 1, window_size, dtype=int)
    processed = []
    for i in indices:
        img = frames[i].astype(np.float32)
        img = letterbox(img)
        img = (img - MEAN) / STD
        img = np.transpose(img, (2, 0, 1))
        processed.append(img)
    return np.stack(processed, axis=1)[None][None].astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class Recognizer:
    def __init__(self, model_path: str):
        logger.info(f"Загрузка модели {model_path}...")
        self.session      = ort.InferenceSession(model_path)
        self.input_name   = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.window_size  = self.session.get_inputs()[0].shape[3]
        dummy = np.zeros((1, 1, 3, self.window_size, CROP_SIZE, CROP_SIZE),
                         dtype=np.float32)
        self.session.run(self.output_names, {self.input_name: dummy})
        logger.info(f"Модель готова (window={self.window_size})")

    def predict(self, frames: list) -> tuple:
        tensor = frames_to_tensor(frames, self.window_size)
        t0     = time.perf_counter()
        out    = self.session.run(self.output_names,
                                  {self.input_name: tensor})[0].squeeze()
        ms     = (time.perf_counter() - t0) * 1000
        probs  = softmax(out)
        idx5   = probs.argsort()[::-1][:5]
        top5   = [(classes.get(int(i), f"cls_{i}"), float(probs[i])) for i in idx5]
        return top5[0][0], top5[0][1], top5, round(ms, 1)


# ===========================================================================
# InferenceWorker — non-blocking background thread
# ===========================================================================

class InferenceWorker:
    def __init__(self, recognizer: Recognizer):
        self.recognizer = recognizer
        # Queue completed gesture segments so quick successive gestures are not lost
        # while previous inference is still running.
        self._q         = queue.Queue(maxsize=8)
        self._busy      = False
        threading.Thread(target=self._loop, daemon=True).start()

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, frames: list, callback) -> bool:
        try:
            self._q.put_nowait((frames, callback))
            return True
        except queue.Full:
            return False

    def _loop(self):
        while True:
            frames, callback = self._q.get()
            self._busy = True
            try:
                callback(self.recognizer.predict(frames))
            except Exception as exc:
                logger.error(f"Inference error: {exc}")
            finally:
                self._busy = False


# ===========================================================================
# YandexGPT
# ===========================================================================

def yandex_gpt(top5_per_gesture: list) -> str:
    lines = []
    for i, top5 in enumerate(top5_per_gesture, 1):
        candidates = ", ".join(
            f'"{w}" ({p*100:.2f}%){" [noise]" if w in NOISE_GLOSS_TOKENS else ""}'
            for w, p in top5
        )
        lines.append(f"Жест {i}: [{candidates}]")

    prompt = (
        "Ты — переводчик русского жестового языка (РЖЯ) в русский язык.\n"
        "Для каждого жеста дан список кандидатов (Top-5) от модели распознавания.\n"
        "Выбери РОВНО ОДНО слово из КАЖДОГО списка.\n"
        "КРИТИЧЕСКОЕ ПРАВИЛО: порядок жестов менять нельзя.\n"
        "Нельзя переставлять, удалять, объединять или пропускать жесты.\n"
        "Нельзя добавлять новые слова, которых нет в списках кандидатов.\n"
        "Старайся выбирать семантически связные слова (местоимения/глаголы/существительные),\n"
        "а не технические или шумовые метки.\n"
        "Кандидаты с пометкой [noise] выбирай ТОЛЬКО если нет осмысленной альтернативы.\n"
        "Верни ТОЛЬКО JSON без markdown и пояснений в формате:\n"
        '{"chosen": ["слово_для_жеста_1", "слово_для_жеста_2", "..."]}\n'
        "В массиве chosen должно быть ровно столько элементов, сколько жестов.\n\n"
        + "\n".join(lines)
    )
    body = json.dumps({
        "modelUri": f"gpt://{YANDEX_FOLDER}/yandexgpt-lite",
        "completionOptions": {"stream": False, "temperature": 0.3, "maxTokens": 150},
        "messages": [{"role": "user", "text": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        data=body,
        headers={
            "Authorization": f"Api-Key {YANDEX_API_KEY}",
            "x-folder-id": YANDEX_FOLDER,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read())
            raw = resp["result"]["alternatives"][0]["message"]["text"].strip()
            cleaned = raw
            if "```" in cleaned:
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            try:
                payload = json.loads(cleaned)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
                if not m:
                    raise
                payload = json.loads(m.group(0))

            chosen = payload.get("chosen")
            if not isinstance(chosen, list):
                raise ValueError("LLM response has no valid 'chosen' list")
            if len(chosen) != len(top5_per_gesture):
                raise ValueError("LLM returned wrong number of chosen words")

            validated = []
            for i, word in enumerate(chosen):
                if not isinstance(word, str):
                    raise ValueError(f"Gesture {i+1}: chosen word is not a string")
                allowed = {w for w, _ in top5_per_gesture[i]}
                if word not in allowed:
                    raise ValueError(
                        f"Gesture {i+1}: '{word}' is not in candidates {sorted(allowed)}"
                    )
                validated.append(word)

            # Guardrail: replace noisy token when meaningful alternatives exist.
            for i, word in enumerate(validated):
                if word not in NOISE_GLOSS_TOKENS:
                    continue
                non_noise = [(w, p) for w, p in top5_per_gesture[i]
                             if w not in NOISE_GLOSS_TOKENS]
                if non_noise:
                    replacement = max(non_noise, key=lambda x: x[1])[0]
                    logger.info(
                        f"LLM correction: '{word}' -> '{replacement}' for gesture {i+1}"
                    )
                    validated[i] = replacement

            sentence = " ".join(validated).strip()
            if not sentence:
                raise ValueError("Empty sentence after validated selection")
            return sentence[0].upper() + sentence[1:] + "."
    except Exception as e:
        logger.error(f"YandexGPT error: {e}")
        fallback = " ".join(t[0][0] for t in top5_per_gesture).strip()
        if not fallback:
            return ""
        return fallback[0].upper() + fallback[1:] + "."


# ===========================================================================
# SileroTTS
# ===========================================================================

class SileroTTS:
    def __init__(self):
        self.model = None
        self._lock = threading.Lock()

    def load(self):
        logger.info("Загрузка Silero TTS...")
        try:
            self.model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts", language="ru", speaker="v4_ru", trust_repo=True,
            )
            logger.info("Silero TTS готов")
        except Exception as e:
            # Network/hub failures must not break the main pipeline thread logic.
            self.model = None
            logger.error(f"Silero TTS load error: {e}. Продолжаю без озвучивания.")

    def speak(self, text: str):
        if self.model is None:
            return
        with self._lock:
            try:
                audio    = self.model.apply_tts(text=text, speaker="aidar",
                                                sample_rate=24000)
                audio_np = (audio.numpy() * 32767).astype(np.int16)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wav_path = f.name
                with wave.open(wav_path, "w") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
                    wf.writeframes(audio_np.tobytes())
                os.system(f"afplay {wav_path} &")
            except Exception as e:
                logger.error(f"TTS error: {e}")


# ===========================================================================
# PipelineWorker
# ===========================================================================

class PipelineWorker:
    def __init__(self, tts: SileroTTS):
        self.tts              = tts
        self._lock            = threading.Lock()
        self.gesture_queue: list = []
        self.last_activity_time  = time.time()
        self.result_text         = ""
        self.processing          = False
        self.gad_active          = False
        threading.Thread(target=self._loop, daemon=True).start()

    def add_gesture(self, top5: list):
        with self._lock:
            self.gesture_queue.append(top5)
            self.last_activity_time = time.time()
        candidates = ", ".join(f"{w} ({p*100:.1f}%)" for w, p in top5)
        logger.info(f"Распознан жест (Top-5): {candidates}")

    def on_gad_started(self):
        with self._lock:
            self.last_activity_time = time.time()

    def set_gad_active(self, is_active: bool):
        with self._lock:
            self.gad_active = is_active

    def _loop(self):
        while True:
            time.sleep(0.3)
            with self._lock:
                elapsed = time.time() - self.last_activity_time
                ready   = (len(self.gesture_queue) >= MIN_GESTURES
                           and elapsed >= PAUSE_THRESHOLD
                           and not self.gad_active
                           and not self.processing)
                if ready:
                    batch           = list(self.gesture_queue)
                    self.gesture_queue.clear()
                    self.processing = True
            if ready:
                llm_payload = [[w for w, _ in t] for t in batch]
                logger.info(f"LLM <- Top-5 кандидаты: {llm_payload}")
                sentence = yandex_gpt(batch)
                logger.info(f"LLM -> {sentence}")
                with self._lock:
                    self.result_text = sentence
                    self.processing  = False
                self.tts.speak(sentence)


# ===========================================================================
# UI
# ===========================================================================

def draw_frame_header(frame: np.ndarray, gad_state: str, inferring: bool):
    """Adds semi-transparent header overlay + colored border. Mutates frame."""
    h, w  = frame.shape[:2]
    hdr_h = 48

    dark          = np.full((hdr_h, w, 3), _bgr(_P_HDR), dtype=np.uint8)
    frame[:hdr_h] = cv2.addWeighted(dark, 0.90, frame[:hdr_h], 0.10, 0)

    strip = Image.fromarray(cv2.cvtColor(frame[:hdr_h], cv2.COLOR_BGR2RGB))
    draw  = ImageDraw.Draw(strip)

    # App name
    draw.text((14, 8), "SLOVO", font=_font(22), fill=_P_WHITE)
    draw.text((100, 15), "RSL PIPELINE", font=_font(13), fill=(146, 156, 184))

    # Model badge (center)
    m = "MViTv2-small-32-2"
    mw, _ = _tsz(m, 10)
    mx = w // 2 - mw // 2
    _rr(draw, mx - 9, 13, mx + mw + 9, 33, (34, 38, 52), r=5)
    draw.text((mx, 15), m, font=_font(10), fill=(146, 156, 186))

    # Status badge (right)
    if gad_state == GestureActivityDetector.STATE_ACTIVE:
        s_t, s_c, s_f = "● ЗАПИСЬ",    _P_REC,   (62, 24, 24)
    elif inferring:
        s_t, s_c, s_f = "◌ ОБРАБОТКА", _P_INFER, (60, 44, 14)
    else:
        s_t, s_c, s_f = "○ ОЖИДАНИЕ",  _P_MUTED, (32, 36, 48)
    sw, _ = _tsz(s_t, 12)
    sx = w - sw - 26
    _rr(draw, sx - 9, 11, sx + sw + 9, 35, s_f, r=5)
    draw.text((sx, 14), s_t, font=_font(12), fill=s_c)

    frame[:hdr_h] = cv2.cvtColor(np.array(strip), cv2.COLOR_RGB2BGR)

    # Colored border
    if gad_state == GestureActivityDetector.STATE_ACTIVE:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), _bgr(_P_REC), 4)
    elif inferring:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), _bgr(_P_INFER), 3)


def build_panel(fw: int, gad_state: str, buf_len: int, inferring: bool,
                pending_gestures: int, result: str, processing: bool,
                tick: bool = True, queue_len: int = 0,
                last_infer_ms: float = 0) -> np.ndarray:
    """
    PANEL_H=162 layout:
      y=0..2   stripe
      y=3..46  status row  (44px)
      y=47     divider
      y=48..95 glosses row (48px)
      y=96     divider
      y=97..148 result row (52px)
      y=149    divider
      y=150..161 footer    (12px)
    """
    # ── OpenCV geometry ───────────────────────────────────────────────────
    panel = np.full((PANEL_H, fw, 3), _bgr(_P_PANEL), dtype=np.uint8)

    # Accent stripe
    if gad_state == GestureActivityDetector.STATE_ACTIVE:
        sc = _P_REC
    elif inferring:
        sc = _P_INFER
    elif result and not processing:
        sc = _P_OK
    else:
        sc = _P_ACCENT
    cv2.rectangle(panel, (0, 0), (fw, 3), _bgr(sc), -1)

    # Dividers
    for y in (47, 96, 149):
        cv2.rectangle(panel, (0, y), (fw, y + 1), _bgr(_P_LINE), -1)

    # Status dot (blinks when recording)
    if gad_state == GestureActivityDetector.STATE_ACTIVE:
        dc = _bgr(_P_REC) if tick else (50, 20, 20)
    elif inferring:
        dc = _bgr(_P_INFER)
    else:
        dc = (50, 52, 68)
    cv2.circle(panel, (18, 25), 7, dc, -1)

    # Progress bar
    if gad_state == GestureActivityDetector.STATE_ACTIVE:
        bx0, bx1 = fw - 210, fw - 16
        by0, by1 = 17, 31
        cv2.rectangle(panel, (bx0, by0), (bx1, by1), (48, 52, 66), -1)
        fill = max(2, int(buf_len / MAX_GESTURE_FRAMES * (bx1 - bx0)))
        cv2.rectangle(panel, (bx0, by0), (bx0 + fill, by1), _bgr(_P_REC), -1)
        cv2.rectangle(panel, (bx0, by0), (bx1, by1), (78, 84, 104), 1)

    # Result card
    if result and not processing:
        cv2.rectangle(panel, (8, 101), (fw - 8, 146), _bgr(_P_CARD), -1)

    # ── PIL text ──────────────────────────────────────────────────────────
    pil  = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    # ── Status row ────────────────────────────────────────────────────────
    if gad_state == GestureActivityDetector.STATE_ACTIVE:
        st, stc = f"ЗАПИСЬ   {buf_len} / {MAX_GESTURE_FRAMES}", _P_REC
    elif inferring:
        st, stc = "ОБРАБОТКА ЖЕСТА...", _P_INFER
    else:
        st, stc = "Ожидание жеста", _P_MUTED
    draw.text((34, 16), st, font=_font(17), fill=stc)

    # Last inference time (right side, idle/inferring states)
    if last_infer_ms > 0 and gad_state != GestureActivityDetector.STATE_ACTIVE:
        ms_t = f"{last_infer_ms:.0f} мс"
        mw, _ = _tsz(ms_t, 10)
        draw.text((fw - mw - 14, 20), ms_t, font=_font(10), fill=_P_MUTED)

    # ── Glosses row ───────────────────────────────────────────────────────
    draw.text((14, 51), "ЖЕСТЫ", font=_font(11), fill=_P_MUTED)

    # Gesture counter (right)
    q_t = f"{queue_len} / {MIN_GESTURES}"
    qw, _ = _tsz(q_t, 10)
    q_col = _P_OK if queue_len >= MIN_GESTURES else _P_MUTED
    draw.text((fw - qw - 14, 51), q_t, font=_font(11), fill=q_col)

    # Chips: show neutral placeholders instead of top-1 words.
    chips_to_draw = min(max(pending_gestures, 0), 8)
    cx = 82
    for i in range(chips_to_draw):
        label = f"Жест {i + 1}"
        tw, th = _tsz(label, 13)
        pad    = 5
        cx2    = cx + tw + pad * 2
        if cx2 > fw - qw - 24:
            draw.text((cx, 68), "…", font=_font(13), fill=_P_MUTED)
            break
        _rr(draw, cx, 63, cx2, 63 + th + pad * 2, _P_CHIP)
        draw.text((cx + pad, 63 + pad), label, font=_font(13), fill=_P_WHITE)
        cx = cx2 + 6
    if chips_to_draw == 0:
        draw.text((82, 68), "—", font=_font(13), fill=_P_MUTED)

    # ── Result row ────────────────────────────────────────────────────────
    draw.text((18, 100), "ПРЕДЛОЖЕНИЕ", font=_font(11), fill=_P_MUTED)

    if processing:
        draw.text((18, 116), "Формирую предложение...", font=_font(15), fill=_P_INFER)
    elif result:
        fs = 18
        while fs >= 10:
            rw, _ = _tsz(result, fs)
            if rw <= fw - 36:
                break
            fs -= 2
        display = result
        if fs < 10:
            fs = 10
            while len(display) > 4:
                display = display[:-1]
                rw, _ = _tsz(display + "…", fs)
                if rw <= fw - 36:
                    break
            display += "…"
        draw.text((18, 116), display, font=_font(fs), fill=_P_OK)
    else:
        draw.text((18, 118), "Накапливаю жесты...", font=_font(14), fill=(86, 94, 116))

    # ── Footer ────────────────────────────────────────────────────────────
    draw.text(
        (14, 153),
        "SLOVO · RSL · MViTv2 + YandexGPT + SileroTTS   [R] сброс   [Q] выход",
        font=_font(9), fill=_P_FOOTER,
    )

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ===========================================================================
# Runner
# ===========================================================================

class Runner:
    def __init__(self, model_path: str, config=None, verbose: bool = False,
                 video_path: str = None):
        self.verbose = verbose
        self.video_path = video_path
        source       = video_path if video_path else 0
        self.cap     = cv2.VideoCapture(source)
        cap_fps      = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        gad_fps_hint = cap_fps if cap_fps > 0 else 30.0
        logger.info(f"Источник: {'файл ' + video_path if video_path else 'вебкамера'}")
        logger.info(f"UI font: {_FONT_PATH or 'default (Cyrillic may not render)'}")
        logger.info(f"GAD FPS hint: {gad_fps_hint:.2f}")

        self.recognizer    = Recognizer(model_path)
        self.gad           = GestureActivityDetector(fps_hint=gad_fps_hint)
        self.tts           = SileroTTS()
        threading.Thread(target=self.tts.load, daemon=True).start()
        self.pipeline      = PipelineWorker(self.tts)
        self._infer_worker = InferenceWorker(self.recognizer)
        self._lock         = threading.Lock()
        self._last_infer_ms: float = 0.0
        self._gad_seg_idx = 0

        if SAVE_GAD_SEGMENTS:
            if CLEAR_GAD_SEGMENTS_ON_START and os.path.isdir(GAD_SEGMENTS_DIR):
                shutil.rmtree(GAD_SEGMENTS_DIR, ignore_errors=True)
            os.makedirs(GAD_SEGMENTS_DIR, exist_ok=True)

    def _save_gad_segment(self, frames_rgb: list) -> None:
        if not SAVE_GAD_SEGMENTS or not frames_rgb:
            return
        self._gad_seg_idx += 1
        seg_dir = os.path.join(GAD_SEGMENTS_DIR, f"segment_{self._gad_seg_idx:04d}")
        os.makedirs(seg_dir, exist_ok=True)
        for i, fr_rgb in enumerate(frames_rgb, 1):
            fr_bgr = cv2.cvtColor(fr_rgb, cv2.COLOR_RGB2BGR)
            out = os.path.join(seg_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(out, fr_bgr)
        if self.verbose:
            logger.info(f"GAD segment saved: {seg_dir} ({len(frames_rgb)} frames)")

    def _on_result(self, result):
        _, _, top5, ms = result
        with self._lock:
            self._last_infer_ms = ms
        self.pipeline.add_gesture(top5)
        if self.verbose:
            logger.info(f"Inference {ms:.0f}мс | Top-5 кандидатов:")
            for i, (w, p) in enumerate(top5, 1):
                marker = " ◀ выбран" if i == 1 else ""
                logger.info(f"  {i}. {w:<25} {p*100:.3f}%{marker}")

    def run(self):
        hold_before_play = bool(self.video_path)
        prev_gad_state = self.gad.get_state()
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                # For file sources, finalize unfinished gesture at EOF.
                flushed = self.gad.flush()
                if flushed is not None:
                    self._save_gad_segment(flushed)
                    submitted = self._infer_worker.submit(flushed, self._on_result)
                    if not submitted and self.verbose:
                        logger.warning("Inference queue is full; flushed EOF segment skipped")
                break

            if hold_before_play:
                # Show UI window immediately, then delay actual file playback.
                gad_state = self.gad.get_state()
                buf_len   = self.gad.get_buffer_len()
                inferring = self._infer_worker.busy
                tick      = (time.time() % 1.0) < 0.5

                with self._lock:
                    last_infer_ms = self._last_infer_ms
                with self.pipeline._lock:
                    result     = self.pipeline.result_text
                    processing = self.pipeline.processing
                    queue_len  = len(self.pipeline.gesture_queue)

                dw = min(frame.shape[1], _DISPLAY_MAX_W)
                if dw < frame.shape[1]:
                    dh = int(frame.shape[0] * dw / frame.shape[1])
                    disp = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
                else:
                    disp = frame.copy()

                draw_frame_header(disp, gad_state, inferring)
                panel = build_panel(
                    disp.shape[1], gad_state, buf_len, inferring, queue_len, result,
                    processing, tick, queue_len, last_infer_ms
                )
                cv2.imshow("Slovo · RSL Pipeline",
                           np.concatenate((disp, panel), axis=0))

                logger.info("UI открыт. Пауза 5с перед стартом воспроизведения файла...")
                t_end = time.time() + 5.0
                while time.time() < t_end:
                    key = cv2.waitKey(30) & 0xFF
                    if key in {ord("q"), ord("Q"), 27}:
                        self.cap.release()
                        cv2.destroyAllWindows()
                        return

                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                hold_before_play = False
                continue

            # GAD + inference run on original frame
            gesture_frames = self.gad.process(frame)
            if gesture_frames is not None:
                self._save_gad_segment(gesture_frames)
                submitted = self._infer_worker.submit(gesture_frames, self._on_result)
                if not submitted and self.verbose:
                    logger.warning("Inference queue is full; gesture segment skipped")

            gad_state = self.gad.get_state()
            buf_len   = self.gad.get_buffer_len()
            inferring = self._infer_worker.busy
            tick      = (time.time() % 1.0) < 0.5
            self.pipeline.set_gad_active(
                gad_state == GestureActivityDetector.STATE_ACTIVE
            )
            if (prev_gad_state == GestureActivityDetector.STATE_IDLE
                    and gad_state == GestureActivityDetector.STATE_ACTIVE):
                self.pipeline.on_gad_started()
            prev_gad_state = gad_state

            with self._lock:
                last_infer_ms  = self._last_infer_ms
            with self.pipeline._lock:
                result     = self.pipeline.result_text
                processing = self.pipeline.processing
                queue_len  = len(self.pipeline.gesture_queue)

            # Scale display frame to fit screen
            dw = min(frame.shape[1], _DISPLAY_MAX_W)
            if dw < frame.shape[1]:
                dh = int(frame.shape[0] * dw / frame.shape[1])
                disp = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
            else:
                disp = frame.copy()

            draw_frame_header(disp, gad_state, inferring)
            panel = build_panel(disp.shape[1], gad_state, buf_len,
                                inferring, queue_len, result, processing,
                                tick, queue_len, last_infer_ms)
            cv2.imshow("Slovo · RSL Pipeline",
                       np.concatenate((disp, panel), axis=0))

            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), ord("Q"), 27}:
                break
            if key == ord("r"):
                with self._lock:
                    self._last_infer_ms = 0.0
                with self.pipeline._lock:
                    self.pipeline.result_text  = ""
                    self.pipeline.gesture_queue.clear()

        self.cap.release()
        cv2.destroyAllWindows()


# ===========================================================================
# CLI
# ===========================================================================

def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slovo pipeline: GAD → MViTv2 → YandexGPT → TTS")
    parser.add_argument("-p", "--config",  required=True,  type=str)
    parser.add_argument("-f", "--file",    required=False, type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_known_args(params)[0]


if __name__ == "__main__":
    args   = parse_arguments()
    conf   = OmegaConf.load(args.config)
    runner = Runner(conf.model_path, conf, verbose=args.verbose,
                    video_path=args.file)
    runner.run()
