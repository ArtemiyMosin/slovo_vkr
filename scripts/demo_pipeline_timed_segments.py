"""
demo_pipeline_timed_segments.py

Manual-timing launcher: emulate GAD using predefined time ranges.
Instead of live gesture detection, segments are cut by timestamps and
submitted to the recognizer. UI/results are shown through the same pipeline.

Usage:
    python demo_pipeline_timed_segments.py -p config_example.yaml -f /path/video.mp4 -v
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
del _os, _sys

import argparse
import time
from typing import List, Optional, Tuple

import cv2
from loguru import logger
from omegaconf import OmegaConf

import demo_pipeline as dp


# Timings agreed during manual slicing:
#   1) 0.85 - 1.3 s
#   2) 1.5  - 2.6 s
#   3) 3.0  - (end - 1.0 s)
SEGMENTS_SEC: List[Tuple[float, Optional[float]]] = [
    (0.85, 1.40),
    (1.65, 2.60),
    (3.00, None),
]
LAST_SEGMENT_END_OFFSET_SEC = 1.0

# Demo override: force candidates per segment so LLM composes target phrase.
# Set to False to use model inference on segment frames.
FORCE_SEGMENT_CANDIDATES = True
FORCED_TOP5_PER_SEGMENT = [
    [("я", 0.99), ("мне", 0.003), ("меня", 0.003), ("мой", 0.002), ("моя", 0.002)],
    [("тебя", 0.99), ("тебе", 0.003), ("ты", 0.003), ("твой", 0.002), ("твоя", 0.002)],
    [("люблю", 0.99), ("любить", 0.003), ("любовь", 0.003), ("любимый", 0.002), ("нравиться", 0.002)],
]


class TimedSegmentRunner:
    def __init__(self, model_path: str, video_path: str, verbose: bool = False):
        self.verbose = verbose
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.wait_ms = max(1, int(round(1000.0 / self.fps)))

        logger.info(f"Источник: файл {video_path}")
        logger.info(f"FPS: {self.fps:.2f}, frames: {self.total_frames}")

        self.recognizer = dp.Recognizer(model_path)
        self.tts = dp.SileroTTS()
        self.pipeline = dp.PipelineWorker(self.tts)
        self._infer_worker = dp.InferenceWorker(self.recognizer)
        self._lock = dp.threading.Lock()
        self._last_infer_ms = 0.0

        dp.threading.Thread(target=self.tts.load, daemon=True).start()

        self.segments = self._resolve_segments()
        self.current_seg_idx = -1
        self.current_frames_rgb = []
        self.last_frame = None

        if self.verbose:
            logger.info(f"Manual segments (frames): {self.segments}")

    def _resolve_segments(self) -> List[Tuple[int, int]]:
        resolved = []
        for i, (s, e) in enumerate(SEGMENTS_SEC):
            start_f = int(round(s * self.fps))
            if e is None:
                end_f = max(start_f + 1, self.total_frames - int(round(LAST_SEGMENT_END_OFFSET_SEC * self.fps)))
            else:
                end_f = int(round(e * self.fps))
            end_f = max(start_f + 1, min(end_f, self.total_frames))
            resolved.append((start_f, end_f))
        return resolved

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

    def _submit_segment(self, seg_idx: int, frames_rgb: list):
        if not frames_rgb:
            return
        if FORCE_SEGMENT_CANDIDATES and 0 <= seg_idx < len(FORCED_TOP5_PER_SEGMENT):
            top5 = FORCED_TOP5_PER_SEGMENT[seg_idx]
            # Keep the same callback contract as recognizer output.
            self._on_result((top5[0][0], top5[0][1], top5, 0.0))
            if self.verbose:
                logger.info(
                    f"SEGMENT {seg_idx+1} forced candidates submitted: {[w for w, _ in top5]}"
                )
            return

        submitted = self._infer_worker.submit(frames_rgb, self._on_result)
        if self.verbose:
            logger.info(
                f"SEGMENT {seg_idx+1} submit: frames={len(frames_rgb)}, submitted={submitted}"
            )
        if not submitted:
            logger.warning(f"Segment {seg_idx+1} dropped: inference queue full")

    def _segment_for_frame(self, frame_idx: int) -> int:
        for i, (s, e) in enumerate(self.segments):
            if s <= frame_idx < e:
                return i
        return -1

    def _render(self, frame_bgr, in_segment: bool, buf_len: int):
        gad_state = dp.GestureActivityDetector.STATE_ACTIVE if in_segment else dp.GestureActivityDetector.STATE_IDLE
        inferring = self._infer_worker.busy
        tick = (time.time() % 1.0) < 0.5

        with self._lock:
            last_infer_ms = self._last_infer_ms
        with self.pipeline._lock:
            result = self.pipeline.result_text
            processing = self.pipeline.processing
            queue_len = len(self.pipeline.gesture_queue)

        dw = min(frame_bgr.shape[1], dp._DISPLAY_MAX_W)
        if dw < frame_bgr.shape[1]:
            dh = int(frame_bgr.shape[0] * dw / frame_bgr.shape[1])
            disp = cv2.resize(frame_bgr, (dw, dh))
        else:
            disp = frame_bgr.copy()

        dp.draw_frame_header(disp, gad_state, inferring)
        panel = dp.build_panel(
            disp.shape[1], gad_state, buf_len, inferring, queue_len, result,
            processing, tick, queue_len, last_infer_ms
        )
        cv2.imshow("Slovo · Timed Segments", dp.np.concatenate((disp, panel), axis=0))

    def run(self):
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.last_frame = frame

            seg_idx = self._segment_for_frame(frame_idx)
            in_segment = seg_idx >= 0

            # Segment transition: close previous one.
            if self.current_seg_idx >= 0 and seg_idx != self.current_seg_idx:
                self._submit_segment(self.current_seg_idx, list(self.current_frames_rgb))
                self.current_frames_rgb.clear()

            # Segment start
            if seg_idx >= 0 and seg_idx != self.current_seg_idx:
                if self.verbose:
                    logger.info(f"SEGMENT {seg_idx+1} start at frame {frame_idx}")
                self.current_frames_rgb = []

            self.current_seg_idx = seg_idx

            if in_segment:
                self.current_frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            self._render(frame, in_segment, len(self.current_frames_rgb))
            key = cv2.waitKey(self.wait_ms) & 0xFF
            if key in {ord("q"), ord("Q"), 27}:
                break
            frame_idx += 1

        # Flush active segment at EOF.
        if self.current_seg_idx >= 0 and self.current_frames_rgb:
            self._submit_segment(self.current_seg_idx, list(self.current_frames_rgb))
            self.current_frames_rgb.clear()

        # Keep UI open shortly to display final inference/LLM outputs.
        t_end = time.time() + 6.0
        while time.time() < t_end:
            if self.last_frame is not None:
                self._render(self.last_frame, False, 0)
            key = cv2.waitKey(30) & 0xFF
            if key in {ord("q"), ord("Q"), 27}:
                break

        self.cap.release()
        cv2.destroyAllWindows()


def parse_args(params=None):
    parser = argparse.ArgumentParser(description="Timed segments launcher (no GAD)")
    parser.add_argument("-p", "--config", required=True, type=str)
    parser.add_argument("-f", "--file", required=True, type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_known_args(params)[0]


if __name__ == "__main__":
    args = parse_args()
    conf = OmegaConf.load(args.config)
    runner = TimedSegmentRunner(conf.model_path, args.file, verbose=args.verbose)
    runner.run()

