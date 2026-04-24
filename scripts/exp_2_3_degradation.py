"""
Эксперимент 2.3 — Влияние деградации FPS и разрешения на точность классификации.
Берёт 63 верно классифицированных видео из эксперимента 2.2 и деградирует их.
Результаты сохраняются в results_2_3.json.
Запуск:
    python -u exp_2_3_degradation.py
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
del _os, _sys


import json
import os

import cv2
import numpy as np
import onnxruntime as ort

from constants import classes

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

MODEL_PATH = "mvit32-2.onnx"
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
CROP_SIZE = 224
TEST_VIDEO_DIR = "slovo_test/test"
RESULTS_2_2 = "results_2_2.json"

FPS_INTERVALS = [1, 2, 3, 4, 6, 8, 12]
RESOLUTIONS = [None, 480, 360, 240, 180, 120, 90]


# ---------------------------------------------------------------------------
# Препроцессинг
# ---------------------------------------------------------------------------

def resize_pad(frame):
    h, w = frame.shape[:2]
    r = min(CROP_SIZE / h, CROP_SIZE / w)
    new_w = int(round(w * r))
    new_h = int(round(h * r))
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = (CROP_SIZE - new_w) / 2
    dh = (CROP_SIZE - new_h) / 2
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    return cv2.copyMakeBorder(frame, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))


def load_raw_frames(path):
    cap = cv2.VideoCapture(path)
    raw = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return raw


def frames_to_tensor(frames, window_size, target_height=None):
    processed = []
    for frame in frames[:window_size]:
        img = frame.astype(np.float32)
        if target_height is not None and img.shape[0] > target_height:
            scale = target_height / img.shape[0]
            new_w = max(1, int(img.shape[1] * scale))
            img = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
        img = resize_pad(img)
        img = (img - MEAN) / STD
        img = np.transpose(img, (2, 0, 1))
        processed.append(img)
    while len(processed) < window_size:
        processed.append(processed[-1].copy())
    tensor = np.stack(processed[:window_size], axis=1)
    return tensor[None][None].astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def predict(session, tensor):
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    out = session.run(output_names, {input_name: tensor})[0].squeeze()
    probs = softmax(out)
    return classes.get(int(probs.argmax()), "unknown")


def warmup(session, window_size):
    dummy = np.zeros((1, 1, 3, window_size, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    session.run(output_names, {input_name: dummy})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 68)
    print("  Эксперимент 2.3 — Деградация FPS и разрешения")
    print("=" * 68)

    print(f"\n[Загрузка модели] {MODEL_PATH} ...", end="", flush=True)
    session = ort.InferenceSession(MODEL_PATH)
    window_size = session.get_inputs()[0].shape[3]
    print(" готово")

    print("[Прогрев] ...", end="", flush=True)
    warmup(session, window_size)
    print(" готово")

    with open(RESULTS_2_2, encoding="utf-8") as f:
        data_2_2 = json.load(f)

    correct_videos = [
        (r["video_id"], r["true_label"])
        for r in data_2_2["per_video"]
        if r["top1_correct"]
    ]
    n_correct = len(correct_videos)
    print(f"\nБазовая выборка: {n_correct} верно классифицированных видео")

    print(f"Загрузка кадров:", end="", flush=True)
    video_frames = {}
    for i, (vid_id, label) in enumerate(correct_videos, 1):
        path = os.path.join(TEST_VIDEO_DIR, f"{vid_id}.mp4")
        video_frames[vid_id] = (load_raw_frames(path), label)
        if i % 10 == 0 or i == n_correct:
            print(f" {i}", end="", flush=True)
    print(" — готово\n")

    # -----------------------------------------------------------------------
    # Часть A — деградация FPS
    # -----------------------------------------------------------------------
    print("=" * 68)
    print("  ЧАСТЬ A: Деградация FPS")
    print("=" * 68)
    print(f"\n  {'Шаг':>5}  {'Эфф. FPS':>10}  {'Верных':>10}  {'Accuracy':>10}")
    print("  " + "-" * 42)

    fps_results = []
    for interval in FPS_INTERVALS:
        correct = 0
        for vid_id, (raw, true_label) in video_frames.items():
            thinned = raw[::interval] if interval > 1 else raw
            if not thinned:
                thinned = raw[:1]
            n = len(thinned)
            indices = np.linspace(0, n - 1, window_size, dtype=int)
            sampled = [thinned[i] for i in indices]
            tensor = frames_to_tensor(sampled, window_size)
            if predict(session, tensor) == true_label:
                correct += 1

        acc = correct / n_correct * 100
        mark = " <- baseline" if interval == 1 else ""
        print(f"  {interval:>5}  ~{30/interval:>6.0f} fps  {correct:>5}/{n_correct}  {acc:>7.1f}%{mark}",
              flush=True)
        fps_results.append({
            "interval": interval,
            "eff_fps": round(30 / interval, 1),
            "correct": correct,
            "accuracy": round(acc, 2),
        })

    # -----------------------------------------------------------------------
    # Часть B — деградация разрешения
    # -----------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  ЧАСТЬ B: Деградация разрешения")
    print("=" * 68)
    print(f"\n  {'Высота':>10}  {'Метка':>10}  {'Верных':>10}  {'Accuracy':>10}")
    print("  " + "-" * 46)

    res_labels = {
        None: "original", 480: "480p", 360: "360p",
        240: "240p", 180: "180p", 120: "120p", 90: "90p",
    }

    res_results = []
    for target_h in RESOLUTIONS:
        correct = 0
        for vid_id, (raw, true_label) in video_frames.items():
            n = len(raw)
            indices = np.linspace(0, n - 1, window_size, dtype=int)
            sampled = [raw[i] for i in indices]
            tensor = frames_to_tensor(sampled, window_size, target_height=target_h)
            if predict(session, tensor) == true_label:
                correct += 1

        acc = correct / n_correct * 100
        label = res_labels.get(target_h, f"{target_h}p")
        h_str = "original" if target_h is None else f"{target_h}px"
        mark = " <- baseline" if target_h is None else ""
        print(f"  {h_str:>10}  {label:>10}  {correct:>5}/{n_correct}  {acc:>7.1f}%{mark}",
              flush=True)
        res_results.append({
            "target_height": target_h,
            "label": label,
            "correct": correct,
            "accuracy": round(acc, 2),
        })

    # -----------------------------------------------------------------------
    # Сохранение + сводка
    # -----------------------------------------------------------------------
    output = {
        "experiment": "2.3",
        "model": MODEL_PATH,
        "n_baseline": n_correct,
        "fps_degradation": fps_results,
        "resolution_degradation": res_results,
    }
    with open("results_2_3.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 68)
    y_fps = [r["accuracy"] for r in fps_results]
    y_res = [r["accuracy"] for r in res_results]
    print(f"  FPS: {y_fps[0]:.1f}% (30fps) -> {y_fps[-1]:.1f}% (~2fps)  | -{y_fps[0]-y_fps[-1]:.1f} п.п.")
    print(f"  Res: {y_res[0]:.1f}% (orig)  -> {y_res[-1]:.1f}% (90p)    | -{y_res[0]-y_res[-1]:.1f} п.п.")
    print("=" * 68)
    print("\nРезультаты сохранены в results_2_3.json")
    print("Для графиков запусти: python plot_2_3.py")


if __name__ == "__main__":
    main()
