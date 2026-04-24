"""
Эксперимент 2.1 — Baseline inference на примерах из репозитория Slovo.
Запуск:
    python exp_2_1_baseline.py
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
del _os, _sys


import os
import platform
import time

import cv2
import numpy as np
import onnxruntime as ort

from constants import classes

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

MODEL_PATH = "mvit32-2.onnx"
FRAME_INT = 2  # каждый 2-й кадр
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
CROP_SIZE = 224

EXAMPLE_VIDEOS = [
    "examples/0a2ffece-2832-4011-b656-915f39aa7850.mp4",
    "examples/2f39d6e2-695f-4238-8061-8764b998de5d.mp4",
    "examples/9a97a1f2-7404-4d8a-bc57-dfa87fa42e93.mp4",
    "examples/f17a6060-6ced-4bd1-9886-8578cfbb864f.mp4",
]

TEST_VIDEO_DIR = "slovo_test/test"
TEST_LABELS_CSV = "slovo_test/labels.csv"


# ---------------------------------------------------------------------------
# Препроцессинг
# ---------------------------------------------------------------------------

def resize_pad(frame: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    """Letterbox resize: масштаб + padding серым (как в оригинальном репозитории)."""
    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = (size - new_w) / 2
    dh = (size - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(frame, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))


def load_video_frames(path: str, window_size: int) -> tuple[list, dict]:
    """Загружает видео и равномерно сэмплирует ровно window_size кадров."""
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    meta = {
        "total_frames": total_frames,
        "fps": fps,
        "resolution": f"{width}×{height}",
        "duration_sec": round(total_frames / fps, 2) if fps > 0 else 0,
    }

    # Uniform sampling: ровно window_size кадров независимо от длины видео
    n = len(raw)
    if n == 0:
        return [], meta
    indices = np.linspace(0, n - 1, window_size, dtype=int)
    sampled = [raw[i] for i in indices]
    return sampled, meta


def frames_to_tensor(frames: list, window_size: int) -> np.ndarray:
    """Конвертирует список кадров в тензор [1, 1, C, T, H, W]."""
    processed = []
    for frame in frames[:window_size]:
        img = resize_pad(frame.astype(np.float32))
        img = (img - MEAN) / STD
        img = np.transpose(img, (2, 0, 1))   # C H W
        processed.append(img)

    # Дополнение до window_size если кадров меньше
    while len(processed) < window_size:
        processed.append(processed[-1].copy())

    tensor = np.stack(processed[:window_size], axis=1)   # C T H W
    return tensor[None][None].astype(np.float32)          # 1 1 C T H W


# ---------------------------------------------------------------------------
# Инференс
# ---------------------------------------------------------------------------

def warmup(session, window_size: int):
    """Один прогрев модели до основного цикла."""
    dummy = np.zeros((1, 1, 3, window_size, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    session.run(output_names, {input_name: dummy})


def run_inference(session, tensor: np.ndarray) -> dict:
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    t0 = time.perf_counter()
    out = session.run(output_names, {input_name: tensor})[0].squeeze()
    infer_ms = (time.perf_counter() - t0) * 1000

    probs = softmax(out)
    top5_idx = probs.argsort()[::-1][:5]
    top5_conf = probs[top5_idx]
    top5_words = [classes.get(int(i), f"cls_{i}") for i in top5_idx]

    return {
        "top1_word": top5_words[0],
        "top1_conf": float(top5_conf[0]),
        "top5": list(zip(top5_words, [float(c) for c in top5_conf])),
        "infer_ms": round(infer_ms, 1),
    }


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# Системная информация
# ---------------------------------------------------------------------------

def system_info() -> dict:
    import onnxruntime as ort
    return {
        "os": platform.system() + " " + platform.release(),
        "machine": platform.machine(),
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "ort": ort.__version__,
        "cv2": cv2.__version__,
        "device": "CPU (ONNX Runtime)",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Эксперимент 2.1 — Baseline inference (MViTv2-small-32-2)")
    print("=" * 60)

    # Системная информация
    info = system_info()
    print("\n[Среда выполнения]")
    for k, v in info.items():
        print(f"  {k:<10}: {v}")

    # Загрузка модели
    print(f"\n[Загрузка модели] {MODEL_PATH}")
    t0 = time.perf_counter()
    session = ort.InferenceSession(MODEL_PATH)
    load_ms = (time.perf_counter() - t0) * 1000
    window_size = session.get_inputs()[0].shape[3]   # T = 32
    input_shape = session.get_inputs()[0].shape
    print(f"  Время загрузки : {load_ms:.0f} мс")
    print(f"  Входной тензор : {input_shape}  (N 1 C T H W)")
    print(f"  Window size T  : {window_size} кадров")
    print(f"  Классов        : {len(classes)}")

    print("\n[Прогрев модели] …", end="", flush=True)
    warmup(session, window_size)
    print(" готово")

    # --- Раздел A: 5 примеров с известными метками из тестовой выборки ---
    import csv as csv_mod
    print("\n[A] Детальный разбор — 5 видео из тестовой выборки Slovo")
    print("=" * 60)

    labels = {}
    if os.path.exists(TEST_LABELS_CSV):
        with open(TEST_LABELS_CSV, newline="", encoding="utf-8") as f:
            for row in csv_mod.DictReader(f):
                labels[row["attachment_id"]] = row["text"]

    demo_ids = list(labels.keys())[:5]
    results_detail = []
    for vid_id in demo_ids:
        video_path = os.path.join(TEST_VIDEO_DIR, f"{vid_id}.mp4")
        true_label = labels[vid_id]
        if not os.path.exists(video_path):
            continue

        frames, meta = load_video_frames(video_path, window_size)
        tensor = frames_to_tensor(frames, window_size)
        result = run_inference(session, tensor)

        correct = result["top1_word"] == true_label
        in_top5 = any(w == true_label for w, _ in result["top5"])
        mark = "✓" if correct else ("~" if in_top5 else "✗")

        print(f"\n  [{mark}] Истина: «{true_label}»")
        print(f"       Разрешение: {meta['resolution']} | FPS: {meta['fps']:.0f} | "
              f"Кадров: {meta['total_frames']} | Длит.: {meta['duration_sec']} с")
        print(f"       Тензор: {tensor.shape}")
        print(f"       Инференс: {result['infer_ms']} мс")
        print("       Top-5:")
        for rank, (word, conf) in enumerate(result["top5"], 1):
            marker = " <-- ВЕРНО" if word == true_label else ""
            print(f"         {rank}. {word:<22} {conf*100:.2f}%{marker}")

        results_detail.append({
            **meta, **result,
            "video": vid_id,
            "true_label": true_label,
            "top1_correct": correct,
            "in_top5": in_top5,
        })

    # --- Раздел B: сводная статистика по 100 тестовым видео ---
    print("\n\n[B] Статистика по 100 видео тестовой выборки")
    print("=" * 60)

    all_results = []
    if os.path.exists(TEST_LABELS_CSV):
        with open(TEST_LABELS_CSV, newline="", encoding="utf-8") as f:
            all_labels = {row["attachment_id"]: row["text"]
                          for row in csv_mod.DictReader(f)}

        items = list(all_labels.items())[:20]
        for i, (vid_id, true_label) in enumerate(items, 1):
            video_path = os.path.join(TEST_VIDEO_DIR, f"{vid_id}.mp4")
            if not os.path.exists(video_path):
                continue
            frames, meta = load_video_frames(video_path, window_size)
            tensor = frames_to_tensor(frames, window_size)
            result = run_inference(session, tensor)
            all_results.append({
                **result,
                **meta,
                "true_label": true_label,
                "top1_correct": result["top1_word"] == true_label,
                "in_top5": any(w == true_label for w, _ in result["top5"]),
            })
            done = sum(r["top1_correct"] for r in all_results)
            print(f"  [{i:2d}/20] «{true_label}» → «{result['top1_word']}» "
                  f"{'✓' if result['top1_word'] == true_label else '✗'} "
                  f"| {result['infer_ms']:.0f} мс")

    n = len(all_results)
    top1 = sum(r["top1_correct"] for r in all_results)
    top5 = sum(r["in_top5"] for r in all_results)
    avg_ms = sum(r["infer_ms"] for r in all_results) / n

    print(f"\n  Протестировано видео : {n}")
    print(f"  Top-1 Accuracy       : {top1}/{n} = {top1/n*100:.2f}%")
    print(f"  Top-5 Accuracy       : {top5}/{n} = {top5/n*100:.2f}%")
    print(f"  Среднее время        : {avg_ms:.1f} мс / видео")

    # Сохраняем результаты
    import json
    out = {
        "system": info,
        "model": MODEL_PATH,
        "n_videos": n,
        "top1_accuracy": round(top1 / n, 4),
        "top5_accuracy": round(top5 / n, 4),
        "avg_infer_ms": round(avg_ms, 1),
        "detail_5_videos": results_detail,
        "all_results": all_results,
    }
    with open("results_2_1.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nРезультаты сохранены в results_2_1.json")


if __name__ == "__main__":
    main()
