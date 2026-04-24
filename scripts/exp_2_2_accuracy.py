"""
Эксперимент 2.2 — Top-1 / Top-5 accuracy + inference time на 100 тестовых видео Slovo.
Запуск:
    python -u exp_2_2_accuracy.py
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
del _os, _sys


import csv
import json
import math
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
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
CROP_SIZE = 224

TEST_VIDEO_DIR = "slovo_test/test"
TEST_LABELS_CSV = "slovo_test/labels.csv"


# ---------------------------------------------------------------------------
# Препроцессинг
# ---------------------------------------------------------------------------

def resize_pad(frame: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
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
        "fps": round(fps, 2),
        "resolution": f"{width}x{height}",
        "duration_sec": round(total_frames / fps, 2) if fps > 0 else 0,
    }

    n = len(raw)
    if n == 0:
        return [], meta
    indices = np.linspace(0, n - 1, window_size, dtype=int)
    return [raw[i] for i in indices], meta


def frames_to_tensor(frames: list, window_size: int) -> np.ndarray:
    processed = []
    for frame in frames[:window_size]:
        img = resize_pad(frame.astype(np.float32))
        img = (img - MEAN) / STD
        img = np.transpose(img, (2, 0, 1))
        processed.append(img)
    while len(processed) < window_size:
        processed.append(processed[-1].copy())
    tensor = np.stack(processed[:window_size], axis=1)
    return tensor[None][None].astype(np.float32)


# ---------------------------------------------------------------------------
# Инференс
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def warmup(session, window_size: int):
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


# ---------------------------------------------------------------------------
# Статистика
# ---------------------------------------------------------------------------

def confidence_interval_95(p: float, n: int) -> float:
    """Wilson score interval half-width (95%)."""
    if n == 0:
        return 0.0
    z = 1.96
    return z * math.sqrt(p * (1 - p) / n)


def print_table_row(rank, word, pred, conf, correct):
    mark = "✓" if correct else "✗"
    print(f"  {rank:3d}. [{mark}] {word:<22} → {pred:<22} {conf*100:5.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Эксперимент 2.2 — Top-1/Top-5 Accuracy + Inference Time (100 видео)")
    print("=" * 70)

    # Системная информация
    print(f"\n  ОС        : {platform.system()} {platform.release()}")
    print(f"  Процессор : {platform.machine()}")
    print(f"  Python    : {platform.python_version()}")
    print(f"  ORT       : {ort.__version__}")
    print(f"  OpenCV    : {cv2.__version__}")

    # Загрузка модели
    print(f"\n[Загрузка модели] {MODEL_PATH}")
    t0 = time.perf_counter()
    session = ort.InferenceSession(MODEL_PATH)
    load_ms = (time.perf_counter() - t0) * 1000
    window_size = session.get_inputs()[0].shape[3]
    print(f"  Время загрузки : {load_ms:.0f} мс")
    print(f"  Входной тензор : {session.get_inputs()[0].shape}")
    print(f"  Window size T  : {window_size} кадров")
    print(f"  Классов        : {len(classes)}")

    print("\n[Прогрев модели] ...", end="", flush=True)
    warmup(session, window_size)
    print(" готово")

    # Загрузка меток
    if not os.path.exists(TEST_LABELS_CSV):
        print(f"\nОШИБКА: файл {TEST_LABELS_CSV} не найден")
        return

    with open(TEST_LABELS_CSV, newline="", encoding="utf-8") as f:
        all_labels = [(row["attachment_id"], row["text"]) for row in csv.DictReader(f)]

    total_in_csv = len(all_labels)
    print(f"\n  Меток в CSV    : {total_in_csv}")

    # Фильтрация: оставляем только те, для которых есть видеофайл
    items = [(vid_id, label) for vid_id, label in all_labels
             if os.path.exists(os.path.join(TEST_VIDEO_DIR, f"{vid_id}.mp4"))]
    print(f"  Найдено видео  : {len(items)}")

    # ---------------------------------------------------------------------------
    # Основной цикл
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(f"  {'#':>3}  {'Истина':<22}  {'Предсказание':<22}  {'Conf':>6}  {'T':>7}  OK")
    print("-" * 70)

    results = []
    for i, (vid_id, true_label) in enumerate(items, 1):
        video_path = os.path.join(TEST_VIDEO_DIR, f"{vid_id}.mp4")
        frames, meta = load_video_frames(video_path, window_size)
        if not frames:
            print(f"  {i:3d}  ПУСТОЙ ФАЙЛ: {vid_id}")
            continue
        tensor = frames_to_tensor(frames, window_size)
        result = run_inference(session, tensor)

        top1_correct = result["top1_word"] == true_label
        in_top5 = any(w == true_label for w, _ in result["top5"])
        mark = "✓" if top1_correct else ("~" if in_top5 else "✗")

        print(f"  {i:3d}  {true_label:<22}  {result['top1_word']:<22}  "
              f"{result['top1_conf']*100:5.1f}%  {result['infer_ms']:6.0f}мс  {mark}", flush=True)

        results.append({
            "video_id": vid_id,
            "true_label": true_label,
            "top1_word": result["top1_word"],
            "top1_conf": result["top1_conf"],
            "top5_words": [w for w, _ in result["top5"]],
            "top5_confs": [float(c) for _, c in result["top5"]],
            "infer_ms": result["infer_ms"],
            "top1_correct": top1_correct,
            "in_top5": in_top5,
            **meta,
        })

    print("-" * 70)

    # ---------------------------------------------------------------------------
    # Итоговая статистика
    # ---------------------------------------------------------------------------
    n = len(results)
    if n == 0:
        print("Нет результатов.")
        return

    top1 = sum(r["top1_correct"] for r in results)
    top5 = sum(r["in_top5"] for r in results)
    top1_acc = top1 / n
    top5_acc = top5 / n
    ci_top1 = confidence_interval_95(top1_acc, n)
    ci_top5 = confidence_interval_95(top5_acc, n)

    times = [r["infer_ms"] for r in results]
    avg_ms = sum(times) / n
    std_ms = math.sqrt(sum((t - avg_ms) ** 2 for t in times) / n)
    min_ms = min(times)
    max_ms = max(times)

    # Средняя уверенность модели
    top1_confs_correct = [r["top1_conf"] for r in results if r["top1_correct"]]
    top1_confs_wrong = [r["top1_conf"] for r in results if not r["top1_correct"]]
    avg_conf_correct = sum(top1_confs_correct) / len(top1_confs_correct) if top1_confs_correct else 0
    avg_conf_wrong = sum(top1_confs_wrong) / len(top1_confs_wrong) if top1_confs_wrong else 0

    # Throughput
    throughput = 1000 / avg_ms  # видео/сек

    print("\n" + "=" * 70)
    print("  СВОДНАЯ СТАТИСТИКА")
    print("=" * 70)
    print(f"\n  Протестировано видео   : {n}")
    print(f"\n  Top-1 Accuracy         : {top1}/{n}  =  {top1_acc*100:.2f}%")
    print(f"  95% CI Top-1           : ±{ci_top1*100:.2f}%  "
          f"[{(top1_acc - ci_top1)*100:.2f}%, {(top1_acc + ci_top1)*100:.2f}%]")
    print(f"\n  Top-5 Accuracy         : {top5}/{n}  =  {top5_acc*100:.2f}%")
    print(f"  95% CI Top-5           : ±{ci_top5*100:.2f}%  "
          f"[{(top5_acc - ci_top5)*100:.2f}%, {(top5_acc + ci_top5)*100:.2f}%]")
    print(f"\n  Inference time (мс):")
    print(f"    Mean ± Std           : {avg_ms:.1f} ± {std_ms:.1f}")
    print(f"    Min / Max            : {min_ms:.1f} / {max_ms:.1f}")
    print(f"    Throughput           : {throughput:.3f} видео/сек")
    print(f"\n  Ср. уверенность (верно): {avg_conf_correct*100:.2f}%")
    print(f"  Ср. уверенность (неверно): {avg_conf_wrong*100:.2f}%")

    # Распределение по уверенности
    bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
    print("\n  Распределение Top-1 уверенности:")
    for lo, hi in bins:
        cnt = sum(1 for r in results if lo <= r["top1_conf"] < hi)
        correct_in_bin = sum(1 for r in results if lo <= r["top1_conf"] < hi and r["top1_correct"])
        bar = "█" * cnt
        pct_correct = correct_in_bin / cnt * 100 if cnt else 0
        print(f"    [{lo:.1f}–{hi:.2f}): {cnt:3d} видео | точных: {correct_in_bin:3d} ({pct_correct:.0f}%)  {bar}")

    # Ошибочные предсказания
    errors = [(r["true_label"], r["top1_word"]) for r in results if not r["top1_correct"]]
    print(f"\n  Неверные предсказания ({len(errors)} из {n}):")
    for true_l, pred_l in errors:
        in_t5 = any(r["true_label"] == true_l and r["in_top5"] for r in results
                    if r["true_label"] == true_l and not r["top1_correct"])
        tag = "(в Top-5)" if in_t5 else ""
        print(f"    «{true_l}» → «{pred_l}» {tag}")

    # ---------------------------------------------------------------------------
    # Сохранение результатов
    # ---------------------------------------------------------------------------
    output = {
        "experiment": "2.2",
        "model": MODEL_PATH,
        "model_load_ms": round(load_ms, 1),
        "window_size": window_size,
        "crop_size": CROP_SIZE,
        "n_videos": n,
        "top1_correct": top1,
        "top5_correct": top5,
        "top1_accuracy": round(top1_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "ci95_top1": round(ci_top1, 4),
        "ci95_top5": round(ci_top5, 4),
        "infer_ms_mean": round(avg_ms, 1),
        "infer_ms_std": round(std_ms, 1),
        "infer_ms_min": round(min_ms, 1),
        "infer_ms_max": round(max_ms, 1),
        "throughput_vps": round(throughput, 3),
        "avg_conf_correct": round(avg_conf_correct, 4),
        "avg_conf_wrong": round(avg_conf_wrong, 4),
        "system": {
            "os": platform.system() + " " + platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "ort": ort.__version__,
            "cv2": cv2.__version__,
        },
        "per_video": results,
    }

    with open("results_2_2.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\nРезультаты сохранены в results_2_2.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
