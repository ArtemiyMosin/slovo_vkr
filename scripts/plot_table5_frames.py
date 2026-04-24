"""
Визуализация контрольных примеров для Таблицы 5.
Запуск: python plot_table5_frames.py
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
del _os, _sys


import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['figure.dpi'] = 200

TEST_VIDEO_DIR = "slovo_test/test"

EXAMPLES = [
    {
        "video_id": "6e83d76e-f85b-4930-8cff-b8891d2414d8",
        "true_label": "Нежный",
        "pred_label": "Нежный",
        "confidence": "0.22%",
        "correct": True,
        "result_str": "✓ Top-1",
    },
    {
        "video_id": "c0ecbb8a-ac67-4ba5-8f38-2fdd72bb27ef",
        "true_label": "Ежик",
        "pred_label": "Ежик",
        "confidence": "0.19%",
        "correct": True,
        "result_str": "✓ Top-1",
    },
    {
        "video_id": "aa2c3cc2-ad10-42eb-9df2-ccb3aa46a34a",
        "true_label": "Ланч",
        "pred_label": "Прием пищи",
        "confidence": "0.14%",
        "correct": False,
        "result_str": "✗ Top-3",
    },
    {
        "video_id": "15884d3a-18b9-4870-a1e6-81c52ffcf557",
        "true_label": "Отчаянный",
        "pred_label": "Отчаянный",
        "confidence": "0.20%",
        "correct": True,
        "result_str": "✓ Top-1",
    },
    {
        "video_id": "4aa78a61-ab45-4d6d-aa77-0fdf5c6fe1a0",
        "true_label": "Рассердиться",
        "pred_label": "Рассердиться",
        "confidence": "0.14%",
        "correct": True,
        "result_str": "✓ Top-1",
    },
]

N_FRAMES = 4


def extract_frames(video_path, n=4):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(np.linspace(0, total - 1, n, dtype=int).tolist())
    sorted_indices = sorted(indices)
    frames = {}
    current = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current in indices:
            frames[current] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current += 1
    cap.release()
    return [frames[i] for i in sorted_indices if i in frames]


def square_crop(frame):
    """Центральный квадратный кроп."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0:y0 + side, x0:x0 + side]


# ---------------------------------------------------------------------------
# Фигура: 5 строк × 4 кадра, подписи через fig.text()
# ---------------------------------------------------------------------------

color_ok  = "#2C6FAC"
color_err = "#C0392B"

fig = plt.figure(figsize=(13, 16), facecolor="white")
gs = GridSpec(5, 4, figure=fig,
              hspace=0.15, wspace=0.05,
              top=0.93, bottom=0.02,
              left=0.22, right=0.98)

for row, ex in enumerate(EXAMPLES):
    path = f"{TEST_VIDEO_DIR}/{ex['video_id']}.mp4"
    frames = extract_frames(path, N_FRAMES)
    border = color_ok if ex["correct"] else color_err

    for col, frame in enumerate(frames):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(square_crop(frame))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(2.8)

    # Вычисляем Y-центр строки в координатах figure
    row_top    = gs[row, 0].get_position(fig).y1
    row_bottom = gs[row, 0].get_position(fig).y0
    y_center   = (row_top + row_bottom) / 2

    # Подпись слева через fig.text — без пересечения с осями
    result_color = border
    fig.text(
        0.01, y_center,
        f"Истинный:\n{ex['true_label']}",
        ha="left", va="center",
        fontsize=9, color="#222222",
    )
    fig.text(
        0.11, y_center,
        f"Предск.:\n{ex['pred_label']}\n"
        f"Conf: {ex['confidence']}\n{ex['result_str']}",
        ha="left", va="center",
        fontsize=8.5, color=result_color, fontweight="bold",
    )


fig.savefig("figure_3_table5_frames.png", dpi=200,
            bbox_inches="tight", facecolor="white")
fig.savefig("figure_3_table5_frames.pdf",
            bbox_inches="tight", facecolor="white")
print("Сохранено: figure_3_table5_frames.png / .pdf")
