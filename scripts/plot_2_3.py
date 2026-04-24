"""
Графики для эксперимента 2.3 — деградация FPS и разрешения.
Читает results_2_3.json и сохраняет figure_5a_fps.png и figure_5b_resolution.png.
Запуск:
    python plot_2_3.py
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
del _os, _sys


import json
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['figure.dpi'] = 200

with open("results_2_3.json", encoding="utf-8") as f:
    data = json.load(f)

fps_results = data["fps_degradation"]
res_results = data["resolution_degradation"]

blue = "#2C6FAC"
orange = "#E07B39"

# ===========================================================================
# РИСУНОК 5а — Деградация FPS
# ===========================================================================

fig1, ax1 = plt.subplots(figsize=(7, 5), facecolor="white")
fig1.subplots_adjust(top=0.88, bottom=0.14, left=0.12, right=0.95)

x_fps = [r["eff_fps"] for r in fps_results]
y_fps = [r["accuracy"] for r in fps_results]

ax1.plot(x_fps, y_fps, "o-", color=blue, linewidth=2.2,
         markersize=8, markerfacecolor="white", markeredgewidth=2.2)
ax1.axhline(100.0, color=blue, linewidth=1, linestyle="--", alpha=0.35)
ax1.fill_between(x_fps, 100.0, y_fps, alpha=0.08, color=blue)

for x, y in zip(x_fps, y_fps):
    offset = 10 if y > 20 else -16
    ax1.annotate(f"{y:.1f}%", (x, y),
                 textcoords="offset points", xytext=(0, offset),
                 ha="center", fontsize=9, color=blue, fontweight="bold")

ax1.set_xlabel("Эффективная частота кадров (fps)", fontsize=11)
ax1.set_ylabel("Retention accuracy, %", fontsize=11)
ax1.set_title("Зависимость точности от частоты кадров\n(MViTv2-small-32-2, база: 63 верных видео)",
              fontsize=10, fontweight="bold")
ax1.set_xlim(-0.5, 32)
ax1.set_ylim(0, 115)
ax1.invert_xaxis()
ax1.set_xticks([r["eff_fps"] for r in fps_results])
ax1.set_xticklabels([f"~{r['eff_fps']:.0f}" for r in fps_results], fontsize=9)
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.spines[["top", "right"]].set_visible(False)

# Аннотация критической точки
ax1.annotate("Критический порог\n~10 fps", xy=(10, 71.43),
             xytext=(16, 50), fontsize=8.5, color="#C0392B",
             arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2))

fig1.savefig("figure_5a_fps.png", dpi=200, bbox_inches="tight", facecolor="white")
fig1.savefig("figure_5a_fps.pdf", bbox_inches="tight", facecolor="white")
print("Сохранено: figure_5a_fps.png / .pdf")


# ===========================================================================
# РИСУНОК 5б — Деградация разрешения
# ===========================================================================

fig2, ax2 = plt.subplots(figsize=(7, 5), facecolor="white")
fig2.subplots_adjust(top=0.88, bottom=0.14, left=0.12, right=0.95)

x_res = list(range(len(res_results)))
y_res = [r["accuracy"] for r in res_results]
x_ticks = [r["label"] for r in res_results]

ax2.plot(x_res, y_res, "s-", color=orange, linewidth=2.2,
         markersize=8, markerfacecolor="white", markeredgewidth=2.2)
ax2.axhline(100.0, color=orange, linewidth=1, linestyle="--", alpha=0.35)
ax2.fill_between(x_res, 100.0, y_res, alpha=0.08, color=orange)

for x, y in zip(x_res, y_res):
    offset = 10 if y > 85 else -16
    ax2.annotate(f"{y:.1f}%", (x, y),
                 textcoords="offset points", xytext=(0, offset),
                 ha="center", fontsize=9, color=orange, fontweight="bold")

ax2.set_xticks(x_res)
ax2.set_xticklabels(x_ticks, fontsize=10)
ax2.set_xlabel("Разрешение (высота кадра)", fontsize=11)
ax2.set_ylabel("Retention accuracy, %", fontsize=11)
ax2.set_title("Зависимость точности от разрешения видео\n(MViTv2-small-32-2, база: 63 верных видео)",
              fontsize=10, fontweight="bold")
ax2.set_ylim(0, 115)
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.spines[["top", "right"]].set_visible(False)

# Аннотация устойчивой зоны
ax2.annotate("Устойчивая зона\n(до 240p)", xy=(3, 100.0),
             xytext=(3.5, 80), fontsize=8.5, color="#27AE60",
             arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.2))

fig2.savefig("figure_5b_resolution.png", dpi=200, bbox_inches="tight", facecolor="white")
fig2.savefig("figure_5b_resolution.pdf", bbox_inches="tight", facecolor="white")
print("Сохранено: figure_5b_resolution.png / .pdf")

plt.show()
