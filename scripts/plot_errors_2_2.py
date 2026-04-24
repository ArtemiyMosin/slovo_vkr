"""
Рисунок 4а — Круговая диаграмма типов ошибок (отдельный файл).
Рисунок 4б — Таблица примеров ошибок (отдельный файл).
Запуск:
    python plot_errors_2_2.py
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
del _os, _sys


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['figure.dpi'] = 200        # retina-качество на экране

# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------

categories = {
    "Семантическая\nблизость": {
        "count": 22,
        "color": "#2C6FAC",
        "examples": [
            "ланч → прием пищи",
            "продолжительность → продолжить",
            "путешествие → миграция",
            "пить → питьевая вода",
            "добропорядочность → верный",
            "вечность → постоянный",
            "7:45 → в восемь пятнадцать",
            "играть → шутить",
        ],
        "description": "Модель путает жесты, близкие по смыслу",
    },
    "Визуальная\nомонимия": {
        "count": 11,
        "color": "#E07B39",
        "examples": [
            "воротник → холодный",
            "влюбленный → С днем рождения",
            "надоело → воротник",
            "аниматор → медведь",
            "пить → вчера",
            "упасть → дать",
        ],
        "description": "Визуально схожие жесты с разными значениями",
    },
    "Технические\nаномалии": {
        "count": 4,
        "color": "#888888",
        "examples": [
            "no_event → ---",
            "обеспокоенный → обеспокоен",
            "Э → минимум",
            "вы → еще",
        ],
        "description": "Несовместимость меток и граничные случаи",
    },
}

labels = list(categories.keys())
sizes  = [v["count"] for v in categories.values()]
colors = [v["color"]  for v in categories.values()]
total  = sum(sizes)   # 37


# ===========================================================================
# РИСУНОК 4а — Круговая диаграмма
# ===========================================================================

fig_pie, ax_pie = plt.subplots(figsize=(7, 6), facecolor="white")
fig_pie.subplots_adjust(top=0.88, bottom=0.20)

wedges, _, autotexts = ax_pie.pie(
    sizes,
    labels=None,
    colors=colors,
    autopct=lambda p: f"{p:.1f}%\n({int(round(p * total / 100))} видео)",
    startangle=110,
    explode=(0.04, 0.04, 0.04),
    wedgeprops=dict(linewidth=1.6, edgecolor="white"),
    pctdistance=0.65,
)

for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight("bold")
    at.set_color("white")

legend_patches = [
    mpatches.Patch(color=colors[i],
                   label=f"{labels[i].replace(chr(10), ' ')}  —  {list(categories.values())[i]['description']}")
    for i in range(len(categories))
]
ax_pie.legend(
    handles=legend_patches,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.22),
    fontsize=8.5,
    frameon=True,
    edgecolor="#cccccc",
    fancybox=False,
)

ax_pie.set_title("Структура ошибок классификации (n = 37)",
                 fontsize=11, fontweight="bold", pad=12)


fig_pie.savefig("figure_4a_pie.png", dpi=200, bbox_inches="tight", facecolor="white")
fig_pie.savefig("figure_4a_pie.pdf", bbox_inches="tight", facecolor="white")
print("Сохранено: figure_4a_pie.png / .pdf")


# ===========================================================================
# РИСУНОК 4б — Таблица примеров
# ===========================================================================

fig_tab, ax_tab = plt.subplots(figsize=(10, 5.5), facecolor="white")
fig_tab.subplots_adjust(top=0.88, bottom=0.06, left=0.02, right=0.98)
ax_tab.axis("off")

col_labels  = ["Категория", "Кол-во", "Примеры ошибочных предсказаний"]
col_x       = [0.0,  0.20, 0.30]
col_widths  = [0.20, 0.10, 0.70]
row_height  = 0.075
hdr_height  = 0.085

# --- Шапка ---
for xi, lbl, w in zip(col_x, col_labels, col_widths):
    ax_tab.add_patch(mpatches.FancyBboxPatch(
        (xi, 1.0 - hdr_height), w - 0.006, hdr_height - 0.004,
        boxstyle="square,pad=0", linewidth=0.8,
        edgecolor="#888888", facecolor="#2C3E50",
        transform=ax_tab.transAxes, clip_on=False,
    ))
    ax_tab.text(
        xi + w / 2 - 0.003, 1.0 - hdr_height / 2, lbl,
        transform=ax_tab.transAxes,
        ha="center", va="center",
        fontsize=10.5, fontweight="bold", color="white",
    )

# --- Строки ---
y_cursor = 1.0 - hdr_height - 0.008
for cat_name, cat_data in categories.items():
    examples     = cat_data["examples"]
    block_h      = len(examples) * row_height

    # Столбец «Категория»
    ax_tab.add_patch(mpatches.FancyBboxPatch(
        (col_x[0], y_cursor - block_h), col_widths[0] - 0.006, block_h,
        boxstyle="square,pad=0", linewidth=0.6,
        edgecolor="#cccccc", facecolor=cat_data["color"] + "28",
        transform=ax_tab.transAxes, clip_on=False,
    ))
    ax_tab.text(
        col_x[0] + col_widths[0] / 2 - 0.003,
        y_cursor - block_h / 2,
        cat_name,
        transform=ax_tab.transAxes,
        ha="center", va="center",
        fontsize=10, fontweight="bold", color=cat_data["color"],
    )

    # Столбец «Кол-во»
    ax_tab.add_patch(mpatches.FancyBboxPatch(
        (col_x[1], y_cursor - block_h), col_widths[1] - 0.006, block_h,
        boxstyle="square,pad=0", linewidth=0.6,
        edgecolor="#cccccc", facecolor=cat_data["color"] + "14",
        transform=ax_tab.transAxes, clip_on=False,
    ))
    ax_tab.text(
        col_x[1] + col_widths[1] / 2 - 0.003,
        y_cursor - block_h / 2,
        str(cat_data["count"]),
        transform=ax_tab.transAxes,
        ha="center", va="center",
        fontsize=15, fontweight="bold", color=cat_data["color"],
    )

    # Столбец «Примеры» — по одной строке на пример
    for i, example in enumerate(examples):
        y_row  = y_cursor - i * row_height
        bg     = "#F5F5F5" if i % 2 == 0 else "white"
        ax_tab.add_patch(mpatches.FancyBboxPatch(
            (col_x[2], y_row - row_height), col_widths[2] - 0.006, row_height,
            boxstyle="square,pad=0", linewidth=0.4,
            edgecolor="#dddddd", facecolor=bg,
            transform=ax_tab.transAxes, clip_on=False,
        ))
        parts = example.split(" → ")
        line  = f"«{parts[0]}»  →  «{parts[1]}»" if len(parts) == 2 else example
        ax_tab.text(
            col_x[2] + 0.014,
            y_row - row_height / 2,
            line,
            transform=ax_tab.transAxes,
            ha="left", va="center",
            fontsize=9.5, color="#1a1a1a",
            fontfamily="monospace",
        )

    y_cursor -= block_h + 0.014

ax_tab.set_title("Репрезентативные примеры ошибок по категориям",
                 fontsize=11, fontweight="bold", pad=8, loc="left")


fig_tab.savefig("figure_4b_table.png", dpi=200, bbox_inches="tight", facecolor="white")
fig_tab.savefig("figure_4b_table.pdf", bbox_inches="tight", facecolor="white")
print("Сохранено: figure_4b_table.png / .pdf")

plt.show()
