import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import config as C

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#a3a29c"
BLUE_SEQ = "Blues"

GROUP_COLOR = dict(zip(C.ATTACK_GROUPS, SERIES))
MODEL_COLOR = {"Random Forest": SERIES[0], "CatBoost": SERIES[1], "Observed": NEUTRAL}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8,
})


def save(fig, name, tight=True):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(C.FIGURES / f"{name}.png", dpi=300)
    plt.close(fig)


def hbar(ax, labels, values, color=SERIES[0], fmt="{:.0f}", xlabel=""):
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, color=color, height=0.62, edgecolor=SURFACE, linewidth=1)
    for yi, v in zip(y, values):
        ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=8, color=INK2)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    ax.grid(axis="y", visible=False); ax.set_xlabel(xlabel)
    ax.set_xlim(0, max(values) * 1.18 if len(values) else 1)
