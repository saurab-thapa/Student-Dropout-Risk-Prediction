from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


FIGURE_DIR = Path(__file__).resolve().parents[1] / "results/figures"


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.figsize": (8, 5), "axes.titlesize": 13})


def save_figure(name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=200, bbox_inches="tight")


def barplot(data, x: str, y: str, title: str, filename: str) -> None:
    set_style()
    sns.barplot(data=data, x=x, y=y, hue=x, legend=False, palette="deep")
    plt.title(title)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    save_figure(filename)
    plt.show()
