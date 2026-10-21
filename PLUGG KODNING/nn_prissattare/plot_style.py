import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["C0", "C1", "C2", "C3"]
SEQ_CMAP = "viridis"


def apply_style():
    plt.rcParams.update({
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "savefig.dpi": 150, "savefig.bbox": "tight",
    })
