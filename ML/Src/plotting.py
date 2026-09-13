import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def set_plotting_style() -> None:
    """Configures global settings for warnings, pandas display options, matplotlib, and seaborn
    themes (GitHub dark mode color scheme).

    No arguments required.
    No return.
    """

    warnings.filterwarnings("ignore")

    ACCENT_PALETTE = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#ffa657', '#79c0ff', '#f85149', '#56d4dd']

    pd.set_option({
        "display.max_rows": None,
        "display.max_columns": None,
        "display.width": None,
        "display.max_colwidth": None,
    })

    plt.rcParams.update({
        'figure.facecolor': '#0d1117',

        'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',
        'axes.labelcolor': '#c9d1d9',
        'axes.titlecolor': '#e6edf3',
        'axes.prop_cycle': plt.cycler(color=ACCENT_PALETTE),

        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        'text.color': '#c9d1d9',

        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': '#cccccc',
        'grid.linestyle': "--",
        'grid.alpha': 0.75,

        'legend.facecolor': '#161b22',
        'legend.edgecolor': '#30363d',

        'font.family': 'DejaVu Sans',
        'axes.titlesize': 14,
        'axes.labelsize': 11,
    })

    sns.set_palette(ACCENT_PALETTE)