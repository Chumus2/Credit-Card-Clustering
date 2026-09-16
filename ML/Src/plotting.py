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


def draw_model_clusters(
    model,
    x,
    title: str,
    **fit_params
) -> None:
    """Draws 2D scatter plot for clusters fitted on x DataFrame.

    Fitting model x argument + extra fit_params.
    Checking if model has attribute labels_:
    1. if model has labels_:
        - finding model labels (model.labels_)
    2. otherwise:
        - predicting labels (model.predict(x))

    Creating copy of x DataFrame to avoid SettingWithCopyWarning.
    Sorting data by cluster to render noise (-1) on background.
    Generating palette with grey color (#555555) for noise (-1).
    Drawing seaborn scatterplot and showing figure.

    function required arguments:
    1. model (scikit-model) the trained model itself:
    2. x (pd.DataFrame) feature features that will be required for model prediction:
    3. title (str) plot title:
    4. **fit_params (optional) extra parameters for model fitting:

    No return.
    """

    data_plot = x.copy()
    model.fit(data_plot, **fit_params)

    if hasattr(model, 'labels_'):
        labels = model.labels_
    else:
        labels = model.predict(data_plot)

    data_plot["cluster"] = labels

    col_x = data_plot.columns[0]
    col_y = data_plot.columns[1]

    data_plot = data_plot.sort_values(by="cluster")

    unique_clusters = sorted(data_plot["cluster"].unique())
    real_clusters = [c for c in unique_clusters if c != -1]

    colors = sns.color_palette("tab10", n_colors=len(real_clusters))
    palette = {}
    color_idx = 0

    for c in unique_clusters:
        if c == -1:
            palette[c] = "#555555"  # Grey color for noise
        else:
            palette[c] = colors[color_idx]
            color_idx += 1

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=data_plot,
        x=col_x,
        y=col_y,
        hue="cluster",
        palette=palette,
        s=70,
        alpha=0.8,
    )

    plt.title(title)
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()