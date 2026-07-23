from gait_measurement_pipeline.gait_dataclasses.gait_evaluation_dataclass import (
    MetricStats,
)
import numpy as np
import matplotlib.pyplot as plt
import os


def compute_icc(x, y) -> float:
    """
    Compute ICC(2,1): two-way random effects, absolute agreement, single measurement.
    """
    n = len(x)
    if n < 2:
        return 0.0

    data = np.column_stack([x, y])
    k = data.shape[1]

    grand_mean = np.mean(data)
    row_means = np.mean(data, axis=1)
    col_means = np.mean(data, axis=0)

    ss_total = np.sum((data - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denominator = ms_rows + (k - 1) * ms_error + (k / n) * (ms_cols - ms_error)
    if denominator == 0:
        return 0.0

    return float((ms_rows - ms_error) / denominator)


# Map each data category to its correct unit label for axis/legend text
CATEGORY_UNITS = {
    "temporal": "sec",
    "spatial": "normalized",
    "derived": "",  # <-- set this once you know derived metrics' units (may vary per metric)
}


def generate_bland_altman_per_metric(
    data_dict,
    stats_dict,
    category,
    pid,
    save_dir="C:\\Users\\BhavyaSehgal\\Downloads\\Video_pre_preocessing\\bland_altman",
):
    """
    Generate one standalone Bland-Altman PNG per metric in a category
    (temporal/spatial/derived), saved individually rather than as a grid.

    Args:
        data_dict (dict): metric_name -> (x, y), e.g. temporal_data / spatial_data / derived_data
        stats_dict (dict): metric_name -> MetricStats, e.g. result.temporal_stats
        category (str): "temporal", "spatial", or "derived" -- used to select units
        pid (str): participant id, used in filename/title
        save_dir (str): output directory

    Returns:
        list[str]: paths to each saved PNG, one per metric
    """
    units = CATEGORY_UNITS.get(category, "")
    metric_names = list(data_dict.keys())
    if len(metric_names) == 0:
        return []

    os.makedirs(save_dir, exist_ok=True)
    outpaths = []

    for metric_name in metric_names:
        x, y = data_dict[metric_name]
        stats = stats_dict[metric_name]

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_bland_altman(
            x,
            y,
            bias=stats.bias,
            std=stats.std,
            loa_lower=stats.loa_lower,
            loa_upper=stats.loa_upper,
            metric_name=metric_name,
            units=units,
            ax=ax,
        )

        fig.suptitle(
            f"Bland-Altman: {metric_name} ({category.capitalize()})",
            fontsize=13,
        )
        plt.tight_layout()

        # Sanitize metric_name for use in filename (spaces/slashes etc.)
        safe_metric_name = (
            metric_name.replace(" ", "_").replace("/", "-").replace("\\", "-")
        )
        outpath = os.path.join(
            save_dir, f"{pid}_{category}_{safe_metric_name}_bland_altman.svg"
        )
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        outpaths.append(outpath)

    return outpaths


def plot_bland_altman(
    x,
    y,
    bias=None,
    std=None,
    loa_lower=None,
    loa_upper=None,
    metric_name="Metric",
    units="",
    ax=None,
):
    """
    Generate a Bland-Altman plot comparing two sets of measurements.

    Args:
        x (array-like): predicted / pipeline values
        y (array-like): reference / ground truth values
        bias, std, loa_lower, loa_upper (float, optional): precomputed stats.
            If None, they are computed here from x and y.
        metric_name (str): label for the metric being compared, used in titles/axes
        units (str): unit string (e.g., "cm", "deg", "s") appended to axis labels
        ax (matplotlib.axes.Axes, optional): axis to plot on. If None, a new
            figure/axis is created.

    Returns:
        matplotlib.axes.Axes: the axis the plot was drawn on
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    diff = x - y
    mean_vals = (x + y) / 2

    if bias is None:
        bias = np.mean(diff)
    if std is None:
        std = np.std(diff)
    if loa_lower is None:
        loa_lower = bias - 1.96 * std
    if loa_upper is None:
        loa_upper = bias + 1.96 * std

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    unit_suffix = f" ({units})" if units else ""

    # Scatter of individual differences
    ax.scatter(mean_vals, diff, alpha=0.6, edgecolor="k", linewidth=0.5)

    # Bias line
    ax.axhline(
        bias,
        color="blue",
        linestyle="-",
        linewidth=1.5,
        label=f"Bias = {bias:.2f}{unit_suffix}",
    )

    # Limits of agreement
    ax.axhline(
        loa_upper,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"+1.96 SD = {loa_upper:.2f}{unit_suffix}",
    )
    ax.axhline(
        loa_lower,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"-1.96 SD = {loa_lower:.2f}{unit_suffix}",
    )

    # Zero-difference reference line (for visual context)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)

    ax.set_xlabel(f"Mean of pipeline & reference{unit_suffix}")
    ax.set_ylabel(f"Difference (pipeline − reference){unit_suffix}")
    ax.set_title(f"Bland-Altman Plot: {metric_name}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    return ax


def compute_stats(x, y) -> MetricStats:
    """
    Compute statistical metrics comparing two sets of numeric values.

    Args:
        x (list or np.ndarray): predicted or pipeline values
        y (list or np.ndarray): reference or ground truth values

    Returns:
        MetricStats: object containing MAE, RMSE, bias, standard deviation,
        correlation, ICC, and Bland-Altman limits of agreement
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) == 0 or len(y) == 0:
        return MetricStats(
            mae=0, rmse=0, bias=0, std=0, r=0, icc=0, loa_lower=0, loa_upper=0
        )

    diff = x - y

    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))

    eps = 1e-8
    mean_percent_error = np.mean((np.abs(diff) / (np.abs(y) + eps)) * 100)

    bias = np.mean(diff)
    std = np.std(diff)
    loa_lower = bias - 1.96 * std
    loa_upper = bias + 1.96 * std

    r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
    icc = compute_icc(x, y)

    return MetricStats(
        mae=mae,
        rmse=rmse,
        percent_error=mean_percent_error,
        bias=bias,
        std=std,
        r=r,
        icc=icc,
        loa_lower=loa_lower,
        loa_upper=loa_upper,
    )
