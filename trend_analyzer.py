from pathlib import Path
from datetime import datetime
import statistics

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from player_database import get_player_history

METRICS = [
    "distance_covered",
    "avg_speed",
    "max_speed",
    "avg_shot_speed",
    "max_shot_speed",
    "consistency_score",
]

FLAT_THRESHOLD = 0.02  


def _linear_trend_slope(values):
    
    n = len(values)
    if n < 2:
        return 0.0

    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(values) / n

    numerator = sum((xs[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    denominator = sum((xs[i] - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def build_player_trend_report(player_name: str, output_dir: str, timestamp: str) -> str:
    
    history = get_player_history(player_name)

    if not history:
        return f"No history found for {player_name}."

    if len(history) == 1:
        return (
            f"=== Trend Report: {player_name} ===\n"
            f"This is the first recorded match for {player_name}. "
            f"No prior data to compare against yet."
        )

    lines = [
        f"=== Trend Report: {player_name} ===",
        f"Matches on record: {len(history)}",
        "",
    ]

    for metric in METRICS:
        values = [h[metric] for h in history if h[metric] is not None]
        if len(values) < 2:
            continue

        current = values[-1]
        past_values = values[:-1]
        career_avg = statistics.mean(past_values)
        slope = _linear_trend_slope(values)

        if slope > FLAT_THRESHOLD:
            direction = "improving"
        elif slope < -FLAT_THRESHOLD:
            direction = "declining"
        else:
            direction = "stable"

        delta_pct = ((current - career_avg) / career_avg * 100.0) if career_avg else 0.0

        lines.append(
            f"{metric:20s} current={current:8.2f} | career_avg={career_avg:8.2f} "
            f"| vs_avg={delta_pct:+6.1f}% | trend={direction} (slope={slope:+.3f})"
        )

    report_text = "\n".join(lines)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(output_dir) / f"{player_name}_trend_report_{timestamp}.txt"
    report_path.write_text(report_text)

    _plot_trend_chart(player_name, history, output_dir, timestamp)

    return report_text


def _plot_trend_chart(player_name, history, output_dir, timestamp):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"{player_name} — Performance Trend Across Matches")

    for ax, metric in zip(axes.flat, METRICS):
        values = [h[metric] for h in history if h[metric] is not None]
        if not values:
            ax.set_visible(False)
            continue

        ax.plot(range(1, len(values) + 1), values, marker="o", linewidth=2)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Match #")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = Path(output_dir) / f"{player_name}_trend_chart_{timestamp}.png"
    plt.savefig(chart_path)
    plt.close(fig)
