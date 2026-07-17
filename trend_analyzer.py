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


def build_match_comparison_chart(player_name: str, output_dir: str, timestamp: str) -> str:
    
    history = get_player_history(player_name)

    if len(history) < 2:
        return (
            f"Need at least 2 matches on record for {player_name} to build a "
            f"match comparison chart. Currently have {len(history)}."
        )

    match1 = history[-2]
    match2 = history[-1]

    seg1 = match1.get("segments") or []
    seg2 = match2.get("segments") or []

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{player_name} — Match Comparison (Previous vs Current)")

    color_prev = "tab:gray"
    color_curr = "tab:blue"

    def _line_panel(ax, metric_key, title, ylabel):
        if not seg1 and not seg2:
            ax.text(
                0.5, 0.5, "No segment data available",
                ha="center", va="center", transform=ax.transAxes, fontsize=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)
            return

        if seg1:
            t1 = [(s["start_seconds"] + s["end_seconds"]) / 2.0 for s in seg1]
            ax.plot(t1, [s.get(metric_key, 0.0) for s in seg1], marker="o", color=color_prev, linewidth=2, label="Previous Match")
        if seg2:
            t2 = [(s["start_seconds"] + s["end_seconds"]) / 2.0 for s in seg2]
            ax.plot(t2, [s.get(metric_key, 0.0) for s in seg2], marker="o", color=color_curr, linewidth=2, label="Current Match")

        ax.set_title(title)
        ax.set_xlabel("Match time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # --- Panel 1: avg movement speed over time ---
    _line_panel(axes[0, 0], "avg_speed", "Avg Movement Speed", "Speed (m/s)")

    # --- Panel 2: avg shot speed over time ---
    _line_panel(axes[0, 1], "avg_shot_speed", "Avg Shot Speed", "Speed (m/s)")

    # --- Panel 3: consistency score over time ---
    _line_panel(axes[1, 0], "consistency_score", "Consistency Score", "Score (/100)")

    # --- Panel 4: fatigue curves overlaid (movement + shot speed, twin axis) ---
    ax = axes[1, 1]
    ax2 = ax.twinx()

    if not seg1 and not seg2:
        ax.text(
            0.5, 0.5,
            "No segment data available\n(matches saved before this feature was added)",
            ha="center", va="center", transform=ax.transAxes, fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax2.set_yticks([])
    else:
        if seg1:
            t1 = [(s["start_seconds"] + s["end_seconds"]) / 2.0 for s in seg1]
            ax.plot(t1, [s["avg_speed"] for s in seg1], color=color_prev, linewidth=2, label="Prev - Movement Speed")
            ax2.plot(t1, [s["avg_shot_speed"] for s in seg1], color=color_prev, linestyle="--", linewidth=2, label="Prev - Shot Speed")
        if seg2:
            t2 = [(s["start_seconds"] + s["end_seconds"]) / 2.0 for s in seg2]
            ax.plot(t2, [s["avg_speed"] for s in seg2], color=color_curr, linewidth=2, label="Current - Movement Speed")
            ax2.plot(t2, [s["avg_shot_speed"] for s in seg2], color=color_curr, linestyle="--", linewidth=2, label="Current - Shot Speed")

        ax.set_xlabel("Match time (s)")
        ax.set_ylabel("Movement Speed (m/s)")
        ax2.set_ylabel("Shot Speed (m/s)")
        ax.grid(True, alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax.set_title("Fatigue Curve Overlay")

    plt.tight_layout()
    chart_path = Path(output_dir) / f"{player_name}_match_comparison_{timestamp}.png"
    plt.savefig(chart_path)
    plt.close(fig)

    return f"Match comparison chart saved to {chart_path}"
