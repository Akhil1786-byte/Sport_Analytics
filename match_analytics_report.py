import statistics
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt


class MatchAnalyticsReport:
   

    SEGMENT_SECONDS = 15.0

    
    WEIGHTS = {
        "distance": 0.30,    
        "shot_speed": 0.40,  
        "consistency": 0.30, 
    }

    def __init__(self, recorder, match_duration_seconds):
        self.recorder = recorder
        self.match_duration_seconds = match_duration_seconds

    # -----------------------------------------------------------
    # SEGMENTS
    # -----------------------------------------------------------
    def _build_segments(self, player_number):
        samples = self.recorder.get_samples(player_number)

        
        num_segments = max(
            1,
            -(-int(self.match_duration_seconds) // int(self.SEGMENT_SECONDS))
        )

        segments = []
        for seg_idx in range(num_segments):
            seg_start = seg_idx * self.SEGMENT_SECONDS
            seg_end = seg_start + self.SEGMENT_SECONDS

            seg_samples = [s for s in samples if seg_start <= s["seconds"] < seg_end]

            total_distance = sum(s["distance_delta"] for s in seg_samples)
            speeds = [s["speed"] for s in seg_samples if s["speed"] > 0]
            shot_speeds = [s["shot_speed"] for s in seg_samples if s["shot_speed"] is not None]

            segments.append({
                "segment_index": seg_idx,
                "start_seconds": seg_start,
                "end_seconds": seg_end,
                "distance": total_distance,
                "avg_speed": statistics.mean(speeds) if speeds else 0.0,
                "shot_count": len(shot_speeds),
                "avg_shot_speed": statistics.mean(shot_speeds) if shot_speeds else 0.0,
            })

        return segments

    # -----------------------------------------------------------
    # MOVEMENT TREND
    # -----------------------------------------------------------
    @staticmethod
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

    def _movement_trend(self, segments):
        avg_speeds = [seg["avg_speed"] for seg in segments]
        slope = self._linear_trend_slope(avg_speeds)

        
        FLAT_THRESHOLD = 0.02

        if slope > FLAT_THRESHOLD:
            label = "increasing"
        elif slope < -FLAT_THRESHOLD:
            label = "decreasing"
        else:
            label = "flat"

        return {"slope": slope, "label": label}

    # -----------------------------------------------------------
    # CONSISTENCY SCORE
    # -----------------------------------------------------------
    @staticmethod
    def _coefficient_of_variation(values):
        if len(values) < 2:
            return None  

        mean = statistics.mean(values)
        if mean == 0:
            return None

        stdev = statistics.stdev(values)
        return stdev / mean

    def _consistency_score(self, player_number):
        samples = self.recorder.get_samples(player_number)

        movement_speeds = [s["speed"] for s in samples if s["speed"] > 0]
        shot_speeds = [s["shot_speed"] for s in samples if s["shot_speed"] is not None]

        cov_movement = self._coefficient_of_variation(movement_speeds)
        cov_shot = self._coefficient_of_variation(shot_speeds)

        
        clamped_covs = [min(c, 1.0) for c in (cov_movement, cov_shot) if c is not None]

        if not clamped_covs:
            return 0.0

        avg_cov = sum(clamped_covs) / len(clamped_covs)

        
        score = max(0.0, 100.0 * (1.0 - min(avg_cov, 1.0)))

        return score

    # -----------------------------------------------------------
    # FATIGUE ANALYSIS
    # -----------------------------------------------------------
    def _fatigue_analysis(self, player_number):
        
        samples = self.recorder.get_samples(player_number)

        if not samples:
            return {
                "movement_change_pct": 0.0,
                "shot_speed_change_pct": 0.0,
                "fatigue_index": 100.0,
                "note": "insufficient data",
            }

        midpoint_seconds = self.match_duration_seconds / 2.0

        first_half = [s for s in samples if s["seconds"] < midpoint_seconds]
        second_half = [s for s in samples if s["seconds"] >= midpoint_seconds]

        first_movement = [s["speed"] for s in first_half if s["speed"] > 0]
        second_movement = [s["speed"] for s in second_half if s["speed"] > 0]

        first_shots = [s["shot_speed"] for s in first_half if s["shot_speed"] is not None]
        second_shots = [s["shot_speed"] for s in second_half if s["shot_speed"] is not None]

        movement_change_pct = self._pct_change(first_movement, second_movement)
        shot_speed_change_pct = self._pct_change(first_shots, second_shots)

        
        declines = [c for c in (movement_change_pct, shot_speed_change_pct) if c is not None and c < 0]

        if not declines:
            fatigue_index = 100.0
        else:
            avg_decline_pct = sum(declines) / len(declines)  
           
            fatigue_index = max(0.0, 100.0 + (avg_decline_pct * 2))

        return {
            "movement_change_pct": movement_change_pct,
            "shot_speed_change_pct": shot_speed_change_pct,
            "fatigue_index": fatigue_index,
        }

    @staticmethod
    def _pct_change(first_half_values, second_half_values):
        
        if not first_half_values or not second_half_values:
            return None

        first_avg = statistics.mean(first_half_values)
        second_avg = statistics.mean(second_half_values)

        if first_avg == 0:
            return None

        return ((second_avg - first_avg) / first_avg) * 100.0

    # -----------------------------------------------------------
    # SEGMENT-BY-SEGMENT FATIGUE CURVE (NEW)
    # -----------------------------------------------------------
    def save_fatigue_curve_chart(self, player_number, filepath, player_label=None):
        
        segments = self._build_segments(player_number)
        label = player_label or f"Player {player_number}"

        # Use segment midpoint as the x-axis time value
        times = [(seg["start_seconds"] + seg["end_seconds"]) / 2.0 for seg in segments]
        avg_speeds = [seg["avg_speed"] for seg in segments]
        avg_shot_speeds = [seg["avg_shot_speed"] for seg in segments]

        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(times, avg_speeds, marker="o", color="tab:blue", linewidth=2, label="Avg Movement Speed (m/s)")
        ax1.set_xlabel("Match time (s)")
        ax1.set_ylabel("Movement Speed (m/s)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(times, avg_shot_speeds, marker="s", color="tab:red", linewidth=2, label="Avg Shot Speed (m/s)")
        ax2.set_ylabel("Shot Speed (m/s)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        fig.suptitle(f"{label} — Fatigue Curve (Speed Across Match Segments)")

        # Combined legend across both y-axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        fig.tight_layout()
        plt.savefig(filepath)
        plt.close(fig)

        return filepath

    # -----------------------------------------------------------
    # PERFORMANCE INDEX
    # -----------------------------------------------------------
    @staticmethod
    def _normalize(value, reference_max):
        
        if reference_max <= 0:
            return 0.0
        return max(0.0, min(100.0, (value / reference_max) * 100.0))

    def _performance_index(self, player_number, total_distance, avg_shot_speed, consistency_score,
                            reference_distance_max, reference_shot_speed_max):

        norm_distance = self._normalize(total_distance, reference_distance_max)
        norm_shot_speed = self._normalize(avg_shot_speed, reference_shot_speed_max)
        norm_consistency = consistency_score  # already 0-100

        index = (
            self.WEIGHTS["distance"] * norm_distance
            + self.WEIGHTS["shot_speed"] * norm_shot_speed
            + self.WEIGHTS["consistency"] * norm_consistency
        )

        return index

    # -----------------------------------------------------------
    # PUBLIC: BUILD FULL REPORT
    # -----------------------------------------------------------
    def generate(self):
        report = {"players": {}}

        player_totals = {}
        for player_number in (1, 2):
            samples = self.recorder.get_samples(player_number)
            total_distance = sum(s["distance_delta"] for s in samples)
            shot_speeds = [s["shot_speed"] for s in samples if s["shot_speed"] is not None]
            avg_shot_speed = statistics.mean(shot_speeds) if shot_speeds else 0.0
            player_totals[player_number] = (total_distance, avg_shot_speed)

        
        reference_distance_max = max(t[0] for t in player_totals.values()) or 1.0
        reference_shot_speed_max = max(t[1] for t in player_totals.values()) or 1.0

        for player_number in (1, 2):
            samples = self.recorder.get_samples(player_number)
            segments = self._build_segments(player_number)
            trend = self._movement_trend(segments)
            consistency = self._consistency_score(player_number)
            fatigue = self._fatigue_analysis(player_number)

            total_distance, avg_shot_speed = player_totals[player_number]
            shot_speeds = [s["shot_speed"] for s in samples if s["shot_speed"] is not None]
            movement_speeds = [s["speed"] for s in samples if s["speed"] > 0]

            performance_index = self._performance_index(
                player_number, total_distance, avg_shot_speed, consistency,
                reference_distance_max, reference_shot_speed_max,
            )

            report["players"][player_number] = {
                "total_distance": total_distance,
                "avg_movement_speed": statistics.mean(movement_speeds) if movement_speeds else 0.0,
                "max_movement_speed": max(movement_speeds) if movement_speeds else 0.0,
                "total_shots": len(shot_speeds),
                "avg_shot_speed": avg_shot_speed,
                "max_shot_speed": max(shot_speeds) if shot_speeds else 0.0,
                "movement_trend": trend,
                "consistency_score": consistency,
                "performance_index": performance_index,
                "fatigue": fatigue,
                "segments": segments,
            }

        return report

    # -----------------------------------------------------------
    # FORMAT AS TEXT REPORT
    # -----------------------------------------------------------
    def generate_text_report(self, heatmap_zones=None):
        
        report = self.generate()
        heatmap_zones = heatmap_zones or {}

        lines = []
        lines.append("=" * 60)
        lines.append("MATCH ANALYTICS REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Match duration: {self.match_duration_seconds:.1f}s")
        lines.append(f"Segment length: {self.SEGMENT_SECONDS}s")
        lines.append("=" * 60)

        for player_number in (1, 2):
            p = report["players"][player_number]
            fatigue = p["fatigue"]

            lines.append("")
            lines.append(f"--- PLAYER {player_number} ---")
            lines.append(f"Total distance covered: {p['total_distance']:.2f}m")
            lines.append(f"Average movement speed: {p['avg_movement_speed']:.2f} m/s")
            lines.append(f"Max movement speed: {p['max_movement_speed']:.2f} m/s")
            lines.append(f"Total shots: {p['total_shots']}")
            lines.append(f"Average shot speed: {p['avg_shot_speed']:.2f} m/s")
            lines.append(f"Max shot speed: {p['max_shot_speed']:.2f} m/s")
            lines.append(
                f"Movement trend: {p['movement_trend']['label']} "
                f"(slope={p['movement_trend']['slope']:.4f})"
            )
            lines.append(f"Consistency score: {p['consistency_score']:.1f} / 100")
            lines.append(f"Performance index: {p['performance_index']:.1f} / 100")

            lines.append("")
            lines.append("Fatigue analysis (first half vs second half of match):")
            if fatigue["movement_change_pct"] is not None:
                lines.append(f"  Movement speed change: {fatigue['movement_change_pct']:+.1f}%")
            else:
                lines.append("  Movement speed change: insufficient data")
            if fatigue["shot_speed_change_pct"] is not None:
                lines.append(f"  Shot speed change: {fatigue['shot_speed_change_pct']:+.1f}%")
            else:
                lines.append("  Shot speed change: insufficient data")
            lines.append(f"  Fatigue index: {fatigue['fatigue_index']:.1f} / 100 (100 = no decline detected)")
            lines.append("  See fatigue curve chart for the full segment-by-segment shape.")

            if player_number in heatmap_zones:
                lines.append("")
                lines.append(f"Dominant court zone: {heatmap_zones[player_number]}")

            lines.append("")
            lines.append(f"Segment breakdown (every {self.SEGMENT_SECONDS:.0f}s):")
            lines.append(
                f"{'Seg':>4} {'Time':>14} {'Dist(m)':>9} {'AvgSpd':>8} "
                f"{'Shots':>6} {'AvgShotSpd':>11}"
            )
            for seg in p["segments"]:
                time_range = f"{seg['start_seconds']:.0f}-{seg['end_seconds']:.0f}s"
                lines.append(
                    f"{seg['segment_index']:>4} {time_range:>14} "
                    f"{seg['distance']:>9.2f} {seg['avg_speed']:>8.2f} "
                    f"{seg['shot_count']:>6} {seg['avg_shot_speed']:>11.2f}"
                )

        lines.append("")
        lines.append("=" * 60)
        lines.append("NOTES ON METRICS")
        lines.append("=" * 60)
        lines.append(
            "Consistency score: based on coefficient of variation (stdev/mean)\n"
            "of both movement speed and shot speed combined. 100 = perfectly\n"
            "steady; lower scores indicate more erratic speed swings."
        )
        lines.append(
            "Performance index: weighted composite of distance covered (30%),\n"
            "average shot speed (40%), and consistency (30%), each normalized\n"
            "against the higher of the two players in THIS match -- it is a\n"
            "relative, within-match score, not an absolute external benchmark."
        )
        lines.append(
            "Fatigue index: compares first-half vs second-half average movement\n"
            "and shot speed. 100 = no decline detected (player maintained or\n"
            "improved pace); lower values indicate a larger drop-off, a common\n"
            "signal of physical fatigue over the course of a match. The fatigue\n"
            "curve chart shows the same underlying data as a continuous line\n"
            "across every segment, rather than a single before/after number."
        )

        return "\n".join(lines)

    def save_text_report(self, filepath, heatmap_zones=None):
        text = self.generate_text_report(heatmap_zones=heatmap_zones)
        with open(filepath, "w") as f:
            f.write(text)
        return filepath
