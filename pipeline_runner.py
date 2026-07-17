import os
from datetime import datetime

import cv2

from motion_tracker import MotionTracker
from player_detector import PlayerDetector
from homography import CourtHomography
from player_tracker import PlayerTracker
from trajectory_manager import TrajectoryManager
from distance_tracker import DistanceTracker
from perspective_corrector import PerspectiveCorrector
from mini_court import MiniCourt
from court_line_detector import CourtLineDetector
from ball_detector import BallDetector
from ball_tracker import BallTracker
from ball_speed_tracker import BallSpeedTracker
from catboost_bounce_detector import CatBoostBounceDetector
from bounce_localizer import BounceLocalizer
from shot_detector import ShotDetector
from shot_speed_tracker import ShotSpeedTracker
from match_analytics import MatchAnalyticsRecorder
from match_analytics_report import MatchAnalyticsReport
from court_heatmap import CourtHeatmapGenerator
from player_database import init_db, save_match_stats
from trend_analyzer import build_player_trend_report, build_match_comparison_chart
from player_pose import PlayerPoseEstimator


def process_video(video_path, player1_name, player2_name, output_root="output"):
    
    init_db()

    cap = cv2.VideoCapture(video_path)

    os.makedirs(output_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    player_names = {1: player1_name, 2: player2_name}

    match_output_dir = os.path.join(output_root, f"{timestamp}_{player1_name}_vs_{player2_name}")
    os.makedirs(match_output_dir, exist_ok=True)

    summary_dir = os.path.join(output_root, "player_summaries")
    os.makedirs(summary_dir, exist_ok=True)

    output_path = os.path.join(match_output_dir, f"tennis_output_{timestamp}.mp4")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    detector = PlayerDetector()
    motion_tracker = MotionTracker()
    ball_detector = BallDetector("models/yolo5_last.pt")
    ball_tracker = BallTracker()
    ball_speed_tracker = BallSpeedTracker()

    bounce_detector = CatBoostBounceDetector(model_path="models/ctb_regr_bounce.cbm", threshold=0.45)
    bounce_localizer = BounceLocalizer(max_recent=10, display_lifetime_frames=int(fps * 1.5))

    shot_detector = ShotDetector(
        proximity_threshold=3.0, speed_threshold=2.0, speed_multiplier=1.2, cooldown_frames=8
    )
    shot_speed_tracker = ShotSpeedTracker(fps=fps, max_plausible_speed=55.0)

    match_analytics_recorder = MatchAnalyticsRecorder(fps=fps)

    player_speeds = {1: 0.0, 2: 0.0}
    speed_sum = {1: 0.0, 2: 0.0}
    speed_sample_count = {1: 0, 2: 0}

    ret, first_frame = cap.read()
    if not ret:
        raise ValueError(f"Could not read video: {video_path}")

    court_detector = CourtLineDetector("models/keypoints_model.pth")
    keypoints = court_detector.predict(first_frame)

    homography = CourtHomography()
    H = homography.compute(keypoints)

    reference_pixel_point = (
        int((keypoints[0] + keypoints[2]) / 2),
        int((keypoints[1] + keypoints[5]) / 2),
    )
    perspective_corrector = PerspectiveCorrector(H, reference_pixel_point=reference_pixel_point)

    tracker = PlayerTracker(homography, H)
    trajectory_manager = TrajectoryManager()
    distance_tracker = DistanceTracker()
    mini_court = MiniCourt()
    pose_estimator = PlayerPoseEstimator(model_complexity=0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = detector.detect(frame)
            ball_detections = ball_detector.detect(frame)
            ball_center, ball_is_real = ball_tracker.update(ball_detections)

            players = tracker.get_players(results)

            player_court_positions = {}
            player_pixel_positions = {}

            for player in players:
                x1, y1, x2, y2 = player["bbox"]
                player_number = player["player_number"]
                player_pixel = (int((x1 + x2) / 2), int(y2 - (y2 - y1) * 0.08))
                player_pixel_positions[player_number] = player_pixel
                court_x, court_y = homography.pixel_to_court(player_pixel, H)
                player_court_positions[player_number] = (court_x, court_y)

            ball_speed = 0.0
            ball_position = None

            if ball_center is not None:
                ball_court_x, ball_court_y = homography.pixel_to_court(ball_center, H)
                ball_position = {"court_x": ball_court_x, "court_y": ball_court_y}
                ball_corrected_point = perspective_corrector.pixel_to_corrected_point(
                    ball_center, anchor_court_pos=(0.0, 0.0)
                )
                ball_speed = ball_speed_tracker.update(ball_corrected_point, fps, is_real_detection=ball_is_real)
            else:
                ball_speed = ball_speed_tracker.update(None, fps, is_real_detection=False)

            bounce_event = bounce_detector.update(ball_pixel_pos=ball_center, is_real_detection=ball_is_real)

            if bounce_event is not None:
                bounce_court_pos = homography.pixel_to_court(bounce_event["pixel_pos"], H)
                NET_EXCLUSION_MARGIN = 1.8
                distance_from_net = abs(bounce_court_pos[1] - 11.885)
                if distance_from_net <= NET_EXCLUSION_MARGIN:
                    bounce_event = None

            if bounce_event is not None:
                localizer_event = {
                    "frame_idx": bounce_event["frame_idx"],
                    "pixel_pos": bounce_event["pixel_pos"],
                    "court_pos": bounce_court_pos,
                }
                bounce_localizer.register_bounce(localizer_event, current_frame_idx=frame_idx)

            active_bounce_markers = bounce_localizer.get_active_markers(frame_idx)
            for marker in active_bounce_markers:
                mbx, mby = marker["pixel_pos"]
                marker_color = (0, 255, 0) if marker["in_bounds"] else (0, 0, 255)
                cv2.circle(frame, (int(mbx), int(mby)), 9, marker_color, -1)
                cv2.circle(frame, (int(mbx), int(mby)), 9, (255, 255, 255), 2)

            shot_event = None
            if ball_center is not None:
                shot_event = shot_detector.update(
                    ball_court_pos=(ball_position["court_x"], ball_position["court_y"]) if ball_position else None,
                    ball_speed=ball_speed,
                    player_court_positions=player_court_positions,
                )
            else:
                shot_event = shot_detector.update(ball_court_pos=None, ball_speed=0.0, player_court_positions=player_court_positions)

            if shot_event is not None:
                shot_speed_record = shot_speed_tracker.register_shot(shot_event)
                if shot_speed_record is not None and shot_speed_record["hit_by"] in (1, 2):
                    match_analytics_recorder.record_shot(
                        shot_speed_record["hit_by"], shot_event["frame_idx"], shot_speed_record["speed"]
                    )

            if ball_center is not None:
                bx, by = ball_center
                ball_color = (0, 255, 255) if ball_is_real else (0, 165, 255)
                cv2.circle(frame, (int(bx), int(by)), 6, ball_color, -1)
                cv2.putText(frame, "BALL", (int(bx) + 10, int(by) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ball_color, 2)

            history = ball_tracker.get_history()
            for i in range(1, len(history)):
                cv2.line(frame, history[i - 1], history[i], (0, 255, 255), 2)

            mini_players = []

            if not tracker.locked:
                cv2.putText(frame, "Initializing Player IDs...", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            for player in players:
                x1, y1, x2, y2 = player["bbox"]
                player_number = player["player_number"]
                court_x, court_y = player_court_positions[player_number]

                trajectory_manager.update(player_number, court_x, court_y)
                speed = motion_tracker.update(player_number, court_x, court_y, fps)
                player_speeds[player_number] = speed
                speed_sum[player_number] += speed
                speed_sample_count[player_number] += 1

                corrected_increment = perspective_corrector.get_corrected_distance(
                    player_number, player_pixel_positions[player_number]
                )
                distance_tracker.update(player_number, corrected_increment)
                distance_tracker.update_speed(player_number, speed)

                recorded_speed = 0.0 if motion_tracker.was_last_update_rejected(player_number) else speed

                box_color = (0, 255, 0) if player_number == 1 else (0, 0, 255)

                keypoints_pose = pose_estimator.get_keypoints(frame, (x1, y1, x2, y2))
                elbow_angles = pose_estimator.get_elbow_angles(keypoints_pose)

                match_analytics_recorder.record_frame(
                    player_number, frame_idx, corrected_increment, recorded_speed,
                    court_x=court_x, court_y=court_y,
                    left_elbow_angle=elbow_angles["left_elbow_angle"],
                    right_elbow_angle=elbow_angles["right_elbow_angle"],
                )

                mini_players.append({"player_number": player_number, "court_x": court_x, "court_y": court_y})

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    frame, player_names.get(player_number, f"P{player_number}"),
                    (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2
                )
                pose_estimator.draw_keypoints(frame, keypoints_pose, color=box_color)

            distances = {1: distance_tracker.get_distance(1), 2: distance_tracker.get_distance(2)}
            speeds = {1: player_speeds.get(1, 0.0), 2: player_speeds.get(2, 0.0)}
            shot_speeds_display = {1: shot_speed_tracker.get_last_speed(1), 2: shot_speed_tracker.get_last_speed(2)}
            avg_speeds_display = {
                1: (speed_sum[1] / speed_sample_count[1]) if speed_sample_count[1] > 0 else 0.0,
                2: (speed_sum[2] / speed_sample_count[2]) if speed_sample_count[2] > 0 else 0.0,
            }
            avg_shot_speeds_display = {1: shot_speed_tracker.get_average_speed(1), 2: shot_speed_tracker.get_average_speed(2)}

            frame = mini_court.draw(
                frame, mini_players, distances, speeds, ball_position,
                bounce_markers=active_bounce_markers, ball_speed=ball_speed,
                shot_speeds=shot_speeds_display, avg_speeds=avg_speeds_display,
                avg_shot_speeds=avg_shot_speeds_display,
            )

            out.write(frame)

            
            ok, buffer = cv2.imencode(".jpg", frame)
            if ok:
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )

            frame_idx += 1

    finally:
        cap.release()
        out.release()
        pose_estimator.close()

    # =====================================
    # POST-PROCESSING 
    # =====================================
    match_duration_seconds = frame_idx / fps

    analytics_report = MatchAnalyticsReport(match_analytics_recorder, match_duration_seconds=match_duration_seconds)
    heatmap_generator = CourtHeatmapGenerator()

    heatmap_zones = {}
    heatmap_paths = {}
    fatigue_curve_paths = {}

    for player_number in (1, 2):
        samples = match_analytics_recorder.get_samples(player_number)
        heatmap_zones[player_number] = heatmap_generator.get_dominant_zone(samples)

        heatmap_path = os.path.join(match_output_dir, f"heatmap_player{player_number}_{timestamp}.png")
        heatmap_generator.save_heatmap(samples, heatmap_path, player_label=f"Player {player_number}")
        heatmap_paths[player_number] = heatmap_path

        fatigue_curve_path = os.path.join(match_output_dir, f"fatigue_curve_player{player_number}_{timestamp}.png")
        analytics_report.save_fatigue_curve_chart(player_number, fatigue_curve_path, player_label=player_names[player_number])
        fatigue_curve_paths[player_number] = fatigue_curve_path

    report_path = os.path.join(match_output_dir, f"match_analytics_{timestamp}.txt")
    report_data = analytics_report.generate()
    analytics_report.save_text_report(report_path, heatmap_zones=heatmap_zones)

    video_filename = os.path.basename(video_path)

    for player_number in (1, 2):
        p = report_data["players"][player_number]
        stats_for_db = {
            "distance_covered": p["total_distance"],
            "avg_speed": p["avg_movement_speed"],
            "max_speed": p["max_movement_speed"],
            "avg_shot_speed": p["avg_shot_speed"],
            "max_shot_speed": p["max_shot_speed"],
            "consistency_score": p["consistency_score"],
            "shot_count": p["total_shots"],
        }
        segments_for_db = [
            {
                "start_seconds": seg["start_seconds"],
                "end_seconds": seg["end_seconds"],
                "avg_speed": seg["avg_speed"],
                "avg_shot_speed": seg["avg_shot_speed"],
                "consistency_score": seg["consistency_score"],
            }
            for seg in p["segments"]
        ]
        save_match_stats(player_names[player_number], video_filename, stats_for_db, segments=segments_for_db)

    for player_number in (1, 2):
        build_player_trend_report(player_names[player_number], summary_dir, "latest")
        build_match_comparison_chart(player_names[player_number], summary_dir, "latest")
