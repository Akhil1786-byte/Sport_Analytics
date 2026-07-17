import os
import uuid

from flask import Flask, render_template, request, redirect, url_for, session, Response

from pipeline_runner import process_video

app = Flask(__name__)
app.secret_key = "dev-secret-key"

UPLOAD_DIR = "input"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/")
def upload_page():
    return render_template("index.html", step="upload")


@app.route("/upload", methods=["POST"])
def upload():
    video_file = request.files.get("video")
    if not video_file or video_file.filename == "":
        return redirect(url_for("upload_page"))

    safe_name = f"{uuid.uuid4().hex}_{video_file.filename}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    video_file.save(save_path)

    session["video_path"] = save_path
    return redirect(url_for("names_page"))


@app.route("/names")
def names_page():
    if "video_path" not in session:
        return redirect(url_for("upload_page"))
    return render_template("index.html", step="names")


@app.route("/start", methods=["POST"])
def start():
    session["player1_name"] = request.form.get("player1", "").strip() or "Player1"
    session["player2_name"] = request.form.get("player2", "").strip() or "Player2"
    return redirect(url_for("watch_page"))


@app.route("/watch")
def watch_page():
    if "player1_name" not in session:
        return redirect(url_for("upload_page"))
    return render_template(
        "index.html",
        step="watch",
        player1_name=session["player1_name"],
        player2_name=session["player2_name"],
    )


@app.route("/video_feed")
def video_feed():
    video_path = session.get("video_path")
    player1_name = session.get("player1_name")
    player2_name = session.get("player2_name")

    if not video_path:
        return "No video queued", 400

    return Response(
        process_video(video_path, player1_name, player2_name),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
