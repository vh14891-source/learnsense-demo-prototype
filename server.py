import json
import joblib
import pandas as pd
import os
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# trained models + evaluation results (created by train_model.py)
vark_model = joblib.load("models/vark_label.joblib")
depth_model = joblib.load("models/engagement_depth.joblib")
results = json.load(open("models/cv_results.json"))

TIPS = {
    "Visual": "Convert notes into diagrams, mind-maps and concept videos.",
    "Auditory": "Explain topics aloud, use lecture recordings, study in discussion groups.",
    "Read/Write": "Rewrite notes in your own words; use summaries and written practice answers.",
    "Kinesthetic": "Learn by doing: practice problems, labs, coding exercises, mini-projects.",
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/results")
def api_results():
    return jsonify(results)


@app.route("/models/<path:filename>")
def model_files(filename):
    return send_from_directory("models", filename)


@app.route("/api/predict", methods=["POST"])
def predict():
    d = request.get_json()
    try:
        video, audio = float(d["video"]), float(d["audio"])
        text, practical = float(d["text"]), float(d["practical"])
        days, hours = float(d["days"]), float(d["hours"])
        sess, rev = float(d["sess"]), float(d["rev"])
        retries, pref = float(d["retries"]), float(d["pref"])
        quiz = float(d["quiz"])
        subject, activity = d["subject"], d["activity"]
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Invalid input"}), 400

    tot = max(video + audio + text + practical, 1)
    row = pd.DataFrame([{
        "video_share": video / tot, "audio_share": audio / tot,
        "text_share": text / tot, "practical_share": practical / tot,
        "total_weekly_min": tot, "study_days_per_week": days,
        "daily_hours": hours, "avg_session_min": sess,
        "revision_freq": rev, "quiz_retries": retries,
        "subject_pref": pref, "quiz_score": quiz,
        "deep_indicator": rev + retries,
        "subject": subject, "activity_type": activity}])

    vp = dict(zip(vark_model.classes_, vark_model.predict_proba(row)[0]))
    dp = dict(zip(depth_model.classes_, depth_model.predict_proba(row)[0]))
    top_vark = max(vp, key=vp.get)
    top_depth = max(dp, key=dp.get)

    return jsonify({
        "vark": top_vark,
        "vark_conf": round(vp[top_vark] * 100),
        "vark_probs": {k: round(v * 100, 1) for k, v in vp.items()},
        "depth": top_depth,
        "depth_conf": round(dp[top_depth] * 100),
        "tip": TIPS[top_vark],
        "days": days, "hours": hours, "quiz": quiz,
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)