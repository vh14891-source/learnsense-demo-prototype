import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)
rng = np.random.default_rng(42)

N_STUDENTS = 300
SUBJECTS = ["Maths", "Programming", "Physics"]
VARK = ["Visual", "Auditory", "Read/Write", "Kinesthetic"]

rows = []
for sid in range(1, N_STUDENTS + 1):
    vark = str(rng.choice(VARK, p=[0.30, 0.15, 0.25, 0.30]))
    base_days = int(rng.integers(2, 7))
    base_hours = rng.uniform(1.0, 4.0)

    for subj in SUBJECTS:
        w = {"Visual": [3, 1, 1, 1.2],
             "Auditory": [1.2, 3, 1, 1],
             "Read/Write": [1, 1, 3, 1],
             "Kinesthetic": [1, 1, 1, 3]}[vark]
        w = np.array(w) * rng.uniform(0.6, 1.4, 4)
        w = w / w.sum()
        total = base_hours * base_days * 60 * rng.uniform(0.7, 1.3)
        video, audio, text, practical = (w * total).round(0)

        sess_len = np.clip(rng.normal(base_hours * 25, 12), 10, 120).round(0)
        revision = np.clip(rng.poisson(base_days * 0.8), 0, 10)
        retries = np.clip(rng.poisson(2.0), 0, 8)
        quiz = np.clip(55 + 4 * revision + 0.15 * sess_len
                       + 2 * retries + rng.normal(0, 8), 20, 100).round(0)
        subj_pref = int(rng.integers(1, 6))

        score = (0.45 * revision + 0.35 * retries + 0.03 * sess_len
                 + 0.3 * subj_pref + rng.normal(0, 1.0))
        depth = "Deep" if score > 4.6 else "Surface"

        acts = ["Memorization", "Problem-solving", "Reading", "Revision"]
        pa = {"Visual": [.2, .2, .2, .4],
              "Auditory": [.4, .2, .2, .2],
              "Read/Write": [.2, .1, .5, .2],
              "Kinesthetic": [.1, .6, .1, .2]}[vark]
        activity = str(rng.choice(acts, p=pa))

        label = vark if rng.random() > 0.10 else str(rng.choice(VARK))

        rows.append([sid, subj, activity, video, audio, text, practical,
                     base_days, round(base_hours, 1), sess_len, revision,
                     retries, subj_pref, quiz, label, depth])

cols = ["student_id", "subject", "activity_type", "video_min", "audio_min",
        "text_min", "practical_min", "study_days_per_week", "daily_hours",
        "avg_session_min", "revision_freq", "quiz_retries", "subject_pref",
        "quiz_score", "vark_label", "engagement_depth"]
df = pd.DataFrame(rows, columns=cols)

for c in ["avg_session_min", "quiz_score", "revision_freq"]:
    df.loc[df.sample(frac=0.03, random_state=1).index, c] = np.nan

df.to_csv("data/students.csv", index=False)
print(df.shape, "-> saved to data/students.csv")
print(df.vark_label.value_counts())
