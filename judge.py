# score factual accuracy, completeness, clinical correctness
import os
import requests

JUDGE_SERVER_URL = os.environ.get("JUDGE_SERVER_URL", "http://localhost:8000")

def judge(actual, expected, session_id=None):
    response = requests.post(
        f"{JUDGE_SERVER_URL}/judge",
        json={"actual": actual, "expected": expected},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    print(f"SCORE: {data['score']}\nREASON: {data['reason']}")
    return data["score"]
