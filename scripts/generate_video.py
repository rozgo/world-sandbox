"""Generate a video with WaveSpeed text-to-video: submit once, save the task ID, resume to download.

    python scripts/generate_video.py submit --prompt docs/synthetic_slam/prompts/junkyard_v1.txt \
        --job outputs/synthetic_slam/junkyard_v1 --duration 6
    python scripts/generate_video.py collect --job outputs/synthetic_slam/junkyard_v1 --wait 900

`submit` creates one paid generation and refuses to run again for the same job folder. `collect`
polls the saved task and downloads the video; it never submits. The key is read from
WAVESPEED_API_KEY in the environment or the repository's ignored .env.
"""

import argparse
import hashlib
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.wavespeed.ai/api/v3"
MODEL = "bytedance/seedance-2.5/text-to-video"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def key():
    if os.environ.get("WAVESPEED_API_KEY"):
        return os.environ["WAVESPEED_API_KEY"]
    env = ROOT/".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        name, _, value = line.partition("=")
        if name.strip() == "WAVESPEED_API_KEY":
            return value.strip().strip("'\"")
    raise SystemExit("WAVESPEED_API_KEY is not set (environment or .env)")


def request(method, url, body=None):
    req = urllib.request.Request(url, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def save(path, record):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1) + "\n")
    tmp.replace(path)


def submit(args):
    job = Path(args.job)
    job.mkdir(parents=True, exist_ok=True)
    path = job/"job.json"
    prompt = Path(args.prompt).read_text().strip()
    body = {"prompt": prompt, "duration": args.duration, "resolution": args.resolution,
            "aspect_ratio": args.aspect_ratio, "generate_audio": False}
    with open(path, "x") as f:  # exclusive create: one submission per job folder
        json.dump({"status": "submitting", "created_at": now()}, f)
    record = {"model": MODEL, "request": body, "prompt_file": str(args.prompt),
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "created_at": now()}
    data = request("POST", f"{API}/{MODEL}", body)["data"]
    record.update(task_id=data["id"], status=data.get("status", "created"), submitted_at=now())
    save(path, record)
    print(json.dumps({"task_id": record["task_id"], "status": record["status"]}))


def collect(args):
    job = Path(args.job)
    path = job/"job.json"
    record = json.loads(path.read_text())
    if not record.get("task_id"):
        raise SystemExit("no task ID saved; check the provider's history before submitting again")
    deadline = time.time() + args.wait
    while True:
        data = request("GET", f"{API}/predictions/{record['task_id']}/result")["data"]
        status = data.get("status")
        if status in ("completed", "failed") or time.time() > deadline:
            break
        time.sleep(10)
    record.update(status=status, checked_at=now(), error=data.get("error") or None)
    if status == "completed":
        url = data["outputs"][0]
        video = job/"video.mp4"
        urllib.request.urlretrieve(url, video)
        record.update(output_url=url, video=str(video), video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                      timings=data.get("timings"), completed_at=now())
    save(path, record)
    print(json.dumps({k: record.get(k) for k in ("task_id", "status", "video", "error")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--prompt", required=True)
    s.add_argument("--job", required=True)
    s.add_argument("--duration", type=int, default=6)
    s.add_argument("--resolution", default="720p")
    s.add_argument("--aspect_ratio", default="16:9")
    c = sub.add_parser("collect")
    c.add_argument("--job", required=True)
    c.add_argument("--wait", type=int, default=0, help="seconds to keep polling")
    args = ap.parse_args()
    (submit if args.cmd == "submit" else collect)(args)


if __name__ == "__main__":
    main()
