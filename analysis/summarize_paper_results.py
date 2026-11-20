"""Recompute the paper's final-return summaries from the included eval logs.

Each seed contributes the mean of its final five evaluations. The reported
standard deviation is the population standard deviation across five seeds.
Run from any working directory with Python 3:

    python analysis/summarize_paper_results.py
"""

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
OUTPUT = ROOT / "results" / "summary"
TASKS = ("catch", "run", "spin", "swingup", "walk")
METHODS = {
    "dmc/rgb_position": ("BASE", "BASE-C", "BASE-P", "IPR-S", "IPR-H"),
    "dmc/rgb_depth": ("BASE", "IPR-S", "IPR-H"),
    "webots/camera_position": ("BASE", "IPR-S", "IPR-H"),
    "webots/camera_lidar_position": ("BASE", "IPR-S", "IPR-H"),
}


def final_five(log_path, cutoff=None):
    records = []
    with log_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                if cutoff is None or record["step"] <= cutoff:
                    records.append(record)
    if len(records) < 5:
        raise ValueError(f"Fewer than five eligible evaluations: {log_path}")
    return statistics.mean(float(row["mean_episode_reward"]) for row in records[-5:])


def main():
    seed_rows = []
    summary_rows = []
    for setting, methods in METHODS.items():
        tasks = TASKS if setting.startswith("dmc/") else ("env1",)
        for task in tasks:
            cutoff = None
            if setting == "dmc/rgb_depth":
                cutoff = 300000 if task in ("catch", "swingup") else 500000
            for method in methods:
                task_dir = RUNS / setting / method / task
                logs = sorted(task_dir.glob("*/eval_info.log"))
                if len(logs) != 5:
                    raise ValueError(f"Expected five seed logs, found {len(logs)}: {task_dir}")
                values = []
                for log in logs:
                    value = final_five(log, cutoff)
                    values.append(value)
                    seed_rows.append((setting, task, method, log.parent.name, value))
                summary_rows.append(
                    (setting, task, method, statistics.mean(values), statistics.pstdev(values), len(values))
                )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "paper_seed_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("setting", "task", "method", "run", "last_five_mean_return"))
        writer.writerows(seed_rows)
    with (OUTPUT / "paper_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("setting", "task", "method", "mean", "std", "num_seeds"))
        writer.writerows(summary_rows)
    print(f"Wrote {len(seed_rows)} seed results and {len(summary_rows)} summary rows")


if __name__ == "__main__":
    main()
