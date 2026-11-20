import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_STYLE_MAP = {
    "BASE": {"color": "#0072B2", "linestyle": "-"},
    "IPR-S": {"color": "#E69F00", "linestyle": "-"},
    "IPR-H": {"color": "#009E73", "linestyle": "-"},
}

METHOD_ORDER = {
    "BASE": 0,
    "IPR-S": 1,
    "IPR-H": 2,
}

FALLBACK_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def method_display_name(method_name):
    """Return the canonical method label used in plots and reports."""
    display_name = method_name.partition("(")[0].strip()
    aliases = {
        "BASE": "BASE",
        "IPR": "IPR-S",
        "IPR-S": "IPR-S",
        "IPR-H": "IPR-H",
    }
    return aliases.get(display_name.upper(), display_name or method_name)


def read_eval_log(path, metric):
    steps = []
    rewards = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc

            if "step" not in record:
                raise KeyError(f"{path} line {line_no} has no 'step' field")
            if metric not in record:
                raise KeyError(f"{path} line {line_no} has no '{metric}' field")

            steps.append(float(record["step"]))
            rewards.append(float(record[metric]))

    if not steps:
        raise ValueError(f"No data found in {path}")

    order = np.argsort(steps)
    return np.asarray(steps, dtype=np.float64)[order], np.asarray(rewards, dtype=np.float64)[order]


def discover_runs(method_dir):
    runs = {}
    method_dir = Path(method_dir)

    for log_path in method_dir.glob("*/*/eval_info.log"):
        task = log_path.parent.parent.name
        seed = log_path.parent.name
        runs.setdefault(task, {})[seed] = log_path

    return runs


def discover_methods(inputs):
    methods = {}
    ignored_names = {"__pycache__", "plots", "baseline_plots", "plots_default_test", "plots_sorted_test"}
    for path in [Path(p) for p in inputs]:
        if not path.exists():
            raise FileNotFoundError(path)

        direct_runs = discover_runs(path)
        if direct_runs:
            methods[path.name] = direct_runs
            continue

        for child in sorted(p for p in path.iterdir() if p.is_dir()):
            if child.name in ignored_names or child.name.startswith("plots"):
                continue
            child_runs = discover_runs(child)
            if child_runs:
                methods[child.name] = child_runs
            else:
                print(
                    f"Skipped {child.name}: no task/seed/eval_info.log files found."
                )

    if not methods:
        raise ValueError(
            "No eval logs found. Expected method/task/seed/eval_info.log "
            "or task/seed/eval_info.log under the input path."
        )

    return methods


def aggregate_seed_curves(seed_logs, metric):
    curves = []
    curve_maps = []
    for seed, log_path in sorted(seed_logs.items()):
        steps, values = read_eval_log(log_path, metric)
        step_to_value = {}
        for step, value in zip(steps, values):
            step_to_value[float(step)] = float(value)
        curves.append((seed, steps, values))
        curve_maps.append((seed, step_to_value))

    same_step_grid = all(
        np.array_equal(curves[0][1], steps)
        for _, steps, _ in curves[1:]
    )
    if same_step_grid:
        x = curves[0][1]
        y = np.stack([values for _, _, values in curves], axis=0)
    else:
        deduped_curves = []
        for seed, step_to_value in curve_maps:
            unique_steps = np.asarray(sorted(step_to_value.keys()), dtype=np.float64)
            unique_values = np.asarray(
                [step_to_value[step] for step in unique_steps],
                dtype=np.float64,
            )
            deduped_curves.append((seed, unique_steps, unique_values))

        min_step = max(float(steps.min()) for _, steps, _ in deduped_curves)
        max_step = min(float(steps.max()) for _, steps, _ in deduped_curves)
        if min_step >= max_step:
            raise ValueError("Seed curves do not share or overlap any step range")
        point_count = min(len(steps) for _, steps, _ in deduped_curves)
        x = np.linspace(min_step, max_step, point_count)
        y = np.stack(
            [
                np.interp(x, steps, values)
                for _, steps, values in deduped_curves
            ],
            axis=0,
        )

    mean = y.mean(axis=0)
    std = y.std(axis=0, ddof=0)
    return x, mean, std, len(curves)


def smooth_curve(y, window):
    if window <= 1:
        return y
    window = int(window)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(y, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def summarize_last_rewards(
    task,
    method_name,
    seed_logs,
    metric,
    last_n=5,
    eval_step=None,
):
    seed_rows = []
    reward_rows = []

    for seed, log_path in sorted(seed_logs.items()):
        steps, rewards = read_eval_log(log_path, metric)
        if eval_step is not None:
            selected = steps <= float(eval_step)
            steps = steps[selected]
            rewards = rewards[selected]
        if rewards.size < last_n:
            step_message = (
                f" at or before eval step {eval_step}"
                if eval_step is not None
                else ""
            )
            raise ValueError(
                f"{log_path} contains only {rewards.size} evaluations"
                f"{step_message}; "
                f"at least {last_n} are required for the final-reward summary."
            )

        final_steps = steps[-last_n:]
        final_rewards = rewards[-last_n:]
        reward_rows.append(final_rewards)
        seed_rows.append(
            {
                "task": task,
                "method": method_name,
                "seed": seed,
                "steps": final_steps,
                "rewards": final_rewards,
                "last_n_mean": float(final_rewards.mean()),
            }
        )

    reward_matrix = np.stack(reward_rows, axis=0)
    seed_last_n_means = np.asarray(
        [row["last_n_mean"] for row in seed_rows],
        dtype=np.float64,
    )
    summary_rows = []
    for index in range(last_n):
        distance_from_end = last_n - index
        summary_rows.append(
            {
                "task": task,
                "method": method_name,
                "evaluation": f"last_{distance_from_end}",
                "mean": float(reward_matrix[:, index].mean()),
                "std": float(reward_matrix[:, index].std(ddof=0)),
                "num_seeds": reward_matrix.shape[0],
            }
        )
    summary_rows.append(
        {
            "task": task,
            "method": method_name,
            "evaluation": f"last_{last_n}_average",
            "mean": float(seed_last_n_means.mean()),
            "std": float(seed_last_n_means.std(ddof=0)),
            "num_seeds": reward_matrix.shape[0],
        }
    )
    return seed_rows, summary_rows


def write_last_reward_reports(
    output_dir,
    seed_rows,
    summary_rows,
    last_n,
    eval_step=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    step_suffix = f"_to_step_{eval_step}" if eval_step is not None else ""
    detail_path = output_dir / (
        f"last_{last_n}_rewards_by_seed{step_suffix}.csv"
    )
    summary_path = output_dir / (
        f"last_{last_n}_reward_summary{step_suffix}.csv"
    )

    detail_fields = ["task", "method", "seed"]
    for distance_from_end in range(last_n, 0, -1):
        detail_fields.extend(
            [f"step_last_{distance_from_end}", f"reward_last_{distance_from_end}"]
        )
    detail_fields.append(f"mean_last_{last_n}")

    with detail_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=detail_fields)
        writer.writeheader()
        for row in seed_rows:
            output = {
                "task": row["task"],
                "method": row["method"],
                "seed": row["seed"],
                f"mean_last_{last_n}": row["last_n_mean"],
            }
            for index, distance_from_end in enumerate(range(last_n, 0, -1)):
                output[f"step_last_{distance_from_end}"] = row["steps"][index]
                output[f"reward_last_{distance_from_end}"] = row["rewards"][index]
            writer.writerow(output)

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        fields = ["task", "method", "evaluation", "mean", "std", "num_seeds"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    return detail_path, summary_path


def plot_task(task, task_methods, output_dir, metric, smooth, show_std, title_prefix):
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(8, 5.5))

    plotted_methods = []
    for method_name, seed_logs in task_methods.items():
        x, mean, std, seed_count = aggregate_seed_curves(seed_logs, metric)
        mean_plot = smooth_curve(mean, smooth)
        std_plot = smooth_curve(std, smooth)
        final_score = float(mean_plot[-1])
        plotted_methods.append(
            (final_score, method_name, x, mean_plot, std_plot, seed_count)
        )

    plotted_methods.sort(
        key=lambda item: (
            METHOD_ORDER.get(method_display_name(item[1]), len(METHOD_ORDER)),
            method_display_name(item[1]),
        )
    )

    visible_lower_bounds = []
    visible_upper_bounds = []

    for rank, (_, method_name, x, mean_plot, std_plot, seed_count) in enumerate(
        plotted_methods
    ):
        display_name = method_display_name(method_name)
        style = METHOD_STYLE_MAP.get(display_name, {})
        color = style.get(
            "color",
            FALLBACK_COLORS[rank % len(FALLBACK_COLORS)],
        )
        linestyle = style.get("linestyle", "-")
        ax.plot(
            x,
            mean_plot,
            linewidth=2.0,
            label=display_name,
            color=color,
            linestyle=linestyle,
        )
        if show_std and seed_count > 1:
            lower_curve = mean_plot - std_plot
            upper_curve = mean_plot + std_plot
            ax.fill_between(
                x,
                lower_curve,
                upper_curve,
                alpha=0.18,
                linewidth=0,
                color=color,
            )
            visible_lower_bounds.append(float(np.min(lower_curve)))
            visible_upper_bounds.append(float(np.max(upper_curve)))
        else:
            visible_lower_bounds.append(float(np.min(mean_plot)))
            visible_upper_bounds.append(float(np.max(mean_plot)))

    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")
    # Scale each task to its observed curves instead of using a fixed
    # Webots reward range. Include the uncertainty shading and add a small
    # margin so low-return experiments remain readable.
    data_min = min(visible_lower_bounds)
    data_max = max(visible_upper_bounds)
    data_span = max(data_max - data_min, 1.0)
    y_padding = 0.08 * data_span
    y_min = min(0.0, data_min - y_padding)
    y_max = data_max + y_padding
    if y_max <= y_min:
        y_max = y_min + 1.0
    ax.set_ylim(y_min, y_max)
    title = task if not title_prefix else f"{title_prefix} {task}"
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{task}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot eval curves from method/task/seed/eval_info.log files. "
            "Passing a method directory such as baseline plots that method; "
            "passing a results directory compares all contained method folders. "
            "With no input, the current working directory is used."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[Path.cwd()],
        help="Method directory or results root directory. Defaults to current directory.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        help="Directory for generated PNG files. Defaults to <first input>/plots.",
    )
    parser.add_argument(
        "--metric",
        default="mean_episode_reward",
        help="Metric field to plot from eval_info.log.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving-average smoothing window. Use 1 to disable.",
    )
    parser.add_argument(
        "--no_std",
        action="store_true",
        help="Do not draw standard-deviation shading across seeds.",
    )
    parser.add_argument(
        "--title_prefix",
        default="",
        help="Optional title prefix.",
    )
    parser.add_argument(
        "--last_n",
        type=int,
        default=5,
        help="Number of final evaluations to report per seed. Defaults to 5.",
    )
    parser.add_argument(
        "--eval_step",
        type=int,
        default=None,
        help=(
            "Only use evaluations at or before this step for the final last-N "
            "summary, for example 300000 or 500000."
        ),
    )
    args = parser.parse_args()
    if args.last_n < 1:
        parser.error("--last_n must be at least 1")
    if args.eval_step is not None and args.eval_step < 0:
        parser.error("--eval_step must be non-negative")

    input_paths = [Path(p).resolve() for p in args.inputs]
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_paths[0] / "plots"
    )

    methods = discover_methods(input_paths)
    tasks = sorted({task for runs in methods.values() for task in runs.keys()})
    generated = []
    all_seed_rows = []
    all_summary_rows = []

    for task in tasks:
        task_methods = {
            method_name: runs[task]
            for method_name, runs in methods.items()
            if task in runs
        }
        generated.append(
            plot_task(
                task=task,
                task_methods=task_methods,
                output_dir=output_dir,
                metric=args.metric,
                smooth=args.smooth,
                show_std=not args.no_std,
                title_prefix=args.title_prefix,
            )
        )

        for method_name, seed_logs in sorted(task_methods.items()):
            seed_rows, summary_rows = summarize_last_rewards(
                task=task,
                method_name=method_display_name(method_name),
                seed_logs=seed_logs,
                metric=args.metric,
                last_n=args.last_n,
                eval_step=args.eval_step,
            )
            all_seed_rows.extend(seed_rows)
            all_summary_rows.extend(summary_rows)

    detail_path, summary_path = write_last_reward_reports(
        output_dir,
        all_seed_rows,
        all_summary_rows,
        args.last_n,
        eval_step=args.eval_step,
    )

    for path in generated:
        print(f"Saved {path}")
    print(f"Saved {detail_path}")
    print(f"Saved {summary_path}")

    average_label = f"last_{args.last_n}_average"
    if args.eval_step is None:
        summary_scope = "from the complete logs"
    else:
        summary_scope = f"at or before eval step {args.eval_step}"
    print(
        f"\nFinal {args.last_n}-evaluation reward across seeds "
        f"{summary_scope}:"
    )
    for row in sorted(
            (row for row in all_summary_rows if row["evaluation"] == average_label),
            key=lambda row: (row["task"], -row["mean"]),
    ):
        print(
            f"  {row['task']} | {row['method']}: "
            f"mean={row['mean']:.6f}, std={row['std']:.6f}, "
            f"seeds={row['num_seeds']}"
        )


if __name__ == "__main__":
    main()
