#!/usr/bin/env python3
"""
wandb_collect.py

Usage:
    python wandb_collect.py --project my_project --entity my_entity --outdir ./out --metric_substr score

What it does:
1. Fetches all runs in a W&B project.
2. Groups them by run.group.
3. Writes a shell script of "recreate" commands for each run.
4. Finds eval groups (groups with "eval" in their name).
5. For each eval group:
   - Collects all metrics whose name contains `metric_substr`.
   - Multi-task runs: average across those metrics per run, then average across runs.
   - Per-task runs: only one metric → average across runs directly.
6. Saves summary CSV with columns:
   [project, group, metric, mean, std, n_runs].
"""

import argparse
import os
import math
from collections import defaultdict
from typing import Any
import wandb
import pandas as pd
from anybody.utils.path_utils import get_wandb_csv_dir


def safe_scalar(v: Any):
    return isinstance(v, (int, float, str, bool)) and not (
        isinstance(v, float) and math.isnan(v)
    )


def flatten_config_for_cli(cfg: dict):
    """Return CLI args for scalar config entries."""
    pairs = []
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        if safe_scalar(v):
            if isinstance(v, bool):
                pairs.append(f"--{k}" if v else f"--no-{k}")
            else:
                val = str(v)
                if " " in val:
                    val = f"'{val}'"
                pairs.append(f"--{k} {val}")
    return pairs


def construct_command_from_run(run):
    cfg = dict(run.config or {})
    program = None
    for candidate in ["program", "script", "entry_point", "cmd", "command"]:
        if candidate in cfg and isinstance(cfg[candidate], str):
            program = cfg[candidate]
            break
    if not program:
        program = "train.py"

    cli_parts = [program]
    cli_parts += flatten_config_for_cli(cfg)

    proj = f"{run.entity}/{run.project}" if run.entity else run.project
    if proj:
        cli_parts.append(f"--wandb_project {proj}")
    cli_parts.append(f"--wandb_run_id {run.id}")

    return "python " + " ".join(cli_parts)


def summarize_eval_groups(groups, project):
    """
    Summarize eval groups: for each group, collect metrics containing metric_substr.
    Multi-task agents: average across multiple metrics.
    Per-task agents: use single metric.
    """
    summary_rows = []

    for gname, runs_list in sorted(groups.items()):
        print(f"Processing eval group: {gname} ({len(runs_list)} runs)")
        run_values = []

        for run in runs_list:
            try:
                hist: pd.DataFrame = run.history(samples=10000000)
            except Exception as e:
                print(f"  Warning: history fetch failed for run {run.id}: {e}")
                continue

            if hist is None or hist.empty:
                continue

            # find relevant metric columns
            metric_cols = [
                c for c in hist.columns if is_metric_column(project, c) and not c.startswith("_")
            ]
            if not metric_cols:
                continue

            per_run_vals = []
            for col in metric_cols:
                vals = pd.to_numeric(hist[col], errors="coerce").dropna().tolist()
                per_run_vals.extend(vals)

            if not per_run_vals:
                continue

            # average across metrics for this run
            run_mean = float(pd.Series(per_run_vals).mean())
            run_values.append(run_mean)

        if not run_values:
            continue

        group_mean = float(pd.Series(run_values).mean())
        group_std = float(pd.Series(run_values).std(ddof=0))

        summary_rows.append(
            {
                "project": project,
                "group": gname,
                # "metric": metric_substr,
                "metric": ""
                "mean": group_mean,
                "std": group_std,
                "n_runs": len(run_values),
            }
        )

    return pd.DataFrame(summary_rows)

def is_metric_column(project: str, col: str):
    if "reach" in project.lower():
        return ("robo_0_ee" in col)
    if "push" in project.lower():
        return ("Success rate" in col)
    if "task" in project.lower():
        return ("Success rate" in col) or ("robo_0_ee" in col)


def collect_project(entity: str, project: str, outdir: str):
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    print(f"Querying wandb runs for: {path} ...")

    runs = list(api.runs(path))
    print(f"Total runs fetched: {len(runs)}")

    groups = defaultdict(list)
    for run in runs:
        group_name = run.group if getattr(run, "group", None) else "__ungrouped__"
        groups[group_name].append(run)

    # 1) Create shell script of commands
    os.makedirs(outdir, exist_ok=True)
    cmdfile = os.path.join(outdir, f"commands_{project}.sh")
    with open(cmdfile, "w") as fh:
        fh.write("#!/usr/bin/env bash\n")
        fh.write(f"# Commands generated from W&B project: {path}\n\n")
        for gname, runs_list in sorted(groups.items()):
            fh.write(f"### GROUP: {gname}  (runs: {len(runs_list)})\n")
            for run in runs_list:
                try:
                    cmd = construct_command_from_run(run)
                except Exception as e:
                    cmd = f"# could not construct command for run {run.id}: {e}"
                fh.write(cmd + "\n")
            fh.write("\n")
    os.chmod(cmdfile, 0o755)
    print(f"Wrote command script to: {cmdfile}")

    # 2) Summarize eval groups
    eval_groups = {k: v for k, v in groups.items() if "eval" in k.lower()}
    print(f"Found {len(eval_groups)} eval groups (name contains 'eval').")

    df_summary = summarize_eval_groups(eval_groups, project, metric_substr=metric_substr)
    out_csv = os.path.join(outdir, f"{project}.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"Wrote eval summary CSV to: {out_csv}")

    return cmdfile, out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect wandb runs, create commands, and summarize eval groups."
    )
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument(
        "--entity",
        default="meenalp_project",
        help="W&B entity (user/org).",
    )
    parser.add_argument(
        "--outdir", default="./wandb_out", help="Output directory for results"
    )
    parser.add_argument(
        "--metric_substr",
        default="score",
        help="Substring to match relevant metrics (default: 'score')",
    )
    args = parser.parse_args()
    args.outdir = get_wandb_csv_dir()

    entity = args.entity if args.entity else None
    collect_project(entity, args.project, args.outdir, args.metric_substr)
