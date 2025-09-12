#!/usr/bin/env python3
"""
plot_compare.py

Usage:
    python plot_compare.py --csvs projectA_eval_summary.csv projectB_eval_summary.csv --metric score --out plot.html
"""

import argparse
import os
import pandas as pd
import plotly.graph_objects as go


def load_csvs(csv_files):
    dfs = []
    for p in csv_files:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found")
        d = pd.read_csv(p)
        required = {"project", "group", "metric", "mean", "std"}
        if not required.issubset(set(d.columns)):
            raise ValueError(f"{p} missing required columns {required - set(d.columns)}")
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def pivot_for_metric(df_all, metric):
    df = df_all[df_all["metric"] == metric].copy()
    if df.empty:
        raise ValueError(f"No rows found for metric '{metric}'")
    mean_pivot = df.pivot(index="project", columns="group", values="mean")
    std_pivot = df.pivot(index="project", columns="group", values="std")
    projects = list(mean_pivot.index)
    methods = list(mean_pivot.columns)
    return mean_pivot, std_pivot, projects, methods


def make_grouped_bar(mean_pivot, std_pivot, projects, methods, out_html):
    fig = go.Figure()
    for method in methods:
        y = mean_pivot[method].tolist()
        err = std_pivot[method].tolist()
        fig.add_trace(
            go.Bar(
                name=str(method),
                x=projects,
                y=y,
                error_y=dict(type="data", array=err, visible=True),
            )
        )
    fig.update_layout(
        barmode="group",
        title="Comparison across projects",
        xaxis_title="Project",
        yaxis_title="Mean value",
        legend_title="Method (group)",
    )
    fig.write_html(out_html)
    print(f"Wrote interactive plot to: {out_html}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot grouped barplot across projects from wandb eval CSVs."
    )
    parser.add_argument("--csvs", nargs="+", required=True, help="Paths to CSV files")
    parser.add_argument(
        "--metric", default=None, help="Metric name (default: auto-select first)"
    )
    parser.add_argument("--out", default="projects_compare.html", help="Output HTML")
    args = parser.parse_args()

    df_all = load_csvs(args.csvs)
    if args.metric is None:
        metric = df_all["metric"].mode().iloc[0]
        print(f"No metric provided. Auto-selected metric: {metric}")
    else:
        metric = args.metric

    mean_pivot, std_pivot, projects, methods = pivot_for_metric(df_all, metric)
    make_grouped_bar(mean_pivot, std_pivot, projects, methods, args.out)
