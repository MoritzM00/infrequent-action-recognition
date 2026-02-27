import wandb

# ==========================================
# CONFIGURATION
# ==========================================
ENTITY = "moritzm00"
PROJECT = "fall-detection-zeroshot-v4"
SPLIT = "cs"

METRIC_SUFFIXES = [
    "balanced_accuracy",
    "accuracy",
    "macro_f1",
    "fall_sensitivity",
    "fall_specificity",
    "fall_f1",
    "fallen_sensitivity",
    "fallen_specificity",
    "fallen_f1",
]

# Each dataset entry: wb_name (W&B metric prefix), display_name (LaTeX),
# specialized_name (row label), specialized_metrics (9 floats as %, or None)
DATASETS = [
    {
        "wb_name": "le2i",
        "display_name": "Le2i",
        "specialized_name": "VMAE-K400",
        "specialized_metrics": [83.2, 89.7, 82.4, 100.0, 99.4, 97.8, 100.0, 100.0, 100.0],
    },
    {
        "wb_name": "up_fall",
        "display_name": "UP-Fall",
        "specialized_name": "VMAE-K400",
        "specialized_metrics": [92.6, 92.0, 82.4, 96.5, 99.3, 96.9, 79.8, 98.0, 85.4],
    },
    {
        "wb_name": "cmdfall",
        "display_name": "CMDFall",
        "specialized_name": "VMAE-K400",
        "specialized_metrics": [83.0, 85.4, 83.3, 92.6, 98.9, 92.6, 90.9, 98.9, 91.0],
    },
]

# Nested dict: model_display_name -> {dataset_wb_name -> run_id}
MLLM_RUN_IDS: dict[str, dict[str, str]] = {
    "InternVL3.5-8B": {
        "le2i": "rxsihd2u",
        "up_fall": "rocdaohb",
        "cmdfall": "5ia833zb",
    },
    "Qwen3-VL-8B": {
        "le2i": "9uhyvciq",
        "up_fall": "8oj3dfnt",
        "cmdfall": "8fa8ojpb",
    },
}


# ==========================================
# FETCH
# ==========================================
def fetch_run_metrics(api, run_id: str, dataset_wb_name: str) -> list[float | None]:
    """Fetch the 9 metrics for a given run and dataset prefix."""
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        summary = run.summary
        values: list[float | None] = []
        for suffix in METRIC_SUFFIXES:
            key = f"{dataset_wb_name}_{SPLIT}_{suffix}"
            val = summary.get(key)
            values.append(val * 100 if val is not None else None)
        return values
    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        return [None] * len(METRIC_SUFFIXES)


# ==========================================
# FORMATTING HELPERS
# ==========================================
def find_best_per_column(rows: list[list[float | None]]) -> list[float | None]:
    """Return the maximum value per column across the given rows."""
    num_cols = len(METRIC_SUFFIXES)
    best: list[float | None] = []
    for col in range(num_cols):
        vals = [row[col] for row in rows if row[col] is not None]
        best.append(max(vals) if vals else None)
    return best


def format_value(val: float | None, best_val: float | None) -> str:
    """Format a metric value, bolding if it equals the best in its group."""
    if val is None:
        return "--"
    val_rounded = round(val, 1)
    formatted = f"{val_rounded:.1f}"
    if best_val is not None and round(best_val, 1) == val_rounded:
        return f"\\textbf{{{formatted}}}"
    return formatted


# ==========================================
# LATEX GENERATION
# ==========================================
def generate_latex() -> None:
    api = wandb.Api()

    # Collect all row data grouped by dataset
    dataset_groups: list[dict] = []
    for ds in DATASETS:
        rows: list[dict] = []

        # Specialized model row
        rows.append(
            {
                "name": ds["specialized_name"],
                "metrics": ds["specialized_metrics"],
            }
        )

        # MLLM rows
        for model_name, run_ids in MLLM_RUN_IDS.items():
            run_id = run_ids[ds["wb_name"]]
            if run_id == "TODO":
                metrics: list[float | None] = [None] * len(METRIC_SUFFIXES)
            else:
                metrics = fetch_run_metrics(api, run_id, ds["wb_name"])
            rows.append({"name": model_name, "metrics": metrics})

        dataset_groups.append(
            {
                "display_name": ds["display_name"],
                "rows": rows,
            }
        )

    # Build LaTeX body
    body_lines: list[str] = []
    for group_idx, group in enumerate(dataset_groups):
        rows = group["rows"]
        metric_rows = [r["metrics"] for r in rows]
        best = find_best_per_column(metric_rows)

        display_name = group["display_name"]

        for row_idx, row in enumerate(rows):
            # Dataset name cell: multirow on the first row of the group
            if row_idx == 0:
                dataset_cell = f"\\multirow{{3}}{{*}}{{{display_name}}}"
            else:
                dataset_cell = ""

            formatted = [format_value(v, best[i]) for i, v in enumerate(row["metrics"])]
            metrics_str = " & ".join(formatted)
            body_lines.append(f"        {dataset_cell} & {row['name']} & {metrics_str} \\\\")

        # Separator between groups (but not after the last one)
        if group_idx < len(dataset_groups) - 1:
            body_lines.append("        \\midrule")

    body = "\n".join(body_lines)

    table = f"""\\begingroup
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{table}}[htp]
\\caption{{\\textbf{{Zero-shot fall detection on additional datasets.}}
Results for InternVL3.5-8B and Qwen3-VL-8B compared to a dataset-specific specialized
baseline on the Le2i, UP-Fall, and CMDFall benchmarks (cross-subject split).
Best results per dataset are highlighted in \\textbf{{bold}}.}}
\\label{{tab:cross_dataset_results}}

\\resizebox{{\\columnwidth}}{{!}}{{
\\begin{{tabular}}{{@{{}}l l rrr rrr rrr@{{}}}}
\\toprule
% Top Header Row
\\multirow{{2}}{{*}}{{\\textbf{{Dataset}}}} &
\\multirow{{2}}{{*}}{{\\textbf{{Model}}}} &
\\multicolumn{{3}}{{c}}{{16-class}} &
\\multicolumn{{3}}{{c}}{{Fall $\\Delta$}} &
\\multicolumn{{3}}{{c}}{{Fallen $\\Delta$}} \\\\
\\cmidrule(lr){{3-5}} \\cmidrule(lr){{6-8}} \\cmidrule(lr){{9-11}}

% Sub Header Row
 & & \\multicolumn{{1}}{{c}}{{BAcc}} & \\multicolumn{{1}}{{c}}{{Acc}} & \\multicolumn{{1}}{{c}}{{F1}}
 & \\multicolumn{{1}}{{c}}{{Se}}   & \\multicolumn{{1}}{{c}}{{Sp}}  & \\multicolumn{{1}}{{c}}{{F1}}
 & \\multicolumn{{1}}{{c}}{{Se}}   & \\multicolumn{{1}}{{c}}{{Sp}}  & \\multicolumn{{1}}{{c}}{{F1}} \\\\
\\midrule

{body}

\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
\\endgroup
"""

    print(table)


def main() -> None:
    generate_latex()


if __name__ == "__main__":
    main()
