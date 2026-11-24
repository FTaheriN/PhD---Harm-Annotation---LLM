"""Evaluate prediction files and generate simple HTML tables.

Only standard-library packages are required.
"""

import argparse
import html
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path


REFERENCE_COLUMNS = ("consensus_binary_label", "any_binary_label")


def binary_label(value):
    """Convert recognized labels to 0/1. Unknown values become None."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value) if value in (0, 1) else None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"positive", "true", "1"}:
            return 1
        if value in {"negative", "false", "0"}:
            return 0
    return None


def normalize_id(value):
    """Match integer IDs and their string equivalents across files."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def read_items(path):
    """Read a JSON array/object or JSONL file and collect format problems."""
    path = Path(path)
    problems = []

    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else [data], problems

    items = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                problems.append(f"Line {line_number}: invalid JSON")
    return items, problems


def read_reference_ids(path):
    """Read IDs from objects with message_id, or from scalar ID values."""
    items, problems = read_items(path)
    ids = []
    invalid = 0

    for item in items:
        value = item.get("message_id") if isinstance(item, dict) else item
        key = normalize_id(value)
        if key is None:
            invalid += 1
        else:
            ids.append(key)

    if not ids:
        raise ValueError("The reference-ID file contains no valid message IDs.")

    return set(ids), [
        len(items), len(set(ids)), len(ids) - len(set(ids)),
        invalid, len(problems),
    ], problems


def has_error(record):
    """Recognize the pipeline's classification-error flags."""
    return (
        str(record.get("label", "")).strip().lower() == "error"
        or bool(record.get("llm_error"))
    )


def prediction(record):
    """Failed classifications are excluded from numerical metrics."""
    return None if has_error(record) else binary_label(record.get("label"))


def divide(a, b):
    return a / b if b else None


def calculate_metrics(truth, predicted):
    """Binary metrics: positive/True/1 is the positive class."""
    tp = sum(t == 1 and p == 1 for t, p in zip(truth, predicted))
    tn = sum(t == 0 and p == 0 for t, p in zip(truth, predicted))
    fp = sum(t == 0 and p == 1 for t, p in zip(truth, predicted))
    fn = sum(t == 1 and p == 0 for t, p in zip(truth, predicted))
    n = len(truth)
    agreement = divide(tp + tn, n)

    denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    correlation = divide(tp * tn - fp * fn, denominator)

    expected = (
        ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / n**2
        if n else None
    )
    kappa = divide(agreement - expected, 1 - expected) if n else None

    return [
        n, tp, tn, fp, fn,
        divide(tp, tp + fp),
        divide(tp, tp + fn),
        divide(2 * tp, 2 * tp + fp + fn),
        agreement, correlation, kappa,
    ]


def table(title, headers, rows):
    """Escape all values so they are safe to display as HTML."""
    def cell(value):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.4f}"
        return html.escape(str(value))

    heading = "".join(f"<th>{cell(value)}</th>" for value in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{cell(value)}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return (
        f"<h2>{html.escape(title)}</h2>"
        f"<div class='scroll'><table><thead><tr>{heading}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def evaluate_predictions(
    prediction_paths,
    reference_ids_path=None,
    output_html="evaluation.html",
    print_html=False,
):
    """Evaluate runs, optionally restricted to IDs in another file.

    Labels used as truth come from each prediction file's
    consensus_binary_label and any_binary_label columns.
    The reference-ID file selects records; it does not supply truth labels.
    """
    if not prediction_paths:
        raise ValueError("Supply at least one prediction file.")

    selected_ids = None
    sections = []
    problems_rows = []

    if reference_ids_path is not None:
        selected_ids, summary, problems = read_reference_ids(
            reference_ids_path
        )
        sections.append(table(
            "Reference-ID file",
            ["File", "Parsed items", "Unique valid IDs", "Repeated IDs",
             "Invalid IDs", "Format problems"],
            [[str(reference_ids_path), *summary]],
        ))
        problems_rows.extend(["Reference IDs", p] for p in problems)

    runs = []
    inventory, coverage, missing_examples = [], [], []
    audits, distributions, flagged = [], [], []

    for number, path in enumerate(prediction_paths, 1):
        name = f"Run {number}"
        items, problems = read_items(path)
        records = [item for item in items if isinstance(item, dict)]
        nonobjects = len(items) - len(records)

        problems_rows.extend([name, p] for p in problems)

        # Count duplicates across the complete prediction file first.
        # All occurrences of a duplicated ID are excluded from metrics.
        counts = Counter(
            key for record in records
            if (key := normalize_id(record.get("message_id"))) is not None
        )
        all_ids = set(counts)

        # Restrict evaluation to the optional reference-ID set.
        scoped = [
            record for record in records
            if selected_ids is None
            or normalize_id(record.get("message_id")) in selected_ids
        ]
        index = {
            normalize_id(record.get("message_id")): record
            for record in scoped
            if normalize_id(record.get("message_id")) is not None
            and counts[normalize_id(record.get("message_id"))] == 1
        }
        runs.append((name, index))

        inventory.append([
            name, str(path), len(records), len(problems), nonobjects,
            sum(normalize_id(r.get("message_id")) is None for r in records),
            len(scoped), len(records) - len(scoped), len(index),
        ])

        if selected_ids is not None:
            matched = selected_ids & all_ids
            missing = selected_ids - all_ids
            coverage.append([
                name, len(selected_ids), len(matched), len(missing),
                len(all_ids - selected_ids), len(index),
                sum(prediction(r) is not None for r in index.values()),
            ])
            # Limit displayed IDs to keep the report manageable.
            missing_examples.extend(
                [name, key] for key in sorted(missing)[:50]
            )

        duplicate_ids = {
            normalize_id(r.get("message_id"))
            for r in scoped
            if counts[normalize_id(r.get("message_id"))] > 1
        }
        audits.append([
            name, len(scoped),
            len(duplicate_ids),
            sum(
                normalize_id(r.get("message_id")) in duplicate_ids
                for r in scoped
            ),
            sum(has_error(r) for r in scoped),
            sum(prediction(r) is None for r in scoped),
            sum(str(r.get("confidence", "")).strip().lower() == "low"
                for r in scoped),
            sum(bool(r.get("prefilter_matched")) for r in scoped),
        ])

        for column in ("label", *REFERENCE_COLUMNS):
            values = [binary_label(r.get(column)) for r in scoped]
            distributions.append([
                name, column, values.count(1),
                values.count(0), values.count(None),
            ])

        for position, record in enumerate(scoped, 1):
            key = normalize_id(record.get("message_id"))
            flags = []
            if key is None:
                flags.append("missing/invalid ID")
            elif key in duplicate_ids:
                flags.append("duplicate ID")
            if has_error(record):
                flags.append("classification error")
            if binary_label(record.get("label")) is None:
                flags.append("invalid/nonbinary prediction")
            if str(record.get("confidence", "")).strip().lower() == "low":
                flags.append("low confidence")
            if record.get("prefilter_matched"):
                flags.append("prefilter match (informational)")
            if flags:
                flagged.append([
                    name, position, key, record.get("label"),
                    ", ".join(flags),
                ])

    metric_headers = [
        "Valid pairs", "TP", "TN", "FP", "FN", "Precision", "Recall",
        "F1", "Accuracy / agreement", "Correlation (phi/MCC)", "Kappa",
    ]

    comparison_rows = []
    for name, index in runs:
        for column in REFERENCE_COLUMNS:
            truth, predicted = [], []
            raw_exact = 0

            for record in index.values():
                t = binary_label(record.get(column))
                p = prediction(record)
                if t is None or p is None:
                    continue

                truth.append(t)
                predicted.append(p)
                left, right = record.get("label"), record.get(column)
                raw_exact += type(left) is type(right) and left == right

            comparison_rows.append([
                name, column, len(index) - len(truth),
                *calculate_metrics(truth, predicted),
                raw_exact, divide(raw_exact, len(truth)),
            ])

    pair_rows, consistency_rows = [], []
    for (name_a, a), (name_b, b) in combinations(runs, 2):
        shared = a.keys() & b.keys()
        labels_a, labels_b = [], []

        for key in shared:
            left, right = prediction(a[key]), prediction(b[key])
            if left is not None and right is not None:
                labels_a.append(left)
                labels_b.append(right)

        scores = calculate_metrics(labels_a, labels_b)
        agree = sum(x == y for x, y in zip(labels_a, labels_b))
        pair_rows.append([
            name_a, name_b,
            len(a.keys() - b.keys()), len(b.keys() - a.keys()),
            len(shared), len(shared) - len(labels_a), len(labels_a),
            agree, len(labels_a) - agree,
            scores[8], scores[9], scores[10],
        ])

        # Different references across runs can affect metric comparisons.
        for column in REFERENCE_COLUMNS:
            valid = conflicts = 0
            for key in shared:
                left = binary_label(a[key].get(column))
                right = binary_label(b[key].get(column))
                if left is not None and right is not None:
                    valid += 1
                    conflicts += left != right
            consistency_rows.append([
                name_a, name_b, column, valid, conflicts,
            ])

    sections.append(table(
        "Files and evaluation scope",
        ["Run", "File", "Loaded records", "Invalid JSON lines",
         "Nonobject items", "Missing/invalid IDs in file",
         "Selected records", "Outside selection", "Unique evaluable IDs"],
        inventory,
    ))

    if selected_ids is not None:
        sections.extend([
            table(
                "Reference-ID coverage",
                ["Run", "Requested IDs", "Found IDs", "Missing IDs",
                 "Extra file IDs", "Unique matched IDs",
                 "Matched IDs with usable predictions"],
                coverage,
            ),
            table(
                "Missing reference IDs — up to 50 per run",
                ["Run", "message_id"], missing_examples,
            ),
        ])

    sections.extend([
        table(
            "Checks within the selected records",
            ["Run", "Records", "Duplicate IDs", "Duplicate records",
             "Classification errors", "Unusable predictions",
             "Low confidence", "Prefilter matches"],
            audits,
        ),
        table(
            "Label counts within the selected records",
            ["Run", "Column", "Positive / True / 1",
             "Negative / False / 0", "Missing / other"],
            distributions,
        ),
        table(
            "Predictions versus reference columns",
            ["Run", "Reference", "Excluded unique records",
             *metric_headers, "Raw exact matches", "Raw exact match rate"],
            comparison_rows,
        ),
        table(
            "Run-to-run agreement",
            ["Run A", "Run B", "Only A", "Only B", "Shared unique IDs",
             "Unusable shared pairs", "Valid pairs", "Agree", "Disagree",
             "Agreement rate", "Correlation (phi/MCC)", "Kappa"],
            pair_rows,
        ),
        table(
            "Reference-column consistency between runs",
            ["Run A", "Run B", "Column", "Valid shared references",
             "Conflicting references"],
            consistency_rows,
        ),
        table(
            "Flagged selected records — first 100",
            ["Run", "Selected record position", "message_id", "Label", "Flags"],
            flagged[:100],
        ),
        table(
            "File-format problems — first 100",
            ["File / run", "Problem"], problems_rows[:100],
        ),
    ])

    report = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LLM prediction evaluation</title>
<style>
body {font:14px Arial,sans-serif; margin:24px; color:#222;}
p {max-width:1050px; line-height:1.5;}
h2 {margin-top:28px;}
.scroll {overflow-x:auto; margin-bottom:20px;}
table {border-collapse:collapse;}
th,td {border:1px solid #ddd; padding:8px; text-align:left;}
th {background:#f3f4f6;}
tr:nth-child(even) {background:#fafafa;}
</style>
</head>
<body>
<h1>LLM prediction evaluation</h1>
<p>The optional reference-ID file selects message IDs only. Truth labels
come from consensus_binary_label and any_binary_label in each prediction
file. Without an ID file, all loaded records are selected.</p>
<p>Positive/True/1 and negative/False/0 are normalized binary classes.
Metrics exclude failed predictions and invalid or missing reference labels.
All occurrences of duplicated message IDs are excluded from comparisons.
Label-count and audit tables include duplicates within the selection.</p>
<p>Raw exact matches require identical stored types and values.
Correlation is binary Pearson correlation (phi/MCC). N/A means undefined.
Prefilter matches are informational. Flags can overlap.</p>
<p>Missing predictions are reported in coverage, but excluded from metric
denominators. Each run's scores use its own valid reference pairs;
run-to-run agreement uses shared valid IDs. Therefore, metric cohorts may
differ. Records absent from prediction files cannot be audited here.</p>
""" + "\n".join(sections) + "\n</body></html>"

    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    if print_html:
        print(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_files", nargs="+")
    parser.add_argument(
        "--reference-ids",
        help="Optional JSON/JSONL file selecting message IDs to evaluate.",
    )
    parser.add_argument("--output", default="evaluation.html")
    parser.add_argument("--print-html", action="store_true")
    args = parser.parse_args()

    evaluate_predictions(
        args.prediction_files,
        reference_ids_path=args.reference_ids,
        output_html=args.output,
        print_html=args.print_html,
    )
    if not args.print_html:
        print(f"Report saved to: {Path(args.output).resolve()}")