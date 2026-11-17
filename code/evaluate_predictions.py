"""Evaluate LLM predictions and write a simple HTML report.

Example:
    python evaluate_predictions.py run1.jsonl run2.jsonl \
        --output evaluation.html

Accepts JSONL, a JSON array, or a single JSON object.
No additional packages are required.
"""

import argparse
import html
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path


def binary_label(value):
    """Convert supported labels to 0/1; return None for unknown values."""
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


def record_id(record):
    """Normalize integer/string IDs for matching across files."""
    value = record.get("message_id")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def read_records(path):
    """Return valid record objects and any file-format problems."""
    path = Path(path)
    records, problems = [], []

    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        items = data if isinstance(data, list) else [data]
        for position, item in enumerate(items, 1):
            if isinstance(item, dict):
                records.append(item)
            else:
                problems.append(f"Item {position}: not a JSON object")
    else:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    problems.append(f"Line {line_number}: invalid JSON")
                    continue

                if isinstance(item, dict):
                    records.append(item)
                else:
                    problems.append(
                        f"Line {line_number}: not a JSON object"
                    )

    return records, problems


def has_error(record):
    """Recognize the error fields produced by classify_llm.py."""
    label_error = (
        str(record.get("label", "")).strip().lower() == "error"
    )
    return label_error or bool(record.get("llm_error"))


def prediction(record):
    """Exclude failed classifications even if they contain a binary label."""
    return None if has_error(record) else binary_label(record.get("label"))


def unique_records(records):
    """Index by message_id, excluding every occurrence of duplicate IDs."""
    counts = Counter(
        key for record in records
        if (key := record_id(record)) is not None
    )
    index = {
        record_id(record): record
        for record in records
        if record_id(record) is not None
        and counts[record_id(record)] == 1
    }
    return index, counts


def divide(numerator, denominator):
    """Undefined metrics are shown as N/A, rather than silently set to zero."""
    return numerator / denominator if denominator else None


def metrics(truth, predicted):
    """Calculate binary metrics, treating positive/True/1 as positive."""
    tp = sum(t == 1 and p == 1 for t, p in zip(truth, predicted))
    tn = sum(t == 0 and p == 0 for t, p in zip(truth, predicted))
    fp = sum(t == 0 and p == 1 for t, p in zip(truth, predicted))
    fn = sum(t == 1 and p == 0 for t, p in zip(truth, predicted))
    n = len(truth)

    # Pearson correlation of binary labels is also called phi/MCC.
    denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    correlation = divide(tp * tn - fp * fn, denominator)

    agreement = divide(tp + tn, n)
    expected = (
        ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / n**2
        if n else None
    )
    kappa = (
        divide(agreement - expected, 1 - expected)
        if n else None
    )

    return [
        n, tp, tn, fp, fn,
        divide(tp, tp + fp),                  # Precision
        divide(tp, tp + fn),                  # Recall
        divide(2 * tp, 2 * tp + fp + fn),      # F1
        agreement, correlation, kappa,
    ]


def table(title, headers, rows):
    """Create an escaped HTML table."""
    def cell(value):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.4f}"
        return html.escape(str(value))

    heading = "".join(f"<th>{cell(x)}</th>" for x in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell(x)}</td>" for x in row) + "</tr>"
        for row in rows
    )
    return (
        f"<h2>{html.escape(title)}</h2>"
        f"<table><thead><tr>{heading}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def evaluate_predictions(
    prediction_paths,
    output_html="evaluation.html",
    print_html=True,
):
    """Evaluate a list of prediction files.

    Returns HTML, saves it to output_html, and optionally prints it.
    Comparisons use unique message IDs and valid binary labels only.
    """
    runs = []
    inventory, audits, distributions = [], [], []
    flagged, format_problems = [], []

    for number, path in enumerate(prediction_paths, 1):
        name = f"Run {number}"
        records, problems = read_records(path)
        index, counts = unique_records(records)
        runs.append((name, index))

        inventory.append([name, str(path), len(records), len(index)])
        format_problems.extend([name, problem] for problem in problems)

        duplicate_ids = {key for key, count in counts.items() if count > 1}

        audits.append([
            name,
            len(problems),
            sum(record_id(r) is None for r in records),
            len(duplicate_ids),
            sum(counts[key] for key in duplicate_ids),
            sum(has_error(r) for r in records),
            sum(prediction(r) is None for r in records),
            sum(str(r.get("confidence", "")).lower() == "low"
                for r in records),
            sum(bool(r.get("prefilter_matched")) for r in records),
        ])

        for column in ("label", "consensus_binary_label", "ground_truth"):
            values = [binary_label(r.get(column)) for r in records]
            unknown = Counter(
                json.dumps(r.get(column), ensure_ascii=False, sort_keys=True)
                for r in records
                if binary_label(r.get(column)) is None
            )
            distributions.append([
                name, column,
                values.count(1), values.count(0), values.count(None),
                "; ".join(f"{key}: {count}"
                          for key, count in unknown.most_common(10)) or "—",
            ])

        # Include IDs and flags only, never the original post content.
        for position, record in enumerate(records, 1):
            flags = []
            key = record_id(record)
            if key is None:
                flags.append("missing/invalid message_id")
            elif key in duplicate_ids:
                flags.append("duplicate message_id")
            if has_error(record):
                flags.append("classification error")
            if binary_label(record.get("label")) is None:
                flags.append("invalid/nonbinary prediction")
            if str(record.get("confidence", "")).lower() == "low":
                flags.append("low confidence")
            if record.get("prefilter_matched"):
                flags.append("prefilter matched (informational)")

            if flags:
                flagged.append([
                    name, position, key or "—",
                    record.get("label"), ", ".join(flags),
                ])

    metric_headers = [
        "Valid pairs", "TP", "TN", "FP", "FN",
        "Precision", "Recall", "F1", "Accuracy / agreement",
        "Correlation (phi/MCC)", "Cohen's kappa",
    ]

    reference_rows = []
    for name, index in runs:
        for reference in ("consensus_binary_label", "ground_truth"):
            truth, predicted = [], []
            raw_exact = 0

            for record in index.values():
                actual = binary_label(record.get(reference))
                assigned = prediction(record)
                if actual is None or assigned is None:
                    continue
                truth.append(actual)
                predicted.append(assigned)

                # Raw exact match distinguishes "positive" from True or 1.
                left, right = record.get("label"), record.get(reference)
                raw_exact += type(left) is type(right) and left == right

            reference_rows.append([
                name, reference, len(index) - len(truth),
                *metrics(truth, predicted),
                raw_exact, divide(raw_exact, len(truth)),
            ])

    pair_rows, reference_conflicts = [], []
    for (name_a, index_a), (name_b, index_b) in combinations(runs, 2):
        shared = index_a.keys() & index_b.keys()
        labels_a, labels_b = [], []

        for key in shared:
            a, b = prediction(index_a[key]), prediction(index_b[key])
            if a is not None and b is not None:
                labels_a.append(a)
                labels_b.append(b)

        # Agreement/correlation are symmetric; neither run is ground truth.
        n, _, _, _, _, _, _, _, agreement, correlation, kappa = metrics(
            labels_a, labels_b
        )
        pair_rows.append([
            name_a, name_b,
            len(index_a.keys() - index_b.keys()),
            len(index_b.keys() - index_a.keys()),
            len(shared), len(shared) - n, n,
            sum(a == b for a, b in zip(labels_a, labels_b)),
            sum(a != b for a, b in zip(labels_a, labels_b)),
            agreement, correlation, kappa,
        ])

        for column in ("consensus_binary_label", "ground_truth"):
            valid = conflicts = 0
            for key in shared:
                a = binary_label(index_a[key].get(column))
                b = binary_label(index_b[key].get(column))
                if a is not None and b is not None:
                    valid += 1
                    conflicts += a != b
            reference_conflicts.append([
                name_a, name_b, column, valid, conflicts,
            ])

    sections = [
        table("Files", ["Run", "Path", "Records", "Unique usable IDs"], inventory),
        table(
            "Input and prediction checks",
            ["Run", "Format problems", "Missing/invalid IDs",
             "Duplicate IDs", "Records with duplicate IDs",
             "Error records", "Unusable predictions",
             "Low confidence", "Prefilter matches"],
            audits,
        ),
        table(
            "Label counts — all loaded records",
            ["Run", "Column", "Positive / True / 1",
             "Negative / False / 0", "Missing / other",
             "Other values (up to 10)"],
            distributions,
        ),
        table(
            "Predictions versus reference labels",
            ["Run", "Reference", "Excluded unique records",
             *metric_headers, "Raw exact matches", "Raw exact match rate"],
            reference_rows,
        ),
        table(
            "Run-to-run comparison, matched by message_id",
            ["Run A", "Run B", "Only A", "Only B", "Shared IDs",
             "Shared IDs excluded", "Valid pairs", "Agree", "Disagree",
             "Agreement rate", "Correlation (phi/MCC)", "Cohen's kappa"],
            pair_rows,
        ),
        table(
            "Reference consistency between files",
            ["Run A", "Run B", "Column", "Valid shared references",
             "Conflicting references"],
            reference_conflicts,
        ),
        table(
            "Flagged records — first 100",
            ["Run", "Record position", "message_id", "Label", "Flags"],
            flagged[:100],
        ),
        table(
            "File-format problems — first 100",
            ["Run", "Problem"],
            format_problems[:100],
        ),
    ]

    report = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LLM prediction evaluation</title>
<style>
body {font: 14px Arial, sans-serif; margin: 24px; color: #222;}
h2 {margin-top: 28px;}
table {border-collapse: collapse; margin-bottom: 20px;}
th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}
th {background: #f3f4f6;}
tr:nth-child(even) {background: #fafafa;}
p {max-width: 1000px; line-height: 1.5;}
</style>
</head>
<body>
<h1>LLM prediction evaluation</h1>
<p>Positive/True/1 is the positive class. Metrics use normalized binary
labels; raw exact match additionally requires identical types and values.
Errors, missing references, and unknown labels are excluded from metrics.
All occurrences of duplicate IDs are excluded from comparisons.</p>
<p>Correlation is Pearson correlation for binary labels (phi/MCC).
N/A means undefined, including correlation for a constant label vector.
Ground-truth columns are used as supplied in each prediction file.
Prefilter matches are informational, not errors. Counts may overlap.</p>
""" + "\n".join(sections) + "\n</body></html>"

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    if print_html:
        print(report)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_files", nargs="+")
    parser.add_argument("--output", default="evaluation.html")
    parser.add_argument(
        "--print-html", action="store_true",
        help="Also print the HTML source to the terminal.",
    )
    args = parser.parse_args()

    evaluate_predictions(
        args.prediction_files,
        output_html=args.output,
        print_html=args.print_html,
    )
    if not args.print_html:
        print(f"Report saved to: {Path(args.output).resolve()}")