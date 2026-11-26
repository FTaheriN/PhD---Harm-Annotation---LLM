import argparse
import json
from math import ceil
from pathlib import Path

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments, ModelArguments
from llamafactory.model import load_tokenizer


def read_rows(path):
    with path.open(encoding="utf-8-sig") as source:
        if path.suffix.lower() == ".jsonl":
            for line_number, line in enumerate(source, start=1):
                if line.strip():
                    try:
                        yield line_number, json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON on line {line_number}") from exc
        else:
            rows = json.load(source)
            if not isinstance(rows, list):
                raise ValueError("A JSON input file must contain an array of records.")
            yield from enumerate(rows, start=1)


def convert(row, args):
    if not isinstance(row, dict):
        raise ValueError("Expected a JSON object.")

    instruction = (
        row[args.instruction_field] if args.instruction_field else args.default_instruction
    )
    input_text = row[args.input_field] if args.input_field else ""
    output = row[args.output_field]

    if isinstance(output, (dict, list)):
        output = json.dumps(output, ensure_ascii=False, allow_nan=False)
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Instruction must be a nonempty string.")
    if not isinstance(input_text, str):
        raise ValueError("Input must be a string.")
    if not isinstance(output, str) or not output.strip():
        raise ValueError("Output must be a nonempty string, JSON object, or JSON array.")

    return {"instruction": instruction, "input": input_text, "output": output}


def token_lengths(row, tokenizer, processor, template):
    prompt = "\n".join(value for value in (row["instruction"], row["input"]) if value)
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": row["output"]},
    ]
    plugin = template.mm_plugin
    messages = plugin.process_messages(messages, [], [], [], processor)
    prefix, _ = plugin.process_token_ids([], [], [], [], [], tokenizer, processor)
    pairs = template.encode_multiturn(tokenizer, messages, None, None, False)

    # Mirrors the pinned SFT processor for text-only examples, before truncation.
    eos = int(template.efficient_eos)
    total = len(prefix) + eos + sum(len(source) + len(target) for source, target in pairs)
    response = eos + sum(len(target) for _, target in pairs)
    return total, response


def summarize(lengths):
    ordered = sorted(lengths)
    return {
        "median": ordered[ceil(len(ordered) * 0.50) - 1],
        "p95": ordered[ceil(len(ordered) * 0.95) - 1],
        "p99": ordered[ceil(len(ordered) * 0.99) - 1],
        "max": ordered[-1],
    }


def write_json(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--instruction-field", required=True)
    parser.add_argument("--input-field", required=True)
    parser.add_argument("--output-field", required=True)
    parser.add_argument("--default-instruction", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cutoff", type=int, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.cutoff < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--cutoff and --limit must be positive.")

    model_args = ModelArguments(
        model_name_or_path=args.model,
        model_revision=args.revision,
    )
    tokenizer_bundle = load_tokenizer(model_args)
    tokenizer = tokenizer_bundle["tokenizer"]
    processor = tokenizer_bundle["processor"]
    template = get_template_and_fix_tokenizer(
        tokenizer, DataArguments(template=args.template, cutoff_len=args.cutoff)
    )

    rows, totals, responses, over_cutoff = [], [], [], []
    for number, raw in read_rows(args.source):
        try:
            row = convert(raw, args)
        except (KeyError, ValueError) as exc:
            raise SystemExit(f"Record {number}: {exc}") from exc

        total, response = token_lengths(row, tokenizer, processor, template)
        rows.append(row)
        totals.append(total)
        responses.append(response)
        if total > args.cutoff:
            over_cutoff.append({"record": number, "tokens": total})
        if args.limit is not None and len(rows) >= args.limit:
            break

    if not rows:
        raise SystemExit("No training rows found.")

    report = {
        "model": args.model,
        "model_revision": args.revision,
        "template": args.template,
        "rows": len(rows),
        "cutoff": args.cutoff,
        "sequence_tokens": summarize(totals),
        "response_tokens": summarize(responses),
        "over_cutoff": len(over_cutoff),
        "first_20_over_cutoff": over_cutoff[:20],
    }
    args.destination.mkdir(parents=True, exist_ok=False)
    write_json(args.destination / "train.json", rows)
    write_json(args.destination / "dataset_info.json", {
        "annotation_train": {
            "file_name": "train.json",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
    })
    write_json(args.destination / "length_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()