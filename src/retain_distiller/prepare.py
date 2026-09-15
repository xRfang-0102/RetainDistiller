import csv
import json
import re
from pathlib import Path

import soundfile as sf

from .data import read_jsonl


ARPABET = "AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH".split()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_dictionary(path, keep_stress=False):
    dictionary = {}
    with Path(path).open(encoding="latin-1") as stream:
        for line in stream:
            parts = line.strip().split()
            if len(parts) < 2 or parts[0].startswith((";;;", "#")):
                continue
            word = re.sub(r"\(\d+\)$", "", parts[0]).upper()
            phones = parts[1:]
            if "#" in phones:
                phones = phones[:phones.index("#")]
            if not keep_stress:
                phones = [re.sub(r"\d+$", "", phone) for phone in phones]
            dictionary.setdefault(word, phones)
    if not dictionary:
        raise ValueError("CMU dictionary is empty")
    return dictionary


def librispeech(args):
    root = Path(args.root).resolve()
    subset = root / args.split
    dictionary = load_dictionary(args.cmudict, args.keep_stress) if args.cmudict else None
    if not args.text_only and dictionary is None:
        raise ValueError("Provide --cmudict or explicitly choose --text-only")
    rows, rejected = [], []
    for transcript in sorted(subset.rglob("*.trans.txt")):
        for line in transcript.read_text(encoding="utf-8").splitlines():
            utterance, text = line.split(maxsplit=1)
            audio = transcript.parent / f"{utterance}.flac"
            info = sf.info(str(audio))
            row = {"id": utterance, "audio": str(audio), "text": text.upper(), "duration": info.duration}
            if dictionary is not None:
                words = re.findall(r"[A-Z]+(?:'[A-Z]+)*", text.upper())
                missing = sorted(set(words) - dictionary.keys())
                if missing:
                    rejected.append({"id": utterance, "unknown_words": missing})
                    continue
                row["tokens"] = [phone for word in words for phone in dictionary[word]]
            rows.append(row)
    if not rows:
        raise ValueError(f"No usable utterances under {subset}")
    if rejected and args.oov == "error":
        raise ValueError(f"{len(rejected)} utterances contain OOV words; extend the local dictionary or use --oov skip. First: {rejected[0]}")
    write_manifest(args.output, rows)
    report = {"split": args.split, "kept": len(rows), "rejected": rejected, "hours": sum(row["duration"] for row in rows) / 3600}
    write_json(str(args.output) + ".report.json", report)
    if args.vocab:
        if dictionary is None:
            symbols = ["<blank>", "|", "'", *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")]
        elif args.keep_stress:
            symbols = ["<blank>", *sorted({phone for phones in dictionary.values() for phone in phones})]
        else:
            symbols = ["<blank>", *ARPABET]
        write_json(args.vocab, symbols)
    print(json.dumps({key: value for key, value in report.items() if key != "rejected"}))


def convert_csv(args):
    root = Path(args.root).resolve()
    rows = []
    with Path(args.input).open(encoding="utf-8", newline="") as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            audio = (root / row[args.audio_column]).resolve()
            item = {"id": row[args.id_column] if args.id_column else f"{Path(args.output).stem}-{i}", "audio": str(audio)}
            if args.text_column:
                item["text"] = row[args.text_column].upper()
            if args.label_columns:
                item["label"] = "|".join(row[column] for column in args.label_columns)
            rows.append(item)
    if not rows:
        raise ValueError("Empty CSV")
    write_manifest(args.output, rows)


def label_vocabulary(args):
    rows = read_jsonl(args.manifest)
    write_json(args.output, sorted({str(row["label"]) for row in rows}))


def character_vocabulary(args):
    write_json(args.output, ["<blank>", "|", "'", *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")])
