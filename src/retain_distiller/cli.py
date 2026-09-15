import argparse
import json

from .config import load_config


def config_arguments(parser):
    parser.add_argument("--config", required=True)
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--set", nargs="*", default=[])


def evaluation_arguments(parser):
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", required=True)


def make_parser():
    parser = argparse.ArgumentParser(prog="retain-distiller")
    commands = parser.add_subparsers(dest="command", required=True)
    config_arguments(commands.add_parser("show-config"))
    for stage in ("probe", "distill", "downstream"):
        command = commands.add_parser(stage)
        config_arguments(command)
        command.add_argument("--resume")
    evaluation_arguments(commands.add_parser("evaluate"))
    analysis = commands.add_parser("analyze")
    evaluation_arguments(analysis)
    analysis.add_argument("--max-frames", type=int)
    analysis.add_argument("--probe")
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    inference = commands.add_parser("infer")
    inference.add_argument("--model", required=True)
    inference.add_argument("--audio", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--device", default="cpu")
    retention = commands.add_parser("retention")
    retention.add_argument("--student", required=True)
    retention.add_argument("--teacher", required=True)
    retention.add_argument("--metric", choices=["per", "accuracy"], required=True)
    retention.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare").add_subparsers(dest="format", required=True)
    libri = prepare.add_parser("librispeech")
    libri.add_argument("--root", required=True)
    libri.add_argument("--split", required=True)
    libri.add_argument("--output", required=True)
    libri.add_argument("--cmudict")
    libri.add_argument("--vocab")
    libri.add_argument("--text-only", action="store_true")
    libri.add_argument("--keep-stress", action="store_true")
    libri.add_argument("--oov", choices=["error", "skip"], default="error")
    csv = prepare.add_parser("csv")
    csv.add_argument("--input", required=True)
    csv.add_argument("--root", required=True)
    csv.add_argument("--output", required=True)
    csv.add_argument("--audio-column", default="path")
    csv.add_argument("--id-column")
    csv.add_argument("--text-column")
    csv.add_argument("--label-columns", nargs="+")
    labels = prepare.add_parser("labels")
    labels.add_argument("--manifest", required=True)
    labels.add_argument("--output", required=True)
    characters = prepare.add_parser("characters")
    characters.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.command == "show-config":
        print(json.dumps(load_config(args.config, args.overlay, args.set), indent=2))
    elif args.command in {"probe", "distill", "downstream"}:
        from .engine import train
        train(load_config(args.config, args.overlay, args.set), args.command, args.resume)
    elif args.command == "prepare":
        from .prepare import character_vocabulary, convert_csv, label_vocabulary, librispeech
        {"librispeech": librispeech, "csv": convert_csv, "labels": label_vocabulary, "characters": character_vocabulary}[args.format](args)
    elif args.command == "evaluate":
        from .engine import evaluate_checkpoint
        evaluate_checkpoint(args.checkpoint, args.manifest, args.device, args.precision, args.batch_size, args.output)
    elif args.command == "analyze":
        from .analysis import analyze
        if args.max_frames is not None and args.max_frames < 2:
            raise ValueError("max-frames must be at least two")
        analyze(args.checkpoint, args.manifest, args.output, args.device, args.precision, args.batch_size, args.max_frames, args.probe)
    elif args.command == "export":
        from .inference import export_student
        export_student(args.checkpoint, args.output)
    elif args.command == "infer":
        from .inference import infer
        infer(args.model, args.audio, args.output, args.device)
    elif args.command == "retention":
        from .analysis import retention_report
        retention_report(args.student, args.teacher, args.metric, args.output)


if __name__ == "__main__":
    main()
