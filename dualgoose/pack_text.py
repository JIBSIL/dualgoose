"""Pack text files into a contiguous token stream."""

from __future__ import annotations

import argparse
import json

from .text_data import discover_text_paths, pack_text_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="*", default=[])
    parser.add_argument("--input-dir", nargs="*", default=[])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--pattern", default="*.txt")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--input-format", choices=["text", "jsonl"], default="text")
    parser.add_argument("--jsonl-text-key", default="text")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", choices=["byte", "gpt2"], default="gpt2")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--shard-tokens", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--min-bytes", type=int, default=0)
    parser.add_argument("--min-chars", type=int, default=0)
    parser.add_argument("--min-alpha-fraction", type=float, default=0.0)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--source-license", default=None)
    parser.add_argument("--dataset-version", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provenance = {
        key: value
        for key, value in {
            "source_name": args.source_name,
            "source_url": args.source_url,
            "source_license": args.source_license,
            "dataset_version": args.dataset_version,
        }.items()
        if value is not None
    }
    paths = discover_text_paths(
        inputs=args.input,
        input_dirs=args.input_dir,
        manifest=args.manifest,
        pattern=args.pattern,
        max_files=args.max_files,
    )
    metadata = pack_text_files(
        paths,
        args.output,
        tokenizer=args.tokenizer,
        vocab_size=args.vocab_size,
        mask_token_id=args.mask_token_id,
        shard_tokens=args.shard_tokens,
        validation_fraction=args.validation_fraction,
        min_bytes=args.min_bytes,
        input_format=args.input_format,
        jsonl_text_key=args.jsonl_text_key,
        min_chars=args.min_chars,
        min_alpha_fraction=args.min_alpha_fraction,
        deduplicate=args.deduplicate,
        provenance=provenance,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
