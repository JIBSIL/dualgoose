"""Pinned OpenWebText manifest and conversion utilities."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

TOKEN_DTYPES = {
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int64": np.int64,
}


def load_policy(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def resolve_token_dtype(token_dtype: str | np.dtype, mask_token_id: int) -> np.dtype:
    dtype = np.dtype(token_dtype)
    if dtype.name not in TOKEN_DTYPES:
        choices = ", ".join(sorted(TOKEN_DTYPES))
        raise ValueError(f"token_dtype must be one of: {choices}")
    if mask_token_id < 0:
        raise ValueError("mask_token_id must be non-negative")
    if mask_token_id > np.iinfo(dtype).max:
        raise ValueError(
            f"mask_token_id {mask_token_id} does not fit in token_dtype {dtype.name}"
        )
    return dtype


def default_manifest_path(policy: dict[str, object]) -> Path:
    revision = str(policy["dataset_revision"])
    return Path("manifests") / "openwebtext" / f"{revision}.jsonl"


def shard_rows_from_siblings(
    policy: dict[str, object],
    siblings: Iterable[object],
) -> list[dict[str, object]]:
    pattern = str(policy["selected_file_pattern"])
    dataset_name = str(policy["dataset_name"])
    dataset_url = str(policy["dataset_url"])
    revision = str(policy["dataset_revision"])
    rows: list[dict[str, object]] = []
    for sibling in siblings:
        filename = getattr(sibling, "rfilename")
        if not fnmatch.fnmatch(filename, pattern):
            continue
        lfs = getattr(sibling, "lfs", None)
        rows.append(
            {
                "dataset_name": dataset_name,
                "dataset_url": dataset_url,
                "dataset_revision": revision,
                "rfilename": filename,
                "size": getattr(sibling, "size", None),
                "blob_id": getattr(sibling, "blob_id", None),
                "lfs_sha256": getattr(lfs, "sha256", None) if lfs is not None else None,
                "resolve_url": (
                    f"https://huggingface.co/datasets/{dataset_name}/resolve/"
                    f"{revision}/{filename}"
                ),
            }
        )
    return sorted(rows, key=lambda row: str(row["rfilename"]))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def create_manifest(policy: dict[str, object]) -> list[dict[str, object]]:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info(
        str(policy["dataset_name"]),
        revision=str(policy["dataset_revision"]),
        files_metadata=True,
    )
    if info.sha != policy["dataset_revision"]:
        raise RuntimeError(f"resolved revision {info.sha} does not match policy")
    rows = shard_rows_from_siblings(policy, info.siblings)
    expected = int(policy["expected_shards"])
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} shards, found {len(rows)}")
    return rows


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_shard(
    row: dict[str, object],
    *,
    local_dir: str | Path,
) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=str(row["dataset_name"]),
        repo_type="dataset",
        revision=str(row["dataset_revision"]),
        filename=str(row["rfilename"]),
        local_dir=local_dir,
    )
    local_path = Path(path)
    expected = row.get("lfs_sha256")
    if expected is not None:
        actual = sha256_file(local_path)
        if actual != expected:
            raise RuntimeError(
                f"sha256 mismatch for {row['rfilename']}: expected {expected}, got {actual}"
            )
    return local_path


def resolve_shard_path(
    row: dict[str, object],
    *,
    local_dir: str | Path,
) -> Path:
    if row.get("local_path") is not None:
        local_path = Path(str(row["local_path"]))
        expected = row.get("lfs_sha256")
        if expected is not None:
            actual = sha256_file(local_path)
            if actual != expected:
                raise RuntimeError(
                    f"sha256 mismatch for {row['rfilename']}: "
                    f"expected {expected}, got {actual}"
                )
        return local_path
    return download_shard(row, local_dir=local_dir)


def parquet_to_jsonl(
    parquet_path: str | Path,
    output_path: str | Path,
    *,
    text_key: str,
    source: dict[str, object],
    max_rows: int | None = None,
    batch_size: int = 1024,
) -> int:
    import pyarrow.parquet as pq

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    parquet_file = pq.ParquetFile(parquet_path)
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for batch in parquet_file.iter_batches(columns=[text_key], batch_size=batch_size):
            values = batch.column(0).to_pylist()
            for value in values:
                if not isinstance(value, str):
                    continue
                row = {
                    "text": value,
                    "source_dataset": source.get("dataset_name"),
                    "source_revision": source.get("dataset_revision"),
                    "source_file": source.get("rfilename"),
                    "source_row": written,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                if max_rows is not None and written >= max_rows:
                    return written
    return written


def alpha_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(char.isalpha() for char in text) / len(text)


def document_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def split_for_document(text: str, validation_fraction: float) -> str:
    if validation_fraction <= 0.0:
        return "train"
    digest = hashlib.sha256(text.strip().encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if bucket < validation_fraction else "train"


class StreamingShardWriter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        split: str,
        shard_tokens: int,
        token_dtype: str | np.dtype,
        metadata: dict[str, object],
    ):
        if shard_tokens <= 0:
            raise ValueError("shard_tokens must be positive")
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_tokens = shard_tokens
        self.token_dtype = np.dtype(token_dtype)
        self.metadata = metadata
        self.buffer: list[int] = []
        self.shards: list[dict[str, object]] = []
        self.token_count = 0

    def _write_shard(self, tokens: list[int]) -> None:
        info = np.iinfo(self.token_dtype)
        if tokens and (min(tokens) < info.min or max(tokens) > info.max):
            raise ValueError(f"token shard values do not fit in {self.token_dtype.name}")
        filename = f"shard_{len(self.shards):05d}.npy"
        np.save(self.output / filename, np.asarray(tokens, dtype=self.token_dtype))
        self.shards.append({"file": filename, "token_count": len(tokens)})
        self.token_count += len(tokens)

    def add_tokens(self, token_ids: list[int]) -> None:
        remaining = token_ids
        while remaining:
            available = self.shard_tokens - len(self.buffer)
            self.buffer.extend(remaining[:available])
            remaining = remaining[available:]
            if len(self.buffer) == self.shard_tokens:
                self._write_shard(self.buffer)
                self.buffer = []

    def finish(self) -> dict[str, object]:
        if self.buffer:
            self._write_shard(self.buffer)
            self.buffer = []
        split_metadata = {
            **self.metadata,
            "format": "sharded_npy",
            "split": self.split,
            "token_count": self.token_count,
            "shards": self.shards,
        }
        (self.output / "metadata.json").write_text(
            json.dumps(split_metadata, indent=2) + "\n"
        )
        return split_metadata


def encode_text(text: str, *, tokenizer: str, encoding: object | None) -> list[int]:
    if tokenizer == "gpt2":
        if encoding is None:
            raise ValueError("gpt2 tokenizer requires an encoding")
        return list(encoding.encode(text))  # type: ignore[attr-defined]
    if tokenizer == "byte":
        return [byte for byte in text.encode("utf-8")]
    raise ValueError("tokenizer must be gpt2 or byte")


def stream_pack_manifest(
    policy: dict[str, object],
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    download_dir: str | Path,
    tokenizer: str,
    vocab_size: int,
    mask_token_id: int,
    shard_tokens: int,
    validation_fraction: float,
    min_chars: int,
    min_alpha_fraction: float,
    deduplicate: bool,
    max_shards: int | None = None,
    max_rows_per_shard: int | None = None,
    batch_size: int = 1024,
    token_dtype: str | np.dtype = "int64",
    max_output_tokens: int | None = None,
    delete_raw_after_pack: bool = False,
) -> dict[str, object]:
    import pyarrow.parquet as pq

    if validation_fraction < 0.0 or validation_fraction >= 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if min_alpha_fraction < 0.0 or min_alpha_fraction > 1.0:
        raise ValueError("min_alpha_fraction must be in [0, 1]")
    if max_output_tokens is not None and max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    if tokenizer == "gpt2":
        import tiktoken

        encoding = tiktoken.get_encoding("gpt2")
        if vocab_size < encoding.n_vocab + 1:
            raise ValueError("vocab_size must include GPT-2 tokens plus mask")
        if mask_token_id < encoding.n_vocab:
            raise ValueError("mask_token_id must not collide with GPT-2 token ids")
    elif tokenizer == "byte":
        encoding = None
        if vocab_size < 257 or mask_token_id < 256:
            raise ValueError("byte tokenizer requires vocab/mask ids above byte range")
    else:
        raise ValueError("tokenizer must be gpt2 or byte")
    resolved_token_dtype = resolve_token_dtype(token_dtype, mask_token_id)

    rows = read_jsonl(manifest_path)
    if max_shards is not None:
        rows = rows[:max_shards]
    if not rows:
        raise ValueError("manifest produced no shards")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "source_name": policy["dataset_name"],
        "source_url": policy["dataset_url"],
        "dataset_version": policy["dataset_revision"],
        "archive_reference_url": policy.get("archive_reference_url"),
    }
    common_metadata = {
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "mask_token_id": mask_token_id,
        "dtype": resolved_token_dtype.name,
        "input_format": "parquet",
        "raw_text_key": policy["raw_text_key"],
        "manifest": str(manifest_path),
        "processed_shards": [row["rfilename"] for row in rows],
        "min_chars": min_chars,
        "min_alpha_fraction": min_alpha_fraction,
        "deduplicate_exact": deduplicate,
        "validation_fraction": validation_fraction,
        "max_output_tokens": max_output_tokens,
        "delete_raw_after_pack": delete_raw_after_pack,
        "provenance": provenance,
    }
    writers = {
        "train": StreamingShardWriter(
            output / "train",
            split="train",
            shard_tokens=shard_tokens,
            token_dtype=resolved_token_dtype,
            metadata=common_metadata,
        ),
        "validation": StreamingShardWriter(
            output / "validation",
            split="validation",
            shard_tokens=shard_tokens,
            token_dtype=resolved_token_dtype,
            metadata=common_metadata,
        ),
    }
    seen_hashes: set[str] = set()
    filtered_documents: list[dict[str, object]] = []
    truncated_documents: list[dict[str, object]] = []
    deleted_raw_shards: list[str] = []
    shard_summaries: list[dict[str, object]] = []
    document_count = 0
    written_tokens = 0
    token_budget_reached = False
    stop_requested = False

    for row in rows:
        if stop_requested:
            break
        parquet_path = resolve_shard_path(row, local_dir=download_dir)
        parquet_file = pq.ParquetFile(parquet_path)
        row_count = 0
        kept = 0
        for batch in parquet_file.iter_batches(
            columns=[str(policy["raw_text_key"])],
            batch_size=batch_size,
        ):
            for value in batch.column(0).to_pylist():
                row_count += 1
                if max_rows_per_shard is not None and row_count > max_rows_per_shard:
                    break
                source = f"{row['rfilename']}:{row_count}"
                if not isinstance(value, str):
                    filtered_documents.append({"source": source, "reason": "non_string"})
                    continue
                text = value.strip()
                if len(text) < min_chars:
                    filtered_documents.append(
                        {"source": source, "reason": "below_min_chars"}
                    )
                    continue
                if alpha_fraction(text) < min_alpha_fraction:
                    filtered_documents.append(
                        {"source": source, "reason": "below_min_alpha_fraction"}
                    )
                    continue
                digest = document_hash(text)
                if deduplicate:
                    if digest in seen_hashes:
                        filtered_documents.append(
                            {"source": source, "reason": "duplicate_document"}
                        )
                        continue
                    seen_hashes.add(digest)
                token_ids = encode_text(text, tokenizer=tokenizer, encoding=encoding)
                if not token_ids:
                    filtered_documents.append({"source": source, "reason": "empty_tokens"})
                    continue
                if max(token_ids) >= mask_token_id:
                    raise ValueError("clean token ids must be below mask_token_id")
                if max_output_tokens is not None:
                    remaining_budget = max_output_tokens - written_tokens
                    if remaining_budget <= 0:
                        token_budget_reached = True
                        stop_requested = True
                        break
                    if len(token_ids) > remaining_budget:
                        token_ids = token_ids[:remaining_budget]
                        truncated_documents.append(
                            {"source": source, "reason": "max_output_tokens"}
                        )
                        token_budget_reached = True
                        stop_requested = True
                split = split_for_document(text, validation_fraction)
                writers[split].add_tokens(token_ids)
                written_tokens += len(token_ids)
                document_count += 1
                kept += 1
                if (
                    max_output_tokens is not None
                    and written_tokens >= max_output_tokens
                ):
                    token_budget_reached = True
                    stop_requested = True
                    break
            if (
                stop_requested
                or (
                    max_rows_per_shard is not None
                    and row_count >= max_rows_per_shard
                )
            ):
                break
        if delete_raw_after_pack and row.get("local_path") is None:
            try:
                parquet_path.resolve().relative_to(Path(download_dir).resolve())
            except ValueError as exc:
                raise RuntimeError(
                    f"refusing to delete raw shard outside download_dir: {parquet_path}"
                ) from exc
            parquet_path.unlink(missing_ok=True)
            deleted_raw_shards.append(str(parquet_path))
        shard_summaries.append(
            {
                "rfilename": row["rfilename"],
                "rows_seen": row_count if max_rows_per_shard is None else min(row_count, max_rows_per_shard),
                "documents_kept": kept,
                "size": row.get("size"),
                "lfs_sha256": row.get("lfs_sha256"),
            }
        )

    train_metadata = writers["train"].finish()
    validation_metadata = writers["validation"].finish()
    metadata = {
        **common_metadata,
        "format": "stream_split_packed",
        "token_count": train_metadata["token_count"] + validation_metadata["token_count"],
        "document_count": document_count,
        "filtered_documents": filtered_documents,
        "truncated_documents": truncated_documents,
        "token_budget_reached": token_budget_reached,
        "deleted_raw_shards": deleted_raw_shards,
        "shards": shard_summaries,
        "splits": {
            "train": {
                "path": "train",
                "token_count": train_metadata["token_count"],
                "format": train_metadata["format"],
            },
            "validation": {
                "path": "validation",
                "token_count": validation_metadata["token_count"],
                "format": validation_metadata["format"],
            },
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="configs/owt_source_policy.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--output", default=None)

    convert = subparsers.add_parser("convert")
    convert.add_argument("--manifest", default=None)
    convert.add_argument("--shard-index", type=int, default=0)
    convert.add_argument("--download-dir", default="data/openwebtext/raw")
    convert.add_argument("--output-jsonl", required=True)
    convert.add_argument("--max-rows", type=int, default=None)
    convert.add_argument("--batch-size", type=int, default=1024)

    stream_pack = subparsers.add_parser("stream-pack")
    stream_pack.add_argument("--manifest", default=None)
    stream_pack.add_argument("--download-dir", default="data/openwebtext/raw")
    stream_pack.add_argument("--output", required=True)
    stream_pack.add_argument("--tokenizer", choices=["byte", "gpt2"], default=None)
    stream_pack.add_argument("--vocab-size", type=int, default=None)
    stream_pack.add_argument("--mask-token-id", type=int, default=None)
    stream_pack.add_argument("--shard-tokens", type=int, default=1_000_000)
    stream_pack.add_argument("--validation-fraction", type=float, default=None)
    stream_pack.add_argument("--min-chars", type=int, default=None)
    stream_pack.add_argument("--min-alpha-fraction", type=float, default=None)
    stream_pack.add_argument("--deduplicate", action="store_true")
    stream_pack.add_argument("--max-shards", type=int, default=None)
    stream_pack.add_argument("--max-rows-per-shard", type=int, default=None)
    stream_pack.add_argument("--max-output-tokens", type=int, default=None)
    stream_pack.add_argument("--delete-raw-after-pack", action="store_true")
    stream_pack.add_argument("--batch-size", type=int, default=1024)
    stream_pack.add_argument("--dtype", choices=sorted(TOKEN_DTYPES), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = load_policy(args.policy)
    if args.command == "manifest":
        rows = create_manifest(policy)
        output = Path(args.output) if args.output else default_manifest_path(policy)
        write_jsonl(output, rows)
        print(json.dumps({"manifest": str(output), "shards": len(rows)}, indent=2))
        return

    if args.command == "convert":
        manifest_path = Path(args.manifest) if args.manifest else default_manifest_path(policy)
        rows = read_jsonl(manifest_path)
        row = rows[args.shard_index]
        parquet_path = download_shard(row, local_dir=args.download_dir)
        written = parquet_to_jsonl(
            parquet_path,
            args.output_jsonl,
            text_key=str(policy["raw_text_key"]),
            source=row,
            max_rows=args.max_rows,
            batch_size=args.batch_size,
        )
        print(
            json.dumps(
                {
                    "source": str(row["rfilename"]),
                    "parquet": str(parquet_path),
                    "jsonl": args.output_jsonl,
                    "rows": written,
                },
                indent=2,
            )
        )
        return

    if args.command == "stream-pack":
        manifest_path = Path(args.manifest) if args.manifest else default_manifest_path(policy)
        tokenizer = args.tokenizer or str(policy["tokenizer"])
        mask_token_id = (
            args.mask_token_id
            if args.mask_token_id is not None
            else int(policy["mask_token_id"])
        )
        vocab_size = (
            args.vocab_size if args.vocab_size is not None else int(policy["vocab_size"])
        )
        filters = policy.get("minimum_filters", {})
        if not isinstance(filters, dict):
            filters = {}
        metadata = stream_pack_manifest(
            policy,
            manifest_path=manifest_path,
            output_dir=args.output,
            download_dir=args.download_dir,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            shard_tokens=args.shard_tokens,
            validation_fraction=(
                args.validation_fraction
                if args.validation_fraction is not None
                else float(policy["validation_fraction"])
            ),
            min_chars=(
                args.min_chars
                if args.min_chars is not None
                else int(filters.get("min_chars", 0))
            ),
            min_alpha_fraction=(
                args.min_alpha_fraction
                if args.min_alpha_fraction is not None
                else float(filters.get("min_alpha_fraction", 0.0))
            ),
            deduplicate=args.deduplicate
            or bool(filters.get("exact_document_deduplicate", False)),
            max_shards=args.max_shards,
            max_rows_per_shard=args.max_rows_per_shard,
            batch_size=args.batch_size,
            token_dtype=args.dtype or str(policy.get("packed_dtype", "int64")),
            max_output_tokens=args.max_output_tokens,
            delete_raw_after_pack=args.delete_raw_after_pack,
        )
        print(
            json.dumps(
                {
                    "output": args.output,
                    "token_count": metadata["token_count"],
                    "document_count": metadata["document_count"],
                    "dtype": metadata["dtype"],
                    "token_budget_reached": metadata["token_budget_reached"],
                    "splits": metadata["splits"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
