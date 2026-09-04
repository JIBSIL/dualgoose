"""Text batchers for training-loop readiness checks."""

from __future__ import annotations

import json
from pathlib import Path
import re
import hashlib

import numpy as np
import torch

from .diffusion import corrupt_absorbing
from .toy_data import ToyBatch


class ByteTextBatcher:
    """Sample fixed-length byte-token sequences from a local text file."""

    def __init__(self, path: str | Path, vocab_size: int, mask_token_id: int):
        data = Path(path).read_bytes()
        if not data:
            raise ValueError("text file is empty")
        if vocab_size <= 2:
            raise ValueError("vocab_size must leave room for clean tokens and mask")
        clean_vocab = vocab_size - 1
        self.tokens = torch.tensor([byte % clean_vocab for byte in data], dtype=torch.long)
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        *,
        device: torch.device | str | None = None,
    ) -> ToyBatch:
        if self.tokens.numel() < seq_len:
            repeats = (seq_len + self.tokens.numel() - 1) // self.tokens.numel()
            tokens = self.tokens.repeat(repeats)
        else:
            tokens = self.tokens
        max_start = tokens.numel() - seq_len + 1
        starts = torch.randint(0, max_start, (batch_size,))
        clean = torch.stack([tokens[start : start + seq_len] for start in starts], dim=0)
        clean = clean.to(device=device)
        t = torch.rand(batch_size, device=device).clamp(0.05, 0.95)
        corrupted, target_mask = corrupt_absorbing(clean, t, self.mask_token_id)
        return ToyBatch(clean, corrupted, target_mask, t)


class WordTextBatcher:
    """Sample fixed-length word-token sequences from a local text file."""

    def __init__(self, path: str | Path, vocab_size: int, mask_token_id: int):
        text = Path(path).read_text(encoding="utf-8")
        pieces = re.findall(r"\w+|[^\w\s]", text.lower())
        if not pieces:
            raise ValueError("text file produced no tokens")
        clean_vocab = vocab_size - 1
        if clean_vocab < 4:
            raise ValueError("vocab_size must leave room for word ids and mask")
        counts: dict[str, int] = {}
        for piece in pieces:
            counts[piece] = counts.get(piece, 0) + 1
        ordered = sorted(counts, key=lambda item: (-counts[item], item))
        self.vocab = {"<unk>": 0}
        for piece in ordered[: clean_vocab - 1]:
            self.vocab[piece] = len(self.vocab)
        self.tokens = torch.tensor(
            [self.vocab.get(piece, 0) for piece in pieces], dtype=torch.long
        )
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        *,
        device: torch.device | str | None = None,
    ) -> ToyBatch:
        if self.tokens.numel() < seq_len:
            repeats = (seq_len + self.tokens.numel() - 1) // self.tokens.numel()
            tokens = self.tokens.repeat(repeats)
        else:
            tokens = self.tokens
        max_start = tokens.numel() - seq_len + 1
        starts = torch.randint(0, max_start, (batch_size,))
        clean = torch.stack([tokens[start : start + seq_len] for start in starts], dim=0)
        clean = clean.to(device=device)
        t = torch.rand(batch_size, device=device).clamp(0.05, 0.95)
        corrupted, target_mask = corrupt_absorbing(clean, t, self.mask_token_id)
        return ToyBatch(clean, corrupted, target_mask, t)


class GPT2TextBatcher:
    """Sample fixed-length GPT-2 BPE token sequences from a local text file."""

    def __init__(self, path: str | Path, vocab_size: int, mask_token_id: int):
        import tiktoken

        self.encoding = tiktoken.get_encoding("gpt2")
        if vocab_size < self.encoding.n_vocab + 1:
            raise ValueError("vocab_size must be at least GPT-2 vocab plus one mask token")
        if mask_token_id < self.encoding.n_vocab:
            raise ValueError("mask_token_id must not collide with GPT-2 token ids")
        text = Path(path).read_text(encoding="utf-8")
        token_ids = self.encoding.encode(text)
        if not token_ids:
            raise ValueError("text file produced no GPT-2 tokens")
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        *,
        device: torch.device | str | None = None,
    ) -> ToyBatch:
        if self.tokens.numel() < seq_len:
            repeats = (seq_len + self.tokens.numel() - 1) // self.tokens.numel()
            tokens = self.tokens.repeat(repeats)
        else:
            tokens = self.tokens
        max_start = tokens.numel() - seq_len + 1
        starts = torch.randint(0, max_start, (batch_size,))
        clean = torch.stack([tokens[start : start + seq_len] for start in starts], dim=0)
        clean = clean.to(device=device)
        t = torch.rand(batch_size, device=device).clamp(0.05, 0.95)
        corrupted, target_mask = corrupt_absorbing(clean, t, self.mask_token_id)
        return ToyBatch(clean, corrupted, target_mask, t)


class PackedTextBatcher:
    """Sample fixed-length token sequences from packed `.npy` token streams."""

    def __init__(self, path: str | Path, vocab_size: int, mask_token_id: int):
        root = Path(path)
        token_path = root if root.suffix == ".npy" else root / "tokens.npy"
        metadata_path = root / "metadata.json" if root.is_dir() else token_path.with_name("metadata.json")
        self.metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )

        if token_path.exists():
            arrays = [np.load(token_path, mmap_mode="r")]
        elif "shards" in self.metadata:
            base = metadata_path.parent
            arrays = [
                np.load(base / str(shard["file"]), mmap_mode="r")
                for shard in self.metadata["shards"]
            ]
        else:
            raise FileNotFoundError(f"packed token file not found: {token_path}")

        if not arrays:
            raise ValueError("packed token file is empty")
        for tokens in arrays:
            if tokens.ndim != 1:
                raise ValueError("packed tokens must be one-dimensional arrays")
            if tokens.size == 0:
                raise ValueError("packed token shard is empty")
            if tokens.max() >= mask_token_id:
                raise ValueError("packed clean tokens must be below mask_token_id")
        if vocab_size <= mask_token_id:
            raise ValueError("vocab_size must include the mask token")
        metadata_mask = self.metadata.get("mask_token_id")
        if metadata_mask is not None and int(metadata_mask) != mask_token_id:
            raise ValueError("mask_token_id does not match packed metadata")
        metadata_vocab = self.metadata.get("vocab_size")
        if metadata_vocab is not None and int(metadata_vocab) != vocab_size:
            raise ValueError("vocab_size does not match packed metadata")
        self.arrays = arrays
        self.total_tokens = int(sum(tokens.size for tokens in arrays))
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

    @staticmethod
    def _slice_array(tokens: np.ndarray, start: int, seq_len: int) -> torch.Tensor:
        if tokens.size < seq_len:
            repeats = (seq_len + tokens.size - 1) // tokens.size
            tokens = np.tile(np.asarray(tokens), repeats)
            start = 0
        return torch.from_numpy(np.array(tokens[start : start + seq_len], copy=True)).long()

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        *,
        device: torch.device | str | None = None,
    ) -> ToyBatch:
        shard_ids = torch.randint(0, len(self.arrays), (batch_size,))
        rows = []
        for shard_id in shard_ids:
            tokens = self.arrays[int(shard_id)]
            max_start = max(1, tokens.shape[0] - seq_len + 1)
            start = int(torch.randint(0, max_start, (1,)).item())
            rows.append(self._slice_array(tokens, start, seq_len))
        clean = torch.stack(rows, dim=0).to(device=device)
        t = torch.rand(batch_size, device=device).clamp(0.05, 0.95)
        corrupted, target_mask = corrupt_absorbing(clean, t, self.mask_token_id)
        return ToyBatch(clean, corrupted, target_mask, t)


def discover_text_paths(
    *,
    inputs: list[str | Path] | None = None,
    input_dirs: list[str | Path] | None = None,
    manifest: str | Path | None = None,
    pattern: str = "*.txt",
    max_files: int | None = None,
) -> list[Path]:
    """Collect text files from explicit inputs, directories, and a manifest."""

    paths: list[Path] = []
    for raw in inputs or []:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(candidate for candidate in path.rglob(pattern) if candidate.is_file()))
        else:
            paths.append(path)
    for raw in input_dirs or []:
        root = Path(raw)
        paths.extend(sorted(candidate for candidate in root.rglob(pattern) if candidate.is_file()))
    if manifest is not None:
        manifest_path = Path(manifest)
        base = manifest_path.parent
        for line in manifest_path.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                path = Path(stripped)
                paths.append(path if path.is_absolute() else base / path)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    if max_files is not None:
        deduped = deduped[:max_files]
    if not deduped:
        raise ValueError("no input text files found")
    return deduped


def _write_packed_array(
    output: Path,
    token_array: np.ndarray,
    *,
    metadata: dict[str, object],
    shard_tokens: int | None,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, object]] = []
    if shard_tokens is not None and shard_tokens > 0:
        for shard_idx, start in enumerate(range(0, token_array.shape[0], shard_tokens)):
            shard = token_array[start : start + shard_tokens]
            filename = f"shard_{shard_idx:05d}.npy"
            np.save(output / filename, shard)
            shards.append({"file": filename, "token_count": int(shard.shape[0])})
        storage_format = "sharded_npy"
    else:
        np.save(output / "tokens.npy", token_array)
        storage_format = "single_npy"

    packed_metadata = {
        **metadata,
        "format": storage_format,
        "token_count": int(token_array.shape[0]),
    }
    if shards:
        packed_metadata["shards"] = shards
    (output / "metadata.json").write_text(json.dumps(packed_metadata, indent=2) + "\n")
    return packed_metadata


def _filter_input_paths(
    paths: list[Path],
    *,
    min_bytes: int = 0,
    deduplicate: bool = False,
) -> tuple[list[Path], list[dict[str, str]]]:
    kept: list[Path] = []
    filtered: list[dict[str, str]] = []
    seen_hashes: dict[str, Path] = {}
    for path in paths:
        data = path.read_bytes()
        if len(data) < min_bytes:
            filtered.append({"file": str(path), "reason": "below_min_bytes"})
            continue
        if deduplicate:
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen_hashes:
                filtered.append(
                    {
                        "file": str(path),
                        "reason": "duplicate",
                        "duplicate_of": str(seen_hashes[digest]),
                    }
                )
                continue
            seen_hashes[digest] = path
        kept.append(path)
    return kept, filtered


def _alpha_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(char.isalpha() for char in text) / len(text)


def _load_documents(
    paths: list[Path],
    *,
    input_format: str,
    jsonl_text_key: str,
    min_chars: int,
    min_alpha_fraction: float,
    deduplicate: bool,
) -> tuple[list[str], list[dict[str, object]]]:
    documents: list[str] = []
    filtered: list[dict[str, object]] = []
    seen_hashes: dict[str, str] = {}

    def add_document(text: str, source: str) -> None:
        stripped = text.strip()
        if len(stripped) < min_chars:
            filtered.append({"source": source, "reason": "below_min_chars"})
            return
        if _alpha_fraction(stripped) < min_alpha_fraction:
            filtered.append({"source": source, "reason": "below_min_alpha_fraction"})
            return
        if deduplicate:
            digest = hashlib.sha256(stripped.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                filtered.append(
                    {
                        "source": source,
                        "reason": "duplicate_document",
                        "duplicate_of": seen_hashes[digest],
                    }
                )
                return
            seen_hashes[digest] = source
        documents.append(stripped)

    if input_format == "text":
        for path in paths:
            add_document(path.read_text(encoding="utf-8", errors="replace"), str(path))
    elif input_format == "jsonl":
        for path in paths:
            with path.open(encoding="utf-8") as handle:
                for line_idx, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        row = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        filtered.append(
                            {
                                "source": f"{path}:{line_idx}",
                                "reason": "invalid_json",
                                "error": str(exc),
                            }
                        )
                        continue
                    value = row.get(jsonl_text_key)
                    if not isinstance(value, str):
                        filtered.append(
                            {
                                "source": f"{path}:{line_idx}",
                                "reason": "missing_text_key",
                            }
                        )
                        continue
                    add_document(value, f"{path}:{line_idx}")
    else:
        raise ValueError("input_format must be text or jsonl")

    if not documents:
        raise ValueError("all input documents were filtered out")
    return documents, filtered


def pack_text_files(
    input_paths: list[str | Path],
    output_dir: str | Path,
    *,
    tokenizer: str = "gpt2",
    vocab_size: int | None = None,
    mask_token_id: int | None = None,
    shard_tokens: int | None = None,
    validation_fraction: float = 0.0,
    min_bytes: int = 0,
    deduplicate: bool = False,
    input_format: str = "text",
    jsonl_text_key: str = "text",
    min_chars: int = 0,
    min_alpha_fraction: float = 0.0,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """Pack text files into a contiguous token stream for packed training."""

    paths = [Path(path) for path in input_paths]
    if not paths:
        raise ValueError("at least one input path is required")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    paths, filtered_files = _filter_input_paths(
        paths, min_bytes=min_bytes, deduplicate=deduplicate
    )
    if not paths:
        raise ValueError("all input text files were filtered out")
    if min_alpha_fraction < 0.0 or min_alpha_fraction > 1.0:
        raise ValueError("min_alpha_fraction must be in [0, 1]")
    documents, filtered_documents = _load_documents(
        paths,
        input_format=input_format,
        jsonl_text_key=jsonl_text_key,
        min_chars=min_chars,
        min_alpha_fraction=min_alpha_fraction,
        deduplicate=deduplicate,
    )

    if tokenizer == "gpt2":
        import tiktoken

        encoding = tiktoken.get_encoding("gpt2")
        vocab_size = vocab_size or encoding.n_vocab + 1
        mask_token_id = mask_token_id if mask_token_id is not None else encoding.n_vocab
        if vocab_size < encoding.n_vocab + 1:
            raise ValueError("vocab_size must be at least GPT-2 vocab plus one mask token")
        if mask_token_id < encoding.n_vocab:
            raise ValueError("mask_token_id must not collide with GPT-2 token ids")
        text = "\n\n".join(documents)
        token_ids = encoding.encode(text)
    elif tokenizer == "byte":
        vocab_size = vocab_size or 257
        mask_token_id = mask_token_id if mask_token_id is not None else vocab_size - 1
        if vocab_size <= 2:
            raise ValueError("vocab_size must leave room for clean tokens and mask")
        clean_vocab = vocab_size - 1
        data = "\n\n".join(documents).encode("utf-8")
        token_ids = [byte % clean_vocab for byte in data]
    else:
        raise ValueError("tokenizer must be gpt2 or byte")

    if not token_ids:
        raise ValueError("input files produced no tokens")
    if max(token_ids) >= mask_token_id:
        raise ValueError("clean token ids must be below mask_token_id")
    if validation_fraction < 0.0 or validation_fraction >= 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")

    output = Path(output_dir)
    token_array = np.asarray(token_ids, dtype=np.int64)
    common_metadata = {
        "tokenizer": tokenizer,
        "vocab_size": int(vocab_size),
        "mask_token_id": int(mask_token_id),
        "dtype": str(token_array.dtype),
        "input_files": [str(path) for path in paths],
        "filtered_files": filtered_files,
        "input_format": input_format,
        "jsonl_text_key": jsonl_text_key if input_format == "jsonl" else None,
        "document_count": len(documents),
        "filtered_documents": filtered_documents,
        "min_bytes": min_bytes,
        "min_chars": min_chars,
        "min_alpha_fraction": min_alpha_fraction,
        "deduplicate_exact": deduplicate,
        "provenance": provenance or {},
    }

    if validation_fraction > 0.0:
        validation_tokens = max(1, int(token_array.shape[0] * validation_fraction))
        train_tokens = token_array.shape[0] - validation_tokens
        if train_tokens <= 0:
            raise ValueError("validation split leaves no training tokens")
        train_metadata = _write_packed_array(
            output / "train",
            token_array[:train_tokens],
            metadata={**common_metadata, "split": "train"},
            shard_tokens=shard_tokens,
        )
        validation_metadata = _write_packed_array(
            output / "validation",
            token_array[train_tokens:],
            metadata={**common_metadata, "split": "validation"},
            shard_tokens=shard_tokens,
        )
        metadata = {
            **common_metadata,
            "format": "split_packed",
            "token_count": int(token_array.shape[0]),
            "validation_fraction": validation_fraction,
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
        output.mkdir(parents=True, exist_ok=True)
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata

    metadata = _write_packed_array(
        output,
        token_array,
        metadata=common_metadata,
        shard_tokens=shard_tokens,
    )
    return metadata


def make_text_batcher(
    path: str | Path,
    vocab_size: int,
    mask_token_id: int,
    *,
    tokenizer: str = "byte",
) -> ByteTextBatcher | WordTextBatcher | GPT2TextBatcher | PackedTextBatcher:
    if tokenizer == "byte":
        return ByteTextBatcher(path, vocab_size, mask_token_id)
    if tokenizer == "word":
        return WordTextBatcher(path, vocab_size, mask_token_id)
    if tokenizer == "gpt2":
        return GPT2TextBatcher(path, vocab_size, mask_token_id)
    if tokenizer == "packed":
        return PackedTextBatcher(path, vocab_size, mask_token_id)
    raise ValueError("tokenizer must be byte, word, gpt2, or packed")
