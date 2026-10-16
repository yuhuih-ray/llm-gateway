"""CPU-only normalized embeddings, threshold sweeps, and warm latency."""

import argparse
import importlib.metadata
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch
from common import DATA, MODELS, load_json, metrics, operating_point, save_json, sha256
from sentence_transformers import SentenceTransformer


def cpu_info() -> str:
    if platform.system() == "Darwin":
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    return platform.processor() or platform.machine()


def latency(model: SentenceTransformer, repeats: int) -> dict:
    words = "A customer wants to compare two simple plans and understand their total costs before choosing the best option for tomorrow".split()
    results = {}
    for count in (20, 300):
        text = " ".join((words * 20)[:count])
        assert len(text.split()) == count
        token_count = len(model.tokenizer.encode(text, truncation=False))
        for _ in range(10):
            model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            model.encode([text], normalize_embeddings=True, show_progress_bar=False)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
        results[str(count)] = dict(
            median_ms=float(np.median(samples)),
            p95_ms=float(np.percentile(samples, 95)),
            tokens_before_truncation=token_count,
            truncated=token_count > model.max_seq_length,
        )
    return results


def encode(
    model,  # Loaded pinned SentenceTransformer in CPU evaluation mode.
    name: str,  # Model identity for local vector checkpoints.
    texts: list[str],  # Deduplicated input texts in stable order.
    batch_size: int,  # Throughput batch size, separate from latency measurements.
):
    stem = DATA / name.split("/")[-1]
    index_path = stem.with_suffix(".texts.json")
    vector_path = stem.with_suffix(".npy")
    old_texts = load_json(index_path) if index_path.exists() else []
    old_vectors = np.load(vector_path) if vector_path.exists() else None
    known = set(old_texts)
    missing = [text for text in texts if text not in known]
    if missing:
        print(f"{name}: embedding {len(missing)} new texts on CPU", flush=True)
        new_vectors = model.encode(
            missing,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        vectors = (
            np.concatenate([old_vectors, new_vectors])
            if old_vectors is not None
            else new_vectors
        )
        old_texts.extend(missing)
        np.save(vector_path, vectors)
        save_json(index_path, old_texts)
    else:
        vectors = old_vectors
    return {text: vector for text, vector in zip(old_texts, vectors)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Precompute base embeddings while paraphrases are generated",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latency-repeats", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    datasets = load_json(DATA / "datasets.json")
    if not args.base_only:
        datasets["gsm8k-positive"] = load_json(DATA / "positives.json")
        if not datasets["gsm8k-positive"]:
            raise ValueError("No accepted paraphrases")
        datasets["gsm8k-combined"] = datasets["gsm8k-hard"] + datasets["gsm8k-positive"]
    texts = list(
        dict.fromkeys(
            text
            for rows in datasets.values()
            for row in rows
            for text in (row["a"], row["b"])
        )
    )
    report = dict(
        environment=dict(
            date=datetime.now(timezone.utc).isoformat(),
            platform=platform.platform(),
            python=platform.python_version(),
            cpu=cpu_info(),
            logical_cpus=os.cpu_count(),
            torch_threads=args.threads,
            interop_threads=1,
            device="cpu",
            dtype="float32",
            batch_size=args.batch_size,
            latency_repeats=args.latency_repeats,
            versions={
                name: importlib.metadata.version(name)
                for name in (
                    "sentence-transformers",
                    "torch",
                    "transformers",
                    "numpy",
                    "pyarrow",
                )
            },
        ),
        models={},
    )
    for name, revision in MODELS.items():
        print(f"Loading {name}@{revision}", flush=True)
        model = SentenceTransformer(
            name, revision=revision, device="cpu", cache_folder=str(DATA / "models")
        )
        model.eval()
        vectors = encode(model, name, texts, args.batch_size)
        results = {}
        for dataset, rows in datasets.items():
            scores = [
                float(np.dot(vectors[row["a"]], vectors[row["b"]])) for row in rows
            ]
            labels = [row["label"] for row in rows]
            sweep = metrics(labels, scores)
            false_indexes = sorted(
                (
                    i
                    for i, (label, score) in enumerate(zip(labels, scores))
                    if label == 0 and score >= 0.95
                ),
                key=lambda i: (-scores[i], rows[i]["id"]),
            )
            token_lengths = model.tokenizer(
                [text for row in rows for text in (row["a"], row["b"])],
                truncation=False,
                padding=False,
                return_length=True,
            )["length"]
            results[dataset] = dict(
                positives=sum(labels),
                negatives=len(labels) - sum(labels),
                sweep=sweep,
                at_001=operating_point(sweep, 0.001),
                at_01=operating_point(sweep, 0.01),
                false_examples=[
                    {**rows[i], "similarity": scores[i]} for i in false_indexes[:10]
                ],
                false_count_095=len(false_indexes),
                truncated_texts=sum(
                    length > model.max_seq_length for length in token_lengths
                ),
                total_texts=len(token_lengths),
            )
            save_json(
                DATA / (name.split("/")[-1] + "." + dataset + ".scores.json"),
                [
                    dict(id=row["id"], label=row["label"], similarity=score)
                    for row, score in zip(rows, scores)
                ],
            )
        report["models"][name] = dict(
            revision=revision,
            max_input_tokens=model.max_seq_length,
            embedding_dimension=model.get_sentence_embedding_dimension(),
            datasets=results,
            latency={} if args.base_only else latency(model, args.latency_repeats),
        )
    report["sources"] = load_json(DATA / "sources.json")
    if not args.base_only:
        report["paraphrases"] = load_json(DATA / "paraphrase_summary.json")
        report["paraphrases"]["sha256"] = sha256(DATA / "positives.json")
    save_json(
        DATA / ("base_results.json" if args.base_only else "results.json"), report
    )
    print("Benchmark complete", flush=True)


if __name__ == "__main__":
    main()
