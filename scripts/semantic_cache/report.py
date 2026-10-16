"""Render the checked-in report from ignored benchmark result files."""

from pathlib import Path

from common import DATA, load_json

NAMES = {
    "paws-wiki": "PAWS-Wiki test",
    "paws-x-zh": "PAWS-X Chinese test",
    "gsm8k-hard": "GSM8K number-change negatives",
    "gsm8k-positive": "GSM8K accepted paraphrases",
    "gsm8k-combined": "GSM8K combined",
    "paws-wiki-easy": "PAWS-Wiki easy controls",
    "paws-x-zh-easy": "PAWS-X Chinese easy controls",
    "gsm8k-easy": "GSM8K easy controls",
}
ALIASES = ["MiniLM-L6", "Multilingual-L12"]


def pct(value):
    return "N/A" if value is None else f"{100 * value:.3f}%"


def table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def cell(text):
    return (
        text.replace("|", "\\|")
        .replace("\n", " ")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def point(result, key):
    row = result[key]
    if not result["negatives"]:
        return "N/A (no negatives)", "N/A"
    return (
        (f"{row['threshold']:.2f}", pct(row["hit_rate"])) if row else ("none", "none")
    )


def main():
    data = load_json(DATA / "results.json")
    models = list(data["models"].items())
    env = data["environment"]
    positives = data["paraphrases"]
    sections = ["# Semantic cache false-hit benchmark\n"]
    # Keep the recommendation tied to the evidence rather than a fixed template.
    hard_sets = ("paws-wiki", "paws-x-zh", "gsm8k-combined")
    if any(
        model["datasets"][dataset]["at_01"] is not None
        for _, model in models
        for dataset in hard_sets
    ):
        raise ValueError(
            "Results changed: review and rewrite the recommendation before publishing"
        )
    sections.append("""## Recommendation

**Do not enable semantic caching with either model at any tested threshold (0.80-0.99).**
No threshold meets even the looser 1% false-hit limit on any of the three hard
benchmark families. There is therefore no recommended model/threshold for deployment.
Keep explicit-opt-in exact caching. A high score is insufficient evidence that a
cached answer remains valid after changing names, roles, word order, or a number.
Easy unrelated controls look much better and must not be used to justify deployment.

This is a decision report, not a gateway implementation. No gateway source,
configuration, schema, request flow, or runtime dependency was changed.

## Scope assumption

A prospective semantic cache applies only to **single-turn requests**: an optional
system prompt that must match exactly, followed by exactly one user message.
**Only the user message is embedded.** Whole-conversation embeddings are excluded:
long shared context can dominate a vector, and these small models truncate their
inputs. Equal tenant/model/generation settings and exact system-prompt matching
would be prerequisites, not similarity features. This benchmark scores isolated
text pairs under that assumption; it does not implement a cache or ANN index.
""")
    sections.append(
        "## Environment and model versions\n\n"
        + table(
            ["Setting", "Measured value"],
            [
                ["Run timestamp (UTC)", env["date"]],
                ["OS", env["platform"]],
                ["CPU", f"{env['cpu']}; {env['logical_cpus']} logical CPUs"],
                [
                    "Execution",
                    f"CPU only; {env['dtype']}; {env['torch_threads']} intra-op threads, {env['interop_threads']} inter-op thread",
                ],
                ["Python", env["python"]],
                ["Embedding batch size", env["batch_size"]],
                *[[name, version] for name, version in env["versions"].items()],
            ],
        )
    )
    sections.append(
        table(
            ["Alias", "Model / pinned revision", "Max input", "Dimensions", "License"],
            [
                [
                    alias,
                    f"[{name}](https://huggingface.co/{name}/tree/{model['revision']})<br>`{model['revision']}`",
                    f"{model['max_input_tokens']} tokens",
                    model["embedding_dimension"],
                    "Apache-2.0",
                ]
                for alias, (name, model) in zip(ALIASES, models)
            ],
        )
        + "\nLimits above are the loaded SentenceTransformer configuration, including special tokens, not word limits or the backbone's theoretical maximum. Embeddings are L2-normalized; scores are their dot products (cosine similarity). No fine-tuning, GPU, MPS, quantization, or threshold-dependent filtering was used.\n"
    )
    sections.append("""## Dataset versions, licenses, and construction

**Google LLC is the source of PAWS-Wiki and PAWS-X.** See the
[PAWS repository and citation](https://github.com/google-research-datasets/paws/tree/02b29f3af1143620d1b7f352247e65c0bdd2ec18)
and [PAWS-X documentation](https://github.com/google-research-datasets/paws/blob/02b29f3af1143620d1b7f352247e65c0bdd2ec18/pawsx/README.md).
Their [dataset-specific license](https://github.com/google-research-datasets/paws/blob/02b29f3af1143620d1b7f352247e65c0bdd2ec18/LICENSE)
permits use for any purpose, requests acknowledgement of Google, and disclaims warranty/liability;
it is not labeled CC-BY or Apache here. PAWS: Zhang, Baldridge, and He (NAACL 2019).
PAWS-X: Yang, Zhang, Tar, and Baldridge (EMNLP 2019).

GSM8K is from [OpenAI, Cobbe et al. (2021)](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c),
under the [MIT license, copyright 2021 OpenAI](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE).
The report's GSM8K perturbations and paraphrases are derived from that release.

The original Google Storage archive URLs returned HTTP 403 during this run.
The benchmark uses the pinned test-split Parquet mirrors in the publisher's
`google-research-datasets` Hugging Face namespace instead; no training or validation
rows were substituted. Downloaded files and generated corpora are ignored by Git.
""")
    sections.append(
        table(
            ["Source", "Version / exact downloaded file", "SHA-256"],
            [
                [name, f"[Pinned test split]({source['url']})", f"`{source['sha256']}`"]
                for name, source in data["sources"].items()
                if name != "seed"
            ],
        )
    )
    sections.append(f"""- PAWS-Wiki: `labeled_final`, test, all 8,000 rows. Label 1 is treated as should-share; label 0 as should-not-share, including adversarial negatives.
- PAWS-X: `zh`, test. Of 2,000 mirror rows, 25 contain the documented `NS` missing-translation placeholder; 1,975 remain, matching the publisher's documented Chinese test count.
- GSM8K: original `test.jsonl`, 1,319 questions. Seed `{data["sources"]["seed"]}` shuffles row indexes; digit-bearing questions are eligible. The first 500 receive one randomly selected digit numeral occurrence incremented by 1, preserving all other characters. IDs are zero-based source row indexes. These are synthetic label-0 pairs, not solved or relabeled by a model.
- The next 200 eligible GSM8K questions, disjoint from the 500 hard-negative sources, are paraphrased through the local `/v1/chat/completions` endpoint with `gemini-flash-lite` (`gemini-3.1-flash-lite`, thinking budget 0), `temperature: 0`, `max_tokens: 2048`, and `X-Gateway-Cache: on`. An isolated local gateway with temporary PostgreSQL and separate Redis instances used the existing gateway code. No direct provider calls bypassed the gateway.
- Up to three differently worded instruction attempts are allowed per source. Accept only a nonidentical, nonempty, `stop` response whose multiset of numeric literals and recognized English number words, including multiplicities, is unchanged. Reordered numbers are allowed, changed or deleted numbers are not. The exact prompt and quantity matcher are in the scripts. This is a conservative automated check, not proof of answer equivalence.
- **{positives["requested"]} source questions attempted; {positives["accepted"]} accepted; {positives["requested"] - positives["accepted"]} excluded** after the number/identity checks. There were {positives["attempts"]} successful completion attempts, using {positives["prompt_tokens"]:,} prompt and {positives["completion_tokens"]:,} completion tokens. The positive denominator is {positives["accepted"]}, not 200; rejected outputs are never labeled positive. Saved accepted-output SHA-256: `{positives["sha256"]}`.
- Easy controls: 1,000 random pairs from each corresponding source pool, label 0, reported separately. Exclude identical text and repeated unordered row pairs; screen normalized-character-trigram Jaccard overlap >= 0.2 to avoid accidentally drawing the same PAWS source family. This screen was added after an initial random-only check exposed related pairs; it uses no embedding scores. It excluded {data["sources"]["paws-wiki"]["easy_excluded_overlap"]}, {data["sources"]["paws-x-zh"]["easy_excluded_overlap"]}, and {data["sources"]["gsm8k"]["easy_excluded_overlap"]} candidates for English PAWS, Chinese PAWS-X, and GSM8K respectively. Remaining pairs are assumed unrelated, not manually adjudicated.
- `GSM8K combined` joins the 500 synthetic negatives and {positives["accepted"]} accepted positives so a threshold can have both a false-hit rate and hit rate. The two component cohorts are also reported individually. No easy negatives are mixed into this hard cohort.
""")
    first = models[0][1]["datasets"]
    sections.append(
        table(
            ["Dataset", "Label 1", "Label 0"],
            [
                [NAMES[ds], first[ds]["positives"], first[ds]["negatives"]]
                for ds in NAMES
            ],
        )
    )
    sections.append("""## Metrics and key results

For each threshold t = 0.80, 0.81, ..., 0.99 (inclusive):

- Hit rate = count(label 1 and cosine >= t) / count(label 1).
- False-hit rate = count(label 0 and cosine >= t) / count(label 0).

Comparisons use unrounded scores. Tables round percentages only for display.
`N/A` means the denominator is absent, never zero percent. `none` means no tested
threshold meets the specified false-hit ceiling. A positive-only set cannot
establish a false-hit ceiling. False-hit rate is not 1-precision, overall cache
hit rate, or a production traffic error probability.
""")
    rows = []
    for alias, (_, model) in zip(ALIASES, models):
        for ds in ("paws-wiki", "paws-x-zh", "gsm8k-combined"):
            result = model["datasets"][ds]
            at95 = next(row for row in result["sweep"] if row["threshold"] == 0.95)
            at99 = result["sweep"][-1]
            rows.append(
                [
                    alias,
                    NAMES[ds],
                    pct(at95["hit_rate"]),
                    pct(at95["false_hit_rate"]),
                    f"{at95['false_hits']}/{result['negatives']}",
                    pct(at99["hit_rate"]),
                    pct(at99["false_hit_rate"]),
                ]
            )
    sections.append(
        table(
            [
                "Model",
                "Dataset",
                "Hit @0.95",
                "False hit @0.95",
                "False / negative @0.95",
                "Hit @0.99",
                "False hit @0.99",
            ],
            rows,
        )
    )
    sections.append("### Lowest qualifying threshold, per model and dataset\n")
    sections.append(
        table(
            [
                "Model",
                "Dataset",
                "Threshold: FHR <=0.1%",
                "Hit rate there",
                "Threshold: FHR <=1%",
                "Hit rate there",
            ],
            [
                [
                    alias,
                    NAMES[ds],
                    *point(model["datasets"][ds], "at_001"),
                    *point(model["datasets"][ds], "at_01"),
                ]
                for alias, (_, model) in zip(ALIASES, models)
                for ds in NAMES
            ],
        )
    )
    sections.append("""## CPU embedding latency

Batch size 1; synchronous `encode`, including tokenization, model forward pass,
pooling, normalization, and conversion to NumPy. Model loading/download and cache
lookup are excluded. Each length uses 10 untimed warmups and 100 timed repetitions;
median and NumPy's linear-interpolated p95 are reported. Texts are exactly 20 and
300 whitespace-delimited English words from a fixed repeated prose fixture.
Paraphrase generation had finished before this latency run. These are warm,
single-process microbenchmarks, not concurrent-serving or end-to-end gateway latency.
""")
    sections.append(
        table(
            [
                "Model",
                "Words",
                "Tokens before truncation",
                "Tokens cap",
                "Truncated?",
                "Median ms",
                "p95 ms",
            ],
            [
                [
                    alias,
                    words,
                    row["tokens_before_truncation"],
                    model["max_input_tokens"],
                    "yes" if row["truncated"] else "no",
                    f"{row['median_ms']:.3f}",
                    f"{row['p95_ms']:.3f}",
                ]
                for alias, (_, model) in zip(ALIASES, models)
                for words, row in model["latency"].items()
            ],
        )
    )
    sections.append(
        "The long-input figures reflect truncation at the model's configured limit, **not** embedding all 300 words. Truncation can hide the changed detail and make unsafe pairs indistinguishable. Dataset text-side truncation counts (each side counted separately, repetitions retained):\n"
    )
    sections.append(
        table(
            ["Model", "Dataset", "Truncated sides / all sides"],
            [
                [
                    alias,
                    NAMES[ds],
                    f"{model['datasets'][ds]['truncated_texts']}/{model['datasets'][ds]['total_texts']}",
                ]
                for alias, (_, model) in zip(ALIASES, models)
                for ds in NAMES
                if ds != "gsm8k-combined"
            ],
        )
    )
    sections.append(
        "## Full threshold sweeps\n\nAll hit and false-hit entries below are percentages. Easy controls remain separate.\n"
    )
    for ds in NAMES:
        sections.append("### " + NAMES[ds] + "\n")
        sweeps = [model["datasets"][ds]["sweep"] for _, model in models]
        sections.append(
            table(
                [
                    "Threshold",
                    "MiniLM-L6 hit",
                    "MiniLM-L6 false hit",
                    "Multilingual-L12 hit",
                    "Multilingual-L12 false hit",
                ],
                [
                    [
                        f"{a['threshold']:.2f}",
                        pct(a["hit_rate"]),
                        pct(a["false_hit_rate"]),
                        pct(b["hit_rate"]),
                        pct(b["false_hit_rate"]),
                    ]
                    for a, b in zip(*sweeps)
                ],
            )
        )
    sections.append("""## False-hit examples at threshold 0.95

Up to 10 examples **per model and dataset**, sorted by descending unrounded
similarity (ties by ID). These illustrate failures, not a random sample or manual
label correction. Text is shown in full, including source punctuation. A displayed
score of 1.000000 can reflect rounding or truncation, not necessarily identical text.
PAWS examples credit Google; GSM8K examples derive from OpenAI's MIT-licensed dataset.
""")
    for alias, (_, model) in zip(ALIASES, models):
        for ds in NAMES:
            result = model["datasets"][ds]
            sections.append(f"### {alias}: {NAMES[ds]}\n")
            if ds == "gsm8k-combined":
                sections.append(
                    "Its negative cohort is exactly `GSM8K number-change negatives`; the same 10 examples in that table apply, with identical scores.\n"
                )
            elif not result["negatives"]:
                sections.append(
                    "No label-0 pairs; false-hit examples are undefined for this positive-only cohort.\n"
                )
            elif not result["false_examples"]:
                sections.append("Zero false hits at 0.95; no examples exist.\n")
            else:
                sections.append(
                    f"{result['false_count_095']} false hits total; showing {len(result['false_examples'])}.\n"
                )
                sections.append(
                    table(
                        ["ID", "Cosine", "Original", "Compared text"],
                        [
                            [
                                row["id"],
                                f"{row['similarity']:.6f}",
                                cell(row["a"]),
                                cell(row["b"]),
                            ]
                            for row in result["false_examples"]
                        ],
                    )
                )
    sections.append("""## Limitations and decision boundaries

- PAWS sentences are mostly declarative Wikipedia text, not user questions. Their paraphrase labels are a proxy for answer reuse, not direct cached-answer validation. PAWS-X contains translations and is not an independent naturally occurring Chinese query corpus. The English-focused model is also being evaluated outside its intended language scope.
- Source label noise is visible in the examples: PAWS-Wiki has four exact-string-identical label-0 pairs (IDs 347, 3480, 5410, 7109), and Chinese PAWS-X has two (3203, 7666). Primary metrics retain the supplied labels as requested. Excluding these pairs still gives false-hit rates at 0.99 of 23.565% / 11.956% for MiniLM-L6 and 35.426% / 13.624% for Multilingual-L12 (English / Chinese), so the recommendation is unchanged.
- GSM8K negatives change one numeric occurrence synthetically. A change can be irrelevant to the answer, violate a real-world constraint, or make a problem inconsistent; label 0 is assigned by construction, without solving every modified problem. This is an adversarial sensitivity test, not a measured real-traffic error rate.
- The paraphrase filter validates numeric literals and a documented English number-word vocabulary, not all mathematical meaning, unit associations, or relational bindings. Generated positives are not independently human-adjudicated. Rejecting source questions can bias the retained positive cohort; the measured hit rate applies to the accepted subset only.
- Easy negatives are random controls with a lexical screen. They have a deliberately easier distribution than cache nearest-neighbor candidates and can still contain accidental semantic relations. Their low false-hit rates do not offset failures on hard negatives.
- Evaluation and lowest-threshold selection use the same test pairs. Operating points are descriptive and optimistic; any future deployment needs separate calibration and held-out, traffic-representative validation. The finite grid does not evaluate thresholds above 0.99.
- Point estimates are not guarantees. With 500 negatives, one error is 0.2%; even zero errors would have a roughly 0.6% one-sided 95% upper bound (rule of three), insufficient to establish a 0.1% production target. False hits can increase when searching many cached candidates; pairwise tests do not simulate nearest-neighbor retrieval, cache size, TTL, or changing world knowledge.
- Truncation, warm CPU timings, one machine, no concurrent load, and model/provider version changes limit generalization. Temperature 0 does not guarantee identical paraphrases across provider revisions. Normalized embeddings and all loaded model settings are pinned here; generated outputs remain local and their hash records the measured cohort.

A future investigation would need independently labeled single-turn answer-equivalence
pairs, strict entity/quantity/negation guards, and held-out calibration before revisiting
the decision. This benchmark does not recommend that those additions will suffice.

## Reproduction

Run from the repository root. `bench` is an optional uv dependency group containing
sentence-transformers, torch, and Parquet support; it is not a runtime dependency.
The scripts and this report are committed; `.data/` (downloads, model weights,
paraphrases, vectors, scores, logs, and machine-readable results) is ignored.

```sh
uv sync --group bench
uv run --group bench python scripts/semantic_cache/prepare.py
# Start the existing local gateway and set GATEWAY_API_KEY securely in the environment.
uv run --group bench python scripts/semantic_cache/paraphrase.py --gateway http://127.0.0.1:8000
uv run --group bench python scripts/semantic_cache/benchmark.py
uv run --group bench python scripts/semantic_cache/report.py
```

Preparation and embedding stages reuse local files; paraphrasing checkpoints each
source ID and resumes without repeating successful calls. To regenerate from scratch,
move the ignored `.data` directory aside. Dataset files are pinned and checksummed;
provider outputs can differ on a fresh run. The runtime type check remains `mypy src`;
benchmark mechanics are covered by ordinary tests without model downloads or provider calls.

Validation for this report: `uv run ruff check .`, `uv run ruff format --check .`,
`uv run mypy src`, and `uv run pytest` all passed. The full suite finished with
272 passed, 4 live-provider tests deselected by project configuration, and 4
third-party deprecation warnings. Benchmark-mechanics tests do not require the
optional bench dependencies. All 500 numeric edits were checked for exactly one
changed occurrence; vector norms and independently re-embedded sample pairs were
also verified for both models.
""")
    output = Path("docs/benchmarks/semantic-cache.md")
    output.write_text("\n".join(sections))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
