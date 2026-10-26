# Router label data

`questions.jsonl` is the fixed manifest for every model pass: 1,319 GSM8K test
questions and 2,000 MMLU-Pro test questions. It contains 2,422 train, 651 test,
and 246 out-of-distribution (`ood`) questions. All law and history questions are
OOD; other splits depend only on the SHA-256 of the dataset-prefixed question ID.

MMLU-Pro category quotas use `round(2000 * category_size / total_size)` and select
by SHA-256 of ID. Independent rounding is preserved, without adjusting quotas;
for this dataset revision the quotas sum to exactly 2,000. Adding questions never
changes an existing ID's split, although rebuilding can change category samples.
Keep this manifest unchanged while collecting/comparing passes.

## Provenance and licenses

Built on 2026-10-25 with `datasets==5.1.0`:

- [GSM8K, OpenAI](https://github.com/openai/grade-school-math): full test JSONL,
  reused from `data/gsm8k/test.jsonl` via the serving benchmark downloader.
  Source file SHA-256: `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
  [MIT license](https://github.com/openai/grade-school-math/blob/master/LICENSE),
  reproduced below.
- [MMLU-Pro, TIGER Lab](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro):
  `test` split, 12,032 source questions, revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`,
  Hugging Face dataset fingerprint `0072dd0a32d256fc`.
  The [dataset card](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/b189ec765aa7ed75c8acfea42df31fdae71f97be/README.md)
  declares the dataset MIT licensed (distinct from the code repository's license).

Manifest SHA-256: `05aa7c8f7648759db0f179c76c9bd8effedaed3cecba92b21435ad3c317e3901`.
No model labeling was run when creating this manifest.

## Commands

From the repository root, rebuild the manifest only when starting a new dataset:

```sh
uv run --group router python scripts/router_data/build_manifest.py
```

The `router` dependency group is only needed for the manifest builder. The
collector and report use existing dependencies. With `GATEWAY_API_KEY` set, run
one requested model per pass against the running gateway:

```sh
uv run python scripts/router_data/label.py --model local-qwen --per-dataset 50
uv run python scripts/router_data/label.py --model gemini-flash-lite --per-dataset 50
```

Omit `--per-dataset` to continue with the full manifest. Default concurrency is 4;
`--base-url`, `--api-key`, `--concurrency`, and `--out-dir` can be overridden.
Requests never opt into fallback or response caching. A missing or different
`X-Gateway-Model` is an error. HTTP 429 allows 20 retries (honoring `Retry-After`,
default 1 second); connection errors and 5xx share a three-attempt budget with
backoff. Other errors are recorded immediately.

Output is appended and flushed after every question. The last row for each ID
wins; only rows with `error: null` are skipped on resume. Run only one collector
per output model file at a time. The pilot subset is chosen by ID hash before
resume filtering, so resuming a pilot does not select additional questions.

Summary `done` counts newly successful rows, `skipped` counts previous successes,
and `errors` counts failures in this pass. Accuracy and total cost use the latest
successful selected rows, including resumed rows. Costs are decimal USD strings
computed with the current gateway registry prices; missing usage yields a null
cost, never a guessed zero. Latency measures each question's request/retry time,
excluding the wait for a concurrency slot. Request IDs allow correlation with
gateway usage logs. The cost sum excludes failed/unknown usage and is not a bill
for all upstream attempts.

Compare only successful shared IDs, grouped by dataset, MMLU-Pro category, and
split; unknown cost makes that group's total `n/a`:

```sh
uv run python scripts/router_data/report.py \
  docs/benchmarks/router/labels/local-qwen.jsonl \
  docs/benchmarks/router/labels/gemini-flash-lite.jsonl \
  docs/benchmarks/router/questions.jsonl
```

## GSM8K license notice

```text
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
