# Offline router training

Install the optional tools with `uv sync --group router`. The `bench` and `router`
groups install separately because their Arrow version requirements conflict.
When ready to train:

```sh
uv run --group router python scripts/router_train/train.py \
  --models local-qwen gemini-flash-lite --run first-cascade
```

This command performs training and may make paid embedding calls. It is not part
of gateway startup. It reads the fixed question manifest and the latest successful
labels shared by every listed model. Supply model names in increasing price order;
the last is the fallback. Costs and latencies come from recorded labels, not a
new serving benchmark. No gateway behavior changes.

Only raw `question` text becomes features. The shared runtime module exposes 17
handcrafted features with a stable named order. TF-IDF uses word unigrams and
bigrams; scaling and vocabulary fitting happen inside each training fold. Logistic
regression chooses C from 0.01, 0.1, 1, 10, 100 by five-fold Brier-score CV.
Handcrafted HistGradientBoosting uses fixed defaults and no early stopping.
An outer five-fold split produces paired out-of-fold predictions for threshold
selection, with an inner five-fold C search. Constant targets use a constant
probability; insufficient nonconstant class counts fail explicitly.

Each variant chooses its own threshold using train only. Among thresholds meeting
last-model train accuracy minus one percentage point, maximize the share routed
before the last model; break ties by lower OOF cost, then higher threshold. If no
threshold is feasible (including exact probability-one cases), use all-last mode.
Test and OOD are evaluated separately and never tune any parameters.

Gemini embeddings use `gemini-embedding-001` with the `CLASSIFICATION` task through
`scripts/semantic_cache/embedding.py`, which uses the existing Google SDK. Export
`GEMINI_API_KEY` (or `GOOGLE_API_KEY`) before running. See the
[Gemini embedding API documentation](https://ai.google.dev/gemini-api/docs/embeddings).
The lazy client is not opened when every embedding is cached. Cache files live in
`data/router/embeddings/<sha256(id)>.npy`, with a sidecar fingerprint of question
text, model, and task to invalidate changed inputs. All variants reuse these files.

JSON and PNG outputs go to `docs/benchmarks/router/results/<run>.*`. Local MLflow
tracking uses gitignored `./mlruns`; inspect it with
`MLFLOW_ALLOW_FILE_STORE=true uv run --group router mlflow ui --backend-store-uri ./mlruns`.
The script opts in to MLflow's filesystem store for this offline workflow.
JSON includes the full 0–1 sweep in 0.01 steps,
per-model ROC-AUC/Brier scores, selected-point metrics, and baselines. ROC-AUC is
null for a single-class evaluation split; relative cost reduction and recovered
gap are null when their denominators are zero. Random mix is the analytical
expected interpolation of cheapest and last; the oracle picks the first correct
model in the supplied price order, otherwise the last. Serving cost/latency omit
feature extraction and classifier overhead, including embedding API cost.

Run offline tests with `uv run --group router pytest tests/test_router_train.py`.
Tests use synthetic data and mocked embeddings, never the real label corpus or
embedding service.
