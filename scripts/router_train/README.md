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

`tfidf_multiclass` fits one multinomial logistic regression to the index of the
first correct model in price order, or the last model if none is correct. It uses
the same nested CV and C grid, scoring multiclass Brier loss. Routing selects the
first model whose cumulative class probability reaches tau. Missing training
classes retain zero probability at their original model indices. Its classifier
metrics are macro one-vs-rest ROC-AUC (null if any class is absent in evaluation)
and mean summed squared error over all classes for Brier score.

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
gap are null when their denominators are zero. The baseline is the upper convex
hull of all single-model cost/accuracy points, linearly interpolated between
adjacent vertices. Equal-cost points retain the highest accuracy. With two distinct
costs this is the original two-model random mixture. `accuracy_vs_hull` subtracts
the interpolated hull accuracy at the selected router cost; it is null outside
the hull's cost range. `oracle_savings_captured` divides router cost reduction by
oracle cost reduction, both relative to the last model; it is null when either
the last-model cost or oracle savings is zero. This ratio is not clipped to [0, 1].
The oracle picks the first correct model in the supplied price order, otherwise
the last. Serving cost/latency omit
feature extraction and classifier overhead, including embedding API cost.

Run offline tests with `uv run --group router pytest tests/test_router_train.py`.
Tests use synthetic data and mocked embeddings, never the real label corpus or
embedding service.

## Portable export and quality levels

When ready to run a real export (this retrains on the train split):

```sh
uv run --group router python scripts/router_train/export.py \
  --models local-qwen gemini-flash-lite --run tfidf-cascade \
  --levels high=0.01 balanced=0.03 economy=0.08
```

The defaults shown allow drops of 1, 3, and 8 percentage points below last-model
train accuracy. Export shares the exact fitting/CV routine with `train.py`; each
level uses only its paired train OOF predictions. Test and OOD predictions are
computed once after thresholds are fixed and reused across levels. The exporter
writes the artifact and `<run>-parity.json` to `src/llm_gateway/router_artifacts/`,
and `<run>-levels.json` to `docs/benchmarks/router/results/`, plus a Markdown table
to stdout. The parity file contains the first 200 shared test questions in manifest
ID order (all available questions for a smaller synthetic/test cohort).

Artifacts contain per-classifier vocabularies, learned IDF/logistic weights,
vectorizer settings, quality thresholds, and run/data metadata. Constant train
targets are stored as exact constant probabilities. A null tau preserves the
explicit all-last policy if no threshold meets the train constraint. JSON artifacts
are included in built packages. No model is exported or activated automatically.

Serving code uses only the standard library. Create one `Router` instance at
startup and reuse it; `route` does no file I/O:

```python
from llm_gateway.router import Router

router = Router("src/llm_gateway/router_artifacts/tfidf-cascade.json")
decision = router.route("How many items remain?", level="balanced")
# decision.model, decision.probabilities (by non-final model), decision.tau
```

The level report includes absolute chosen/last-model latencies in milliseconds
and their ratio, accuracy retained as a ratio, traffic shares, cost reduction,
hull advantage, and oracle savings captured. Ratios with zero denominators are
null. This module does not change gateway request routing.

The default suite checks a committed synthetic sklearn parity fixture to 1e-9,
without importing sklearn or NumPy. Router-group tests exercise export and report
logic using synthetic data only; no real export or embedding request runs.
