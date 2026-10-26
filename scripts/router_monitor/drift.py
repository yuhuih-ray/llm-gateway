"""Compare router feature distributions with the fixed test-split reference."""

import argparse
import asyncio
import json
import math
import sys
from bisect import bisect_right
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import select

from llm_gateway.config import get_settings
from llm_gateway.db import get_engine, get_sessionmaker
from llm_gateway.models import UsageLog
from llm_gateway.router import Router
from scripts.router_data.common import MANIFEST, read_manifest

PROBABILITY_BINS = [i / 10 for i in range(11)]
OOV_BINS = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
EPSILON = 1e-4


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("Timestamp must include a timezone")
    return parsed.astimezone(UTC)


def proportions(values: list[float], edges: list[float]) -> list[float]:
    if not values:
        raise ValueError("Cannot measure an empty feature")
    counts = [0] * (len(edges) - 1)
    for value in values:
        if not math.isfinite(value) or not edges[0] <= value <= edges[-1]:
            raise ValueError(f"Feature outside bin range: {value}")
        # Left-inclusive bins, with the final bin including 1.0.
        counts[min(bisect_right(edges, value) - 1, len(counts) - 1)] += 1
    return [count / len(values) if count else EPSILON for count in counts]


def psi(reference: list[float], current: list[float], edges: list[float]) -> float:
    return sum(
        (actual - expected) * math.log(actual / expected)
        for expected, actual in zip(
            proportions(reference, edges), proportions(current, edges)
        )
    )


async def current_rows(
    sessions,  # Existing async session factory; injectable for isolated DB tests.
    run: str,
    since: datetime,  # Inclusive start, exclusive end avoids overlapping windows.
    until: datetime,
) -> list[dict]:
    statement = select(UsageLog.router_info).where(
        UsageLog.model_requested == "auto",
        UsageLog.router_info["artifact_run"].astext == run,
        UsageLog.created_at >= since,
        UsageLog.created_at < until,
    )
    async with sessions() as session:
        return list((await session.scalars(statement)).all())


def reference_rows(router: Router, manifest: Path) -> list[dict]:
    rows = []
    for row in read_manifest(manifest):
        if row["split"] == "test":
            decision = router.route(row["question"])
            rows.append(
                dict(oov_ratio=decision.oov_ratio, probabilities=decision.probabilities)
            )
    return rows


def compare(
    reference: list[dict], current: list[dict], models: list[str], minimum: int
) -> dict:
    if len(current) < minimum:
        raise ValueError(f"Insufficient requests: {len(current)}; need {minimum}")
    result = {}
    for feature in ["oov_ratio", *models]:

        def values(rows):
            raw = [
                row.get("oov_ratio")
                if feature == "oov_ratio"
                else row.get("probabilities", {}).get(feature)
                for row in rows
            ]
            return [value for value in raw if value is not None]

        ref, cur = values(reference), values(current)
        # Null OOV (empty tokenization) and pre-monitoring missing values are not zeros.
        if not ref or not cur:
            raise ValueError(
                f"Insufficient observations for {feature}: reference={len(ref)}, current={len(cur)}"
            )
        score = psi(ref, cur, OOV_BINS if feature == "oov_ratio" else PROBABILITY_BINS)
        result[feature] = dict(
            psi=score,
            status="stable" if score < 0.1 else "watch" if score < 0.2 else "drift",
            reference_mean=mean(ref),
            current_mean=mean(cur),
            reference_n=len(ref),
            current_n=len(cur),
        )
    return result


def markdown(features: dict) -> str:
    lines = [
        "| Feature | Reference n | Current n | Reference mean | Current mean | PSI | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, row in features.items():
        lines.append(
            f"| {name} | {row['reference_n']} | {row['current_n']} | {row['reference_mean']:.4f} | {row['current_mean']:.4f} | {row['psi']:.4f} | {row['status']} |"
        )
    return "\n".join(lines)


async def run(args) -> int:
    router = Router(args.artifact)
    if not router.run_name:
        raise ValueError("Artifact run name is missing")
    reference = reference_rows(router, args.manifest)
    try:
        current = await current_rows(
            get_sessionmaker(), router.run_name, args.since, args.until
        )
    finally:
        await get_engine().dispose()
    features = compare(reference, current, list(router.classifiers), args.min_requests)
    report = dict(
        artifact_run=router.run_name,
        since=args.since.isoformat(),
        until=args.until.isoformat(),
        requests=len(current),
        epsilon=EPSILON,
        bins=dict(probabilities=PROBABILITY_BINS, oov_ratio=OOV_BINS),
        features=features,
    )
    print(markdown(features))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return int(any(row["status"] == "drift" for row in features.values()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=get_settings().router_artifact)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--since", type=timestamp, required=True)
    parser.add_argument("--until", type=timestamp, default=datetime.now(UTC))
    parser.add_argument("--min-requests", type=int, default=50)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.min_requests < 1 or args.since >= args.until:
        parser.error("Require positive --min-requests and --since before --until")
    try:
        return asyncio.run(run(args))
    except ValueError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
