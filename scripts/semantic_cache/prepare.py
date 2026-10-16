"""Download original test releases and construct deterministic benchmark pairs."""

import json
import random
import re
import urllib.request

import pyarrow.parquet as pq
from common import (
    DATA,
    NUMERAL,
    SEED,
    lexical_overlap,
    perturb_number,
    save_json,
    sha256,
)

SOURCES = {
    "paws-wiki": "https://huggingface.co/datasets/google-research-datasets/paws/resolve/161ece9501cf0a11f3e48bd356eaa82de46d6a09/labeled_final/test-00000-of-00001.parquet",
    "paws-x-zh": "https://huggingface.co/datasets/google-research-datasets/paws-x/resolve/4cd8187c404bda33cb1f62b49b001115862acf37/zh/test-00000-of-00001.parquet",
    "gsm8k": "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl",
}


EXPECTED_SHA256 = {
    "paws-wiki": "ae342ff12bb84b84b95f468abf5db6cb7c7bd578271299fe9c99be75b8132f4d",
    "paws-x-zh": "cdae3237c12a49165aacf489d3ab60b832b7672e7775abf937a1495f3ab75c80",
    "gsm8k": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
}


def download(
    name: str,  # Dataset identity with an expected content checksum.
    url: str,  # Immutable publisher mirror revision or original commit.
):
    path = DATA / (name + (".jsonl" if name == "gsm8k" else ".parquet"))
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    if sha256(path) != EXPECTED_SHA256[name]:
        raise ValueError(f"Dataset checksum mismatch: {name}")
    return path


def main():
    DATA.mkdir(exist_ok=True)
    paths = {name: download(name, url) for name, url in SOURCES.items()}
    manifest = {
        name: dict(url=SOURCES[name], sha256=sha256(path))
        for name, path in paths.items()
    }
    datasets, pools = {}, {}
    for name in ("paws-wiki", "paws-x-zh"):
        rows = pq.read_table(paths[name]).to_pylist()
        valid = [
            row
            for row in rows
            if all(
                row[field].strip() not in {"", "NS"}
                for field in ("sentence1", "sentence2")
            )
        ]
        manifest[name].update(
            raw_rows=len(rows), excluded_missing=len(rows) - len(valid), rows=len(valid)
        )
        datasets[name] = [
            dict(
                id=row["id"],
                a=row["sentence1"],
                b=row["sentence2"],
                label=int(row["label"]),
            )
            for row in valid
        ]
        pools[name] = [row["sentence1"] for row in valid]
    gsm = [json.loads(line) for line in paths["gsm8k"].read_text().splitlines()]
    manifest["gsm8k"]["rows"] = len(gsm)
    indexes = list(range(len(gsm)))
    rng = random.Random(SEED)
    rng.shuffle(indexes)
    eligible = [i for i in indexes if re.search(NUMERAL, gsm[i]["question"])]
    hard = []
    for i in eligible[:500]:
        original = gsm[i]["question"]
        changed, edit = perturb_number(original, rng)
        hard.append(dict(id=str(i), a=original, b=changed, label=0, edit=edit))
    assert len(hard) == 500
    datasets["gsm8k-hard"] = hard
    # Disjoint source questions avoid counting the same question in both cohorts.
    positive_sources = [
        dict(id=str(i), question=gsm[i]["question"]) for i in eligible[500:700]
    ]
    assert len(positive_sources) == 200
    save_json(DATA / "positive_sources.json", positive_sources)
    pools["gsm8k"] = [row["question"] for row in gsm]
    for name, pool in pools.items():
        # Random different source rows, excluding identical text. These are assumed
        # unrelated, not human-verified answer-equivalence labels.
        pairs, used = [], set()
        excluded_overlap = 0
        while len(pairs) < 1000:
            i, j = rng.sample(range(len(pool)), 2)
            if pool[i] == pool[j] or tuple(sorted((i, j))) in used:
                continue
            if lexical_overlap(pool[i], pool[j]) >= 0.2:
                excluded_overlap += 1
                continue
            used.add(tuple(sorted((i, j))))
            pairs.append(dict(id=f"{i}:{j}", a=pool[i], b=pool[j], label=0))
        datasets[name + "-easy"] = pairs
        manifest[name]["easy_excluded_overlap"] = excluded_overlap
    save_json(DATA / "datasets.json", datasets)
    manifest["seed"] = SEED
    save_json(DATA / "sources.json", manifest)
    print(json.dumps({name: len(rows) for name, rows in datasets.items()}))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
