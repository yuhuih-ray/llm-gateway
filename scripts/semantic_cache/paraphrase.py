"""Generate resumable GSM8K positives through an authenticated local gateway."""

import argparse
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from common import DATA, load_json, save_json, valid_paraphrase

SYSTEM = (
    "Paraphrase the supplied math question without solving it. Preserve every fact, "
    "relationship, unit, name, and the question being asked. Preserve ALL numbers "
    "exactly as written, including digits, spelled-out numbers, fractions, and their "
    "number of occurrences. Do not convert words to digits or digits to words. "
    "Reword the prose meaningfully. Return only the paraphrased question."
)


def generate(
    source: dict,  # Pinned GSM8K row ID and original question.
    url: str,  # Local authenticated gateway, never a direct provider endpoint.
    key: str,  # Bearer secret read from the environment; never saved or logged.
) -> dict:
    path = DATA / "paraphrases" / (source["id"] + ".json")
    if path.exists():
        return load_json(path)
    attempts = []
    for attempt in range(3):
        instruction = SYSTEM + (
            " Be especially careful to keep the exact numeric expressions and number words."
            * attempt
        )
        payload = dict(
            model="gemini-flash-lite",
            temperature=0,
            max_tokens=2048,
            messages=[
                dict(role="system", content=instruction),
                dict(role="user", content=source["question"]),
            ],
        )
        request = urllib.request.Request(
            url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + key,
                "X-Gateway-Cache": "on",
            },
        )
        for retry in range(6):
            try:
                with urllib.request.urlopen(request, timeout=150) as response:
                    body = json.load(response)
                    cache_status = response.headers.get("X-Cache")
                    request_id = response.headers.get("X-Request-ID")
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 502, 503, 504} or retry == 5:
                    raise RuntimeError(f"Gateway HTTP status {exc.code}") from None
                time.sleep(
                    max(float(exc.headers.get("Retry-After", 0)), min(30, 2**retry))
                )
        candidate = body["choices"][0]["message"]["content"].strip()
        valid = body["choices"][0]["finish_reason"] == "stop" and valid_paraphrase(
            source["question"], candidate
        )
        attempts.append(
            dict(
                candidate=candidate,
                valid=valid,
                usage=body["usage"],
                cache_status=cache_status,
                request_id=request_id,
            )
        )
        if valid:
            break
    result = dict(
        id=source["id"],
        a=source["question"],
        b=candidate,
        label=1,
        accepted=valid,
        attempts=attempts,
    )
    save_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", default="http://127.0.0.1:8000")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if urllib.parse.urlparse(args.gateway).hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        parser.error(
            "Use the local gateway; do not send its credentials to remote hosts"
        )
    key = os.environ["GATEWAY_API_KEY"]
    sources = load_json(DATA / "positive_sources.json")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(generate, source, args.gateway.rstrip("/"), key)
            for source in sources
        ]
        rows = []
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
            print(
                f"Paraphrases complete: {len(rows)}/{len(sources)}; accepted: {sum(row['accepted'] for row in rows)}",
                flush=True,
            )
    rows.sort(key=lambda row: int(row["id"]))
    save_json(DATA / "positives.json", [row for row in rows if row["accepted"]])
    save_json(
        DATA / "paraphrase_summary.json",
        dict(
            model="gemini-flash-lite",
            temperature=0,
            cache_opt_in=True,
            system_prompt=SYSTEM,
            requested=len(sources),
            accepted=sum(row["accepted"] for row in rows),
            attempts=sum(len(row["attempts"]) for row in rows),
            prompt_tokens=sum(
                attempt["usage"]["prompt_tokens"] or 0
                for row in rows
                for attempt in row["attempts"]
            ),
            completion_tokens=sum(
                attempt["usage"]["completion_tokens"] or 0
                for row in rows
                for attempt in row["attempts"]
            ),
        ),
    )


if __name__ == "__main__":
    main()
