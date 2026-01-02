"""Mass testing harness for ImageComparatorService.

Usage:
  python scripts/mass_test.py --pairs-file pairs.csv --concurrency 50 --total 1000

pairs.csv format: url1,url2 per line (commas allowed inside URLs if quoted)

Optional flags:
  --prometheus-port PORT  Start a Prometheus metrics endpoint
  --csv-out PATH          Write per-request metrics to CSV

This script runs internal calls (imports `ImageComparatorService`) so it's fast and bypasses HTTP.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import time
import statistics
from typing import List, Tuple, Optional
import sys
from pathlib import Path

try:
    from prometheus_client import start_http_server, Counter, Histogram
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.services.comparator import ImageComparatorService

REQ_COUNTER = None
REQ_ERRORS = None
LATENCY_H = None


def load_pairs_from_file(path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            # take first two columns
            u1 = row[0].strip()
            u2 = row[1].strip() if len(row) > 1 else ""
            if u1 and u2:
                pairs.append((u1, u2))
    return pairs


async def run_one(svc: ImageComparatorService, pair: Tuple[str, str], timeout: float = 30.0):
    start = time.perf_counter()
    try:
        res = await asyncio.wait_for(svc.compare(pair[0], pair[1]), timeout)
        lat = time.perf_counter() - start
        if REQ_COUNTER:
            REQ_COUNTER.inc()
            LATENCY_H.observe(lat)

        # Treat comparator-returned errors as failures
        if isinstance(res, dict) and res.get("error"):
            if REQ_ERRORS:
                REQ_ERRORS.inc()
            return False, lat, str(res.get("error"))

        return True, lat, None
    except Exception as e:
        if REQ_ERRORS:
            REQ_ERRORS.inc()
        return False, None, str(e)


async def worker(svc: ImageComparatorService, queue: asyncio.Queue, results: List[Tuple[bool, Optional[float], Optional[str]]]):
    while True:
        try:
            pair = await queue.get()
        except asyncio.CancelledError:
            break
        ok, lat, err = await run_one(svc, pair)
        results.append((ok, lat, err))
        queue.task_done()


async def main_async(pairs: List[Tuple[str, str]], concurrency: int, total: int, prom_port: Optional[int], csv_out: Optional[str]):
    global REQ_COUNTER, REQ_ERRORS, LATENCY_H

    if prom_port is not None:
        if not PROM_AVAILABLE:
            raise RuntimeError("prometheus_client not installed; install prometheus_client to enable --prometheus-port")
        REQ_COUNTER = Counter("mass_test_requests_total", "Total requests")
        REQ_ERRORS = Counter("mass_test_errors_total", "Total errors")
        LATENCY_H = Histogram("mass_test_request_latency_seconds", "Request latency seconds")
        start_http_server(prom_port)

    svc = ImageComparatorService()
    q: asyncio.Queue = asyncio.Queue()

    if not pairs:
        # placeholder sample pairs - user should provide real ones
        raise RuntimeError("No pairs provided. Use --pairs-file with pairs.csv containing url1,url2 lines.")

    # enqueue pairs repeatedly until total
    for i in range(total):
        q.put_nowait(pairs[i % len(pairs)])

    results: List[Tuple[bool, Optional[float], Optional[str]]] = []

    workers = [asyncio.create_task(worker(svc, q, results)) for _ in range(concurrency)]

    start = time.perf_counter()
    await q.join()
    elapsed = time.perf_counter() - start

    for w in workers:
        w.cancel()
    await svc.close()

    # summarize
    successes = sum(1 for r in results if r[0])
    failures = len(results) - successes
    latencies = [r[1] for r in results if r[0] and r[1] is not None]

    print(f"total requests: {len(results)}")
    print(f"successes: {successes}")
    print(f"failures: {failures}")
    print(f"elapsed(s): {elapsed:.2f}")
    if elapsed > 0:
        print(f"throughput req/s: {len(results)/elapsed:.2f}")

    if latencies:
        lat_sorted = sorted(latencies)
        def pct(p):
            idx = min(len(lat_sorted)-1, max(0, int(len(lat_sorted)*p)-1))
            return lat_sorted[idx]
        print(f"p50: {pct(0.50):.4f}s")
        print(f"p95: {pct(0.95):.4f}s")
        print(f"p99: {pct(0.99):.4f}s")
        print(f"mean: {statistics.mean(latencies):.4f}s")

    if csv_out:
        with open(csv_out, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["success", "latency_seconds", "error"])
            for ok, lat, err in results:
                w.writerow([int(ok), "" if lat is None else f"{lat:.6f}", "" if err is None else err])
        print(f"Wrote CSV to {csv_out}")


def main():
    parser = argparse.ArgumentParser(description="Mass test ImageComparatorService")
    parser.add_argument("--pairs-file", type=str, help="CSV file with url1,url2 per line")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--prometheus-port", type=int, default=None)
    parser.add_argument("--csv-out", type=str, default=None)
    args = parser.parse_args()

    pairs: List[Tuple[str, str]] = []
    if args.pairs_file:
        pairs = load_pairs_from_file(args.pairs_file)

    try:
        asyncio.run(main_async(pairs, args.concurrency, args.total, args.prometheus_port, args.csv_out))
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
