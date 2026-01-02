from __future__ import annotations

import asyncio
import time
import statistics
from typing import List, Tuple, Optional, Dict, Any

from app.services.comparator_service import ImageComparatorService


class MassTestService:
    """Runs mass test using an ImageComparatorService instance.

    Methods are async and return a dictionary with aggregate metrics.
    """

    def __init__(self, comparator: ImageComparatorService):
        self._comparator = comparator

    async def run(
        self,
        pairs: List[Tuple[str, str]],
        concurrency: int = 10,
        total: Optional[int] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        if not pairs:
            raise ValueError("pairs must not be empty")

        total = total or len(pairs)

        q: asyncio.Queue = asyncio.Queue()
        for i in range(total):
            q.put_nowait(pairs[i % len(pairs)])

        results: List[Tuple[bool, Optional[float]]] = []

        async def worker() -> None:
            while True:
                try:
                    pair = await q.get()
                except asyncio.CancelledError:
                    break
                start = time.perf_counter()
                try:
                    res = await asyncio.wait_for(self._comparator.compare(pair[0], pair[1]), timeout)
                except Exception:
                    results.append((False, None))
                    q.task_done()
                    continue
                lat = time.perf_counter() - start
                # treat comparator error as failure
                if isinstance(res, dict) and res.get("error"):
                    results.append((False, lat))
                else:
                    results.append((True, lat))
                q.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(max(1, concurrency))]

        start = time.perf_counter()
        await q.join()
        elapsed = time.perf_counter() - start

        for w in workers:
            w.cancel()

        successes = sum(1 for r in results if r[0])
        failures = len(results) - successes
        latencies = [r[1] for r in results if r[0] and r[1] is not None]

        metrics: Dict[str, Any] = {
            "total": len(results),
            "successes": successes,
            "failures": failures,
            "elapsed_seconds": elapsed,
            "throughput_rps": (len(results) / elapsed) if elapsed > 0 else 0.0,
            "p50": None,
            "p95": None,
            "p99": None,
            "mean": None,
        }

        if latencies:
            lat_sorted = sorted(latencies)
            def pct(p: float) -> float:
                idx = min(len(lat_sorted) - 1, max(0, int(len(lat_sorted) * p) - 1))
                return lat_sorted[idx]

            metrics.update(
                {
                    "p50": pct(0.50),
                    "p95": pct(0.95),
                    "p99": pct(0.99),
                    "mean": statistics.mean(latencies),
                }
            )

        return {"metrics": metrics, "sample_latencies": latencies[:200]}
