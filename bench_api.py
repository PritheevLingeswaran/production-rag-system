import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import httpx

try:
    import psutil  # optional
except ImportError:
    psutil = None


@dataclass
class Result:
    ok: bool
    status_code: Optional[int]
    latency_ms: float
    error: Optional[str]


def percentile(arr: List[float], p: float) -> float:
    if not arr:
        return float("nan")
    return float(np.percentile(np.array(arr, dtype=float), p))


async def worker(
    client: httpx.AsyncClient,
    url: str,
    method: str,
    payload: Optional[Dict[str, Any]],
    headers: Dict[str, str],
    timeout_s: float,
    total_requests: int,
    counter: Dict[str, int],
    lock: asyncio.Lock,
    results: List[Result],
):
    while True:
        async with lock:
            if counter["sent"] >= total_requests:
                return
            counter["sent"] += 1

        start = time.perf_counter()
        try:
            if method == "GET":
                r = await client.get(url, headers=headers, timeout=timeout_s)
            else:
                r = await client.request(method, url, json=payload, headers=headers, timeout=timeout_s)

            latency_ms = (time.perf_counter() - start) * 1000.0
            ok = 200 <= r.status_code < 300
            results.append(Result(ok=ok, status_code=r.status_code, latency_ms=latency_ms,
                                  error=None if ok else f"HTTP_{r.status_code}"))
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000.0
            results.append(Result(ok=False, status_code=None, latency_ms=latency_ms, error=str(e)))


async def run_benchmark(
    url: str,
    method: str,
    payload: Optional[Dict[str, Any]],
    headers: Dict[str, str],
    concurrency: int,
    total_requests: int,
    timeout_s: float,
    warmup: int,
) -> Tuple[List[Result], float, Optional[Dict[str, float]]]:
    # optional system stats
    proc = psutil.Process() if psutil else None
    cpu_start = None
    rss_start = None
    if proc:
        cpu_start = proc.cpu_times()
        rss_start = proc.memory_info().rss

    @asynccontextmanager
    async def _client_ctx():
        try:
            async with httpx.AsyncClient(http2=True) as c:
                yield c
        except ImportError:
            # h2 extra is optional; gracefully fallback to HTTP/1.1.
            async with httpx.AsyncClient(http2=False) as c:
                yield c

    async with _client_ctx() as client:
        # warmup
        if warmup > 0:
            for _ in range(warmup):
                try:
                    if method == "GET":
                        await client.get(url, headers=headers, timeout=timeout_s)
                    else:
                        await client.request(method, url, json=payload, headers=headers, timeout=timeout_s)
                except Exception:
                    pass

        counter = {"sent": 0}
        lock = asyncio.Lock()
        results: List[Result] = []

        start_wall = time.perf_counter()
        tasks = [
            asyncio.create_task(
                worker(client, url, method, payload, headers, timeout_s, total_requests, counter, lock, results)
            )
            for _ in range(concurrency)
        ]
        await asyncio.gather(*tasks)
        end_wall = time.perf_counter()

    duration_s = max(1e-9, end_wall - start_wall)

    sys_stats = None
    if proc and cpu_start and rss_start is not None:
        cpu_end = proc.cpu_times()
        rss_end = proc.memory_info().rss
        # approximate CPU seconds used during run
        cpu_used_s = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)
        sys_stats = {
            "cpu_used_seconds": float(cpu_used_s),
            "rss_start_mb": float(rss_start / (1024 * 1024)),
            "rss_end_mb": float(rss_end / (1024 * 1024)),
            "rss_delta_mb": float((rss_end - rss_start) / (1024 * 1024)),
        }

    return results, duration_s, sys_stats


def summarize(results: List[Result], duration_s: float) -> Dict[str, Any]:
    total = len(results)
    oks = [r for r in results if r.ok]
    fails = [r for r in results if not r.ok]

    lat_all = [r.latency_ms for r in results]
    lat_ok = [r.latency_ms for r in oks]

    rps = total / duration_s if duration_s > 0 else float("nan")
    err_rate = (len(fails) / total * 100.0) if total > 0 else float("nan")

    summary = {
        "total_requests": total,
        "success": len(oks),
        "fail": len(fails),
        "error_rate_percent": round(err_rate, 3),
        "duration_seconds": round(duration_s, 3),
        "throughput_rps": round(rps, 3),

        # latencies (all)
        "latency_ms_avg_all": round(float(np.mean(lat_all)) if lat_all else float("nan"), 3),
        "latency_ms_p50_all": round(percentile(lat_all, 50), 3),
        "latency_ms_p95_all": round(percentile(lat_all, 95), 3),
        "latency_ms_p99_all": round(percentile(lat_all, 99), 3),

        # latencies (success-only) - usually what you should put on resume
        "latency_ms_avg_ok": round(float(np.mean(lat_ok)) if lat_ok else float("nan"), 3),
        "latency_ms_p50_ok": round(percentile(lat_ok, 50), 3),
        "latency_ms_p95_ok": round(percentile(lat_ok, 95), 3),
        "latency_ms_p99_ok": round(percentile(lat_ok, 99), 3),
    }

    # show top errors
    if fails:
        err_counts: Dict[str, int] = {}
        for f in fails:
            key = f.error or "unknown"
            err_counts[key] = err_counts.get(key, 0) + 1
        summary["top_errors"] = sorted(err_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    else:
        summary["top_errors"] = []

    return summary


def main():
    ap = argparse.ArgumentParser(description="Benchmark an HTTP API and output resume-ready metrics.")
    ap.add_argument("--url", required=True, help="Endpoint URL, e.g., http://127.0.0.1:8000/query")
    ap.add_argument("--method", default="POST", help="GET/POST/PUT/etc (default POST)")
    ap.add_argument("--payload", default=None, help="JSON string payload for non-GET methods")
    ap.add_argument("--header", action="append", default=[], help="Extra headers like 'Authorization: Bearer xxx'")
    ap.add_argument("--concurrency", type=int, default=50, help="Concurrent workers (default 50)")
    ap.add_argument("--requests", type=int, default=1000, help="Total requests (default 1000)")
    ap.add_argument("--timeout", type=float, default=10.0, help="Request timeout seconds (default 10)")
    ap.add_argument("--warmup", type=int, default=20, help="Warmup requests (default 20)")
    ap.add_argument("--out", default="bench_results.json", help="Output JSON file (default bench_results.json)")
    args = ap.parse_args()

    method = args.method.upper().strip()
    payload = None
    if method != "GET" and args.payload:
        payload = json.loads(args.payload)

    headers: Dict[str, str] = {}
    for h in args.header:
        if ":" not in h:
            raise ValueError(f"Bad header format: {h} (use 'Key: Value')")
        k, v = h.split(":", 1)
        headers[k.strip()] = v.strip()

    results, duration_s, sys_stats = asyncio.run(
        run_benchmark(
            url=args.url,
            method=method,
            payload=payload,
            headers=headers,
            concurrency=args.concurrency,
            total_requests=args.requests,
            timeout_s=args.timeout,
            warmup=args.warmup,
        )
    )

    summary = summarize(results, duration_s)
    output = {
        "summary": summary,
        "system_stats": sys_stats,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": {
            "resume_tip": "Use latency_ms_p95_ok, throughput_rps, error_rate_percent, dataset/doc count separately."
        },
    }

    print("\n=== RESUME-READY METRICS ===")
    print(f"Throughput (RPS): {summary['throughput_rps']}")
    print(f"Error rate (%):   {summary['error_rate_percent']}")
    print(f"Latency avg (ms): {summary['latency_ms_avg_ok']}  (success-only)")
    print(f"Latency p95 (ms): {summary['latency_ms_p95_ok']}  (success-only)")
    print(f"Latency p99 (ms): {summary['latency_ms_p99_ok']}  (success-only)")
    if sys_stats:
        print("\n=== SYSTEM STATS (optional) ===")
        print(f"CPU used (s):     {sys_stats['cpu_used_seconds']}")
        print(f"RAM start (MB):   {sys_stats['rss_start_mb']:.2f}")
        print(f"RAM end (MB):     {sys_stats['rss_end_mb']:.2f}")
        print(f"RAM delta (MB):   {sys_stats['rss_delta_mb']:.2f}")

    if summary["top_errors"]:
        print("\nTop errors:", summary["top_errors"])

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved full results to: {args.out}")


if __name__ == "__main__":
    main()
