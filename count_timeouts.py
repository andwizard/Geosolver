#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys


def collect(runs_dir, filename):
    rows = []
    pattern = os.path.join(runs_dir, "*", filename)
    for path in sorted(glob.glob(pattern)):
        method = os.path.basename(os.path.dirname(path))
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            rows.append({"method": method, "error": str(e)})
            continue
        # timeout_count is recorded by run_agent.py under the run's metadata;
        # total task count falls back to the number of tasks actually recorded.
        metadata = data.get("metadata", {})
        total = metadata.get("total_counts") or len(data.get("tasks", [])) or None
        timeout = metadata.get("timeout_count")
        rate = (timeout / total) if (timeout is not None and total) else None
        rows.append({
            "method": method,
            "total_tasks": total,
            "timeout_count": timeout,
            "timeout_rate": rate,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir",
                    default="./Output/runs")
    ap.add_argument("--filename", default="generated_solutions.json")
    ap.add_argument("--csv", metavar="PATH")
    args = ap.parse_args()

    rows = collect(args.runs_dir, args.filename)
    if not rows:
        print(f"Can not find {args.filename} in {args.runs_dir}", file=sys.stderr)
        sys.exit(1)

    header = f"{'Method':<45} {'Total Tasks':>7} {'Timeouts':>7} {'Timeout Rate':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if "error" in r:
            print(f"{r['method']:<45} {'Failed to read: ' + r['error']}")
            continue
        total = r["total_tasks"] if r["total_tasks"] is not None else "?"
        if r["timeout_count"] is None:
            timeout, rate = "N/A", "N/A"
        else:
            timeout = str(r["timeout_count"])
            rate = f"{r['timeout_rate'] * 100:.2f}%"
        print(f"{r['method']:<45} {str(total):>7} {timeout:>7} {rate:>9}")

    valid = [r for r in rows if r.get("timeout_count") is not None and r.get("total_tasks")]
    if valid:
        tot_timeout = sum(r["timeout_count"] for r in valid)
        tot_tasks = sum(r["total_tasks"] for r in valid)
        print("-" * len(header))
        print(f"{'Total (' + str(len(valid)) + ' methods)':<45} "
              f"{tot_tasks:>7} {tot_timeout:>7} "
              f"{tot_timeout / tot_tasks * 100:>8.2f}%")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method", "total_tasks", "timeout_count", "timeout_rate"])
            for r in rows:
                if "error" in r:
                    w.writerow([r["method"], "", "", ""])
                    continue
                rate = "" if r["timeout_rate"] is None else f"{r['timeout_rate']:.4f}"
                w.writerow([r["method"], r["total_tasks"],
                            r["timeout_count"] if r["timeout_count"] is not None else "",
                            rate])
        print(f"\nCSV: {args.csv}")


if __name__ == "__main__":
    main()
