#!/usr/bin/env python3
"""Sample Linux process-tree RSS and system host-memory usage."""

from __future__ import annotations

import argparse
import csv
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


STOP_REQUESTED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    return parser.parse_args()


def parse_proc_stat(text: str) -> tuple[int, int]:
    """Return (parent PID, process start ticks) from /proc/<pid>/stat."""
    closing = text.rfind(")")
    if closing < 0:
        raise ValueError("Malformed /proc stat row")
    fields = text[closing + 2 :].split()
    if len(fields) < 20:
        raise ValueError("Incomplete /proc stat row")
    return int(fields[1]), int(fields[19])


def read_process(pid: int, page_size: int) -> tuple[int, int, int] | None:
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        ppid, start_ticks = parse_proc_stat(stat_text)
        statm = Path(f"/proc/{pid}/statm").read_text(encoding="utf-8").split()
        rss_bytes = int(statm[1]) * page_size
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        return None
    return ppid, start_ticks, rss_bytes


def process_table(page_size: int) -> dict[int, tuple[int, int, int]]:
    table = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        record = read_process(pid, page_size)
        if record is not None:
            table[pid] = record
    return table


def descendant_pids(
    table: dict[int, tuple[int, int, int]],
    root_pid: int,
) -> set[int]:
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _, _) in table.items():
            if pid not in descendants and ppid in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def read_system_memory() -> tuple[int, int]:
    values = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            fields = value.split()
            if fields:
                values[key] = int(fields[0]) * 1024
    total = values["MemTotal"]
    available = values["MemAvailable"]
    return total - available, available


def request_stop(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    if not Path("/proc").is_dir():
        raise RuntimeError("Process-tree sampling requires Linux /proc")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    page_size = os.sysconf("SC_PAGE_SIZE")
    root = read_process(args.root_pid, page_size)
    if root is None:
        raise RuntimeError(f"Root PID {args.root_pid} is not running")
    root_start_ticks = root[1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_time_iso",
        "elapsed_seconds",
        "process_count",
        "process_tree_rss_mib",
        "system_memory_used_mib",
        "system_memory_available_mib",
    ]
    started = time.monotonic()
    mib = 1024**2
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        handle.flush()
        while not STOP_REQUESTED:
            table = process_table(page_size)
            current_root = table.get(args.root_pid)
            if current_root is None or current_root[1] != root_start_ticks:
                break
            pids = descendant_pids(table, args.root_pid)
            rss_bytes = sum(table[pid][2] for pid in pids if pid in table)
            system_used, system_available = read_system_memory()
            writer.writerow(
                {
                    "sample_time_iso": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": f"{time.monotonic() - started:.3f}",
                    "process_count": len(pids),
                    "process_tree_rss_mib": f"{rss_bytes / mib:.3f}",
                    "system_memory_used_mib": f"{system_used / mib:.3f}",
                    "system_memory_available_mib": f"{system_available / mib:.3f}",
                }
            )
            handle.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
