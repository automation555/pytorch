"""Collect instruction counts for continuous integration."""
import argparse
import hashlib
import json
import time
from typing import Dict, List, Optional, Union
import uuid

from core.expand import materialize
from definitions.standard import BENCHMARKS
from execution.runner import Runner
from execution.work import WorkOrder


REPEATS = 5
TIMEOUT = 600  # Seconds
RETRIES = 2

VERSION = 0
MD5 = "b984f1ebad546ed3ee612244b8c72e53"


def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=str, default=None)
    parser.add_argument("--subset", action="store_true")
    args = parser.parse_args(argv)

    t0 = int(time.time())
    version = VERSION
    benchmarks = materialize(BENCHMARKS)

    # Useful for local development, since e2e time for the full suite is O(1 hour)
    in_debug_mode = (args.subset or args.destination is None)
    if args.subset:
        version = -1
        benchmarks = benchmarks[:10]

    work_orders = tuple(
        WorkOrder(label, autolabels, timer_args, timeout=TIMEOUT, retries=RETRIES)
        for label, autolabels, timer_args in benchmarks * REPEATS
    )

    keys = tuple({str(work_order): None for work_order in work_orders}.keys())
    md5 = hashlib.md5()
    for key in keys:
        md5.update(key.encode("utf-8"))

    # Fail early, since collection takes a long time.
    if md5.hexdigest() != MD5 and not args.subset:
        msg = f"Expected {MD5}, got {md5.hexdigest()} instead"
        if in_debug_mode:
            print(f"WARNING: {msg}")
        else:
            raise ValueError(msg)

    results = Runner(work_orders).run()

    # TODO: Annotate with TypedDict when 3.8 is the minimum supported verson.
    grouped_results: Dict[str, Dict[str, List[Union[float, int]]]] = {
        key: {"times": [], "counts": []} for key in keys}

    for work_order, r in results.items():
        key = str(work_order)
        grouped_results[key]["times"].extend(r.wall_times)
        grouped_results[key]["counts"].extend(r.instructions)

    final_results = {
        "run_id": str(uuid.uuid4()),
        "version": version,
        "md5": md5.hexdigest(),
        "start_time": t0,
        "end_time": int(time.time()),
        "values": grouped_results,
    }

    if args.destination is None:
        result_str = json.dumps(final_results)
        print(f"{result_str[:30]} ... {result_str[-30:]}\n")
        import pdb
        pdb.set_trace()
    else:
        with open(args.destination, "wt") as f:
            json.dump(final_results, f)
