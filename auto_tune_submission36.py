import argparse
import copy
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


FINISH_RE = re.compile(r"36-section direct finished in (?P<time>[0-9.]+) seconds")
CRASH_RE = re.compile(r"major collision of intensity (?P<intensity>[0-9.]+)")
FAILED_RE = re.compile(r"36-section direct failed to finish")
SECTION_RE = re.compile(
    r"\[section36\] t=(?P<t>\d+) sec=(?P<section>-?\d+) "
    r"speed=(?P<speed>-?[0-9.]+).*wp=(?P<wp>-?\d+)"
)


GROUPS: Dict[str, List[int]] = {
    "A": [6, 7, 8],          # early braking / recovery
    "B": [12, 13, 14],       # original section 3 special steering
    "C": [16, 17, 18],       # original section 4 heavy braking
    "D": [23, 24],           # high-speed into turn
    "E": [27, 28, 29],       # long fast approach
    "F": [30, 31, 32, 33, 34, 35],  # final technical section
}


PARAM_DEFAULTS = {
    "friction_mus": 2.75,
    "lookahead_scales": 1.0,
    "steer_gain_scales": 1.0,
    "brake_distance_scales": 1.0,
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return raw


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def absolutize_bundle_paths(bundle: Dict[str, Any], source_path: Path) -> None:
    globals_map = bundle.get("globals")
    if not isinstance(globals_map, dict):
        return
    raw = globals_map.get("competition_section_indices_json")
    if not raw:
        return

    raw_path = Path(str(raw))
    if raw_path.is_absolute():
        return

    candidates = [
        raw_path,
        source_path.parent / raw_path,
        source_path.resolve().parents[1] / raw_path,
        source_path.resolve().parents[1] / "data" / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            globals_map["competition_section_indices_json"] = str(candidate.resolve())
            return


def parse_groups(raw: str) -> List[int]:
    sections: List[int] = []
    seen = set()
    for item in raw.split(","):
        item = item.strip().upper()
        if not item:
            continue
        if item in GROUPS:
            values = GROUPS[item]
        else:
            values = [int(item)]
        for value in values:
            if value not in seen:
                sections.append(value)
                seen.add(value)
    return sections


def parse_params(raw: str) -> List[str]:
    allowed = set(PARAM_DEFAULTS)
    out: List[str] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(f"Unsupported param {item!r}; choose from {sorted(allowed)}")
        out.append(item)
    return out


def candidate_deltas(param: str, args: argparse.Namespace) -> List[float]:
    if param == "friction_mus":
        base = float(args.mu_step)
        vals = [base, -base]
        if args.aggressive:
            vals.extend([2.0 * base, -2.0 * base])
        return vals
    if param == "lookahead_scales":
        base = float(args.lookahead_step)
        return [base, -base] if not args.prefer_smoothing else [base, -base, 2.0 * base]
    if param == "steer_gain_scales":
        base = float(args.steer_step)
        # Try lower gain first because it often reduces scrub/oscillation.
        return [-base, base]
    if param == "brake_distance_scales":
        base = float(args.brake_distance_step)
        # Try earlier braking first; it is safer when repairing unstable sections.
        vals = [base, -base]
        if args.aggressive:
            vals.extend([2.0 * base, -2.0 * base])
        return vals
    raise ValueError(param)


def clamp_param(param: str, value: float) -> float:
    if param == "friction_mus":
        return round(max(0.5, min(5.0, value)), 3)
    if param in ("lookahead_scales", "steer_gain_scales"):
        return round(max(0.6, min(1.5, value)), 3)
    if param == "brake_distance_scales":
        return round(max(0.7, min(1.5, value)), 3)
    return round(value, 3)


def set_param(bundle: Dict[str, Any], param: str, section: int, value: float) -> None:
    values = bundle.setdefault(param, {})
    if not isinstance(values, dict):
        values = {}
        bundle[param] = values
    values[str(int(section))] = float(value)


def get_param(bundle: Dict[str, Any], param: str, section: int) -> float:
    values = bundle.get(param, {})
    if isinstance(values, dict) and str(int(section)) in values:
        return float(values[str(int(section))])
    return float(PARAM_DEFAULTS[param])


def parse_run_output(text: str) -> Dict[str, Any]:
    finish = FINISH_RE.search(text)
    crash = CRASH_RE.search(text)
    last_section: Optional[Dict[str, Any]] = None
    for match in SECTION_RE.finditer(text):
        last_section = {
            "t": int(match.group("t")),
            "section": int(match.group("section")),
            "speed": float(match.group("speed")),
            "wp": int(match.group("wp")),
        }
    return {
        "finished": finish is not None,
        "time": float(finish.group("time")) if finish else None,
        "crashed": crash is not None,
        "collision_intensity": float(crash.group("intensity")) if crash else None,
        "last_section": last_section,
    }


def run_bundle(bundle_path: Path, log_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    script_dir = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        "-u",
        str(script_dir / "run_36sections.py"),
        "--bundle",
        str(bundle_path),
        "--max-seconds",
        str(float(args.max_seconds)),
    ]
    if args.no_rendering:
        cmd.append("--no-rendering")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_parts: List[str] = []
    timed_out = False
    fail_seen_at: Optional[float] = None

    def emit(line: str, log_file: Any) -> None:
        output_parts.append(line)
        print(line, end="", flush=True)
        log_file.write(line)
        log_file.flush()

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        emit(f"[auto-tune] command: {' '.join(cmd)}\n", log_file)
        proc = subprocess.Popen(
            cmd,
            cwd=str(script_dir),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )

        lines: "queue.Queue[str]" = queue.Queue()

        def read_stdout() -> None:
            assert proc.stdout is not None
            for line in iter(proc.stdout.readline, ""):
                lines.put(line)

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()
        started_at = time.monotonic()

        while True:
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                line = None
            if line is not None:
                emit(line, log_file)
                if FAILED_RE.search(line):
                    fail_seen_at = time.monotonic()
                elif CRASH_RE.search(line) and fail_seen_at is None:
                    fail_seen_at = time.monotonic()

            if proc.poll() is not None and lines.empty():
                break
            if fail_seen_at is not None and time.monotonic() - fail_seen_at > float(args.failure_grace_seconds):
                emit(
                    f"\n[auto-tune] failed candidate detected; stopping child after "
                    f"{args.failure_grace_seconds} seconds\n",
                    log_file,
                )
                proc.terminate()
                break
            if time.monotonic() - started_at > float(args.timeout_seconds):
                timed_out = True
                emit(f"\n[auto-tune] run timed out after {args.timeout_seconds} seconds\n", log_file)
                proc.terminate()
                break

        try:
            returncode = int(proc.wait(timeout=5))
        except subprocess.TimeoutExpired:
            proc.kill()
            returncode = -1

        while True:
            try:
                emit(lines.get_nowait(), log_file)
            except queue.Empty:
                break

    output = "".join(output_parts)
    parsed = parse_run_output(output)
    parsed["returncode"] = returncode
    parsed["timed_out"] = timed_out
    parsed["log"] = str(log_path)
    parsed["cmd"] = cmd
    return parsed


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    default_base = data_dir / "section_tuning_36.json"
    default_out = data_dir / "auto_tune_submission36"

    parser = argparse.ArgumentParser(description="Coordinate-search tuner for direct 36-section submission.py.")
    parser.add_argument("--base-bundle", default=str(default_base))
    parser.add_argument("--out-dir", default=str(default_out))
    parser.add_argument("--groups", default="A,B,C,D,E,F", help="Comma list of group names or section ids.")
    parser.add_argument(
        "--params",
        default="friction_mus",
        help="Comma list: friction_mus,lookahead_scales,steer_gain_scales,brake_distance_scales",
    )
    parser.add_argument("--max-seconds", type=float, default=400.0)
    parser.add_argument("--timeout-seconds", type=float, default=650.0)
    parser.add_argument("--failure-grace-seconds", type=float, default=3.0)
    parser.add_argument("--mu-step", type=float, default=0.05)
    parser.add_argument("--lookahead-step", type=float, default=0.05)
    parser.add_argument("--steer-step", type=float, default=0.03)
    parser.add_argument("--brake-distance-step", type=float, default=0.05)
    parser.add_argument("--min-improvement", type=float, default=0.03)
    parser.add_argument("--max-trials", type=int, default=9999)
    parser.add_argument("--initial-best-time", type=float, default=None)
    parser.add_argument("--aggressive", action="store_true")
    parser.add_argument("--prefer-smoothing", action="store_true")
    parser.add_argument("--no-rendering", action="store_true", default=True)
    parser.add_argument(
        "--stop-on-baseline-fail",
        action="store_true",
        help="Stop immediately if the starting bundle crashes instead of searching for a repair.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_path = Path(args.base_bundle).resolve()
    out_dir = Path(args.out_dir).resolve()
    candidates_dir = out_dir / "candidates"
    logs_dir = out_dir / "logs"
    summary_path = out_dir / "summary.jsonl"
    best_path = out_dir / "best_bundle.json"

    sections = parse_groups(args.groups)
    params = parse_params(args.params)
    best_bundle = load_json(base_path)
    absolutize_bundle_paths(best_bundle, base_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.initial_best_time is None:
        baseline_path = candidates_dir / f"{timestamp}_baseline.json"
        write_json(baseline_path, best_bundle)
        if args.dry_run:
            print(f"[dry-run] would run baseline {baseline_path}")
            return
        baseline = run_bundle(baseline_path, logs_dir / f"{timestamp}_baseline.txt", args)
        append_jsonl(summary_path, {"kind": "baseline", **baseline})
        if not baseline.get("finished"):
            print(f"[auto-tune] Baseline did not finish. See {baseline['log']}")
            if args.stop_on_baseline_fail:
                return
            best_time = float("inf")
            print("[auto-tune] entering repair mode; first finished candidate becomes the best bundle")
        else:
            best_time = float(baseline["time"])
    else:
        best_time = float(args.initial_best_time)

    write_json(best_path, best_bundle)
    print(f"[auto-tune] starting best_time={best_time:.3f}s sections={sections} params={params}")

    trial = 0
    for param in params:
        for section in sections:
            for delta in candidate_deltas(param, args):
                if trial >= int(args.max_trials):
                    print(f"[auto-tune] reached max_trials={args.max_trials}")
                    return
                current = get_param(best_bundle, param, section)
                value = clamp_param(param, current + float(delta))
                if abs(value - current) < 1e-9:
                    continue

                trial += 1
                candidate = copy.deepcopy(best_bundle)
                set_param(candidate, param, section, value)
                tag = f"{timestamp}_trial{trial:03d}_{param}_s{section}_{value:g}"
                candidate_path = candidates_dir / f"{tag}.json"
                log_path = logs_dir / f"{tag}.txt"
                write_json(candidate_path, candidate)

                row: Dict[str, Any] = {
                    "kind": "trial",
                    "trial": trial,
                    "param": param,
                    "section": section,
                    "old_value": current,
                    "new_value": value,
                    "candidate": str(candidate_path),
                }
                print(f"[auto-tune] trial={trial} {param}[{section}] {current:g}->{value:g}")
                if args.dry_run:
                    row["dry_run"] = True
                    append_jsonl(summary_path, row)
                    continue

                result = run_bundle(candidate_path, log_path, args)
                row.update(result)
                accepted = bool(result.get("finished")) and float(result["time"]) < best_time - float(args.min_improvement)
                row["accepted"] = accepted
                append_jsonl(summary_path, row)

                if accepted:
                    best_time = float(result["time"])
                    best_bundle = candidate
                    write_json(best_path, best_bundle)
                    print(f"[auto-tune] accepted best_time={best_time:.3f}s -> {best_path}")
                else:
                    if result.get("timed_out"):
                        reason = "timeout"
                    elif result.get("crashed"):
                        reason = "crash"
                    elif not result.get("finished"):
                        reason = "failed to finish"
                    else:
                        reason = f"no improvement ({float(result['time']):.3f}s)"
                    print(f"[auto-tune] rejected ({reason}); best_time={best_time:.3f}s")

    print(f"[auto-tune] done best_time={best_time:.3f}s best_bundle={best_path}")


if __name__ == "__main__":
    main()
