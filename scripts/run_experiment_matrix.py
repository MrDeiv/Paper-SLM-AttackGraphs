#!/usr/bin/env python3
"""
Run an experiment matrix from one or more YAML config directories.

This wrapper executes scripts/run_case.py repeatedly while keeping pilot and full
runs isolated. It can modify run_id/output_root in temporary config copies before
launching each run, so smoke tests do not overwrite final experiment outputs.

Examples:
    # Pilot on one case and three steps, with monitoring, isolated run IDs.
    python scripts/run_experiment_matrix.py \
      --config-dir configs/experiments \
      --case goofy \
      --limit-steps 3 \
      --run-id-suffix _pilot3 \
      --output-root outputs/pilot_runs \
      --monitor \
      --overwrite

    # Full matrix on all cases from each config.
    python scripts/run_experiment_matrix.py \
      --config-dir configs/experiments \
      --output-root outputs/final_runs \
      --monitor \
      --continue-on-error
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an experiment matrix from YAML configs.")
    parser.add_argument(
        "--config-dir",
        action="append",
        default=[],
        help="Directory containing YAML configs. Can be repeated.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Explicit YAML config path. Can be repeated.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only configs whose filename stem or run_id contains this substring. Can be repeated.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude configs whose filename stem or run_id contains this substring. Can be repeated.",
    )
    parser.add_argument("--case", action="append", help="Optional case_id filter. Can be repeated.")
    parser.add_argument("--case-file", help="Optional text file containing one case_id per line.")
    parser.add_argument("--limit-steps", type=int, default=None, help="Optional maximum attack steps per case.")
    parser.add_argument("--output-root", default=None, help="Override run.output_root in temporary config copies.")
    parser.add_argument("--run-id-prefix", default="", help="Prefix added to each config run_id.")
    parser.add_argument("--run-id-suffix", default="", help="Suffix added to each config run_id.")
    parser.add_argument("--monitor", action="store_true", help="Pass --monitor to run_case.py.")
    parser.add_argument("--monitor-interval", type=float, default=None, help="Pass --monitor-interval to run_case.py.")
    parser.add_argument("--offline-run", action="store_true", help="Pass --offline-run to run_case.py.")
    parser.add_argument("--overwrite", action="store_true", help="Pass --overwrite to run_case.py.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run when its output directory exists and contains config.json.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue executing remaining configs after a failed run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write plan without executing.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch run_case.py.")
    parser.add_argument("--log-dir", default="outputs/experiment_logs", help="Directory for matrix logs.")
    parser.add_argument("--plan-out", default=None, help="Optional CSV path for the execution plan.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = discover_configs(args)
    if not configs:
        raise FileNotFoundError("No experiment YAML configs selected")

    cases = resolve_cases(args)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    plan_rows = []
    start_time = time.strftime("%Y%m%d_%H%M%S")
    default_plan_out = log_dir / f"experiment_plan_{start_time}.csv"
    plan_out = Path(args.plan_out) if args.plan_out else default_plan_out

    print(f"Selected configs: {len(configs)}")
    for config_path in configs:
        original_config = load_yaml(config_path)
        modified_config = modify_config(original_config, args)
        run_id = modified_config.get("run", {}).get("run_id", config_path.stem)
        output_root = Path(modified_config.get("run", {}).get("output_root", "outputs/runs"))
        run_dir = output_root / run_id

        command_preview = build_run_case_command(
            python_exe=args.python,
            config_path="<temporary_config>",
            cases=cases,
            limit_steps=args.limit_steps,
            monitor=args.monitor,
            monitor_interval=args.monitor_interval,
            offline_run=args.offline_run,
            overwrite=args.overwrite,
        )

        plan_rows.append(
            {
                "config_path": str(config_path),
                "original_run_id": original_config.get("run", {}).get("run_id", config_path.stem),
                "effective_run_id": run_id,
                "output_root": str(output_root),
                "run_dir": str(run_dir),
                "cases": ";".join(cases) if cases else "<config_cases>",
                "limit_steps": args.limit_steps if args.limit_steps is not None else "",
                "monitor": args.monitor,
                "offline_run": args.offline_run,
                "skip_existing": args.skip_existing,
                "command_preview": " ".join(command_preview),
            }
        )

    write_csv(plan_out, plan_rows)
    print(f"Execution plan: {plan_out}")

    if args.dry_run:
        for row in plan_rows:
            print(row["command_preview"])
        return

    results = []
    with tempfile.TemporaryDirectory(prefix="slm_matrix_configs_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        for idx, config_path in enumerate(configs, start=1):
            original_config = load_yaml(config_path)
            modified_config = modify_config(original_config, args)
            run_id = modified_config.get("run", {}).get("run_id", config_path.stem)
            output_root = Path(modified_config.get("run", {}).get("output_root", "outputs/runs"))
            run_dir = output_root / run_id

            if args.skip_existing and (run_dir / "config.json").exists():
                print(f"[{idx}/{len(configs)}] SKIP existing run: {run_id}")
                results.append(
                    {
                        "config_path": str(config_path),
                        "run_id": run_id,
                        "status": "skipped_existing",
                        "returncode": 0,
                        "elapsed_seconds": 0.0,
                        "log_file": "",
                    }
                )
                continue

            tmp_config_path = tmp_dir_path / f"{idx:03d}_{config_path.stem}.yaml"
            write_yaml(tmp_config_path, modified_config)

            cmd = build_run_case_command(
                python_exe=args.python,
                config_path=str(tmp_config_path),
                cases=cases,
                limit_steps=args.limit_steps,
                monitor=args.monitor,
                monitor_interval=args.monitor_interval,
                offline_run=args.offline_run,
                overwrite=args.overwrite,
            )

            log_file = log_dir / f"{idx:03d}_{run_id}.log"
            print(f"[{idx}/{len(configs)}] RUN {run_id}")
            print(" ".join(cmd))

            t0 = time.perf_counter()
            with log_file.open("w", encoding="utf-8") as log:
                log.write("COMMAND: " + " ".join(cmd) + "\n\n")
                completed = subprocess.run(
                    cmd,
                    cwd=REPO_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            elapsed = time.perf_counter() - t0

            status = "completed" if completed.returncode == 0 else "failed"
            print(f"[{idx}/{len(configs)}] {status.upper()} {run_id} in {elapsed:.1f}s; log={log_file}")

            results.append(
                {
                    "config_path": str(config_path),
                    "run_id": run_id,
                    "status": status,
                    "returncode": completed.returncode,
                    "elapsed_seconds": round(elapsed, 3),
                    "log_file": str(log_file),
                }
            )

            if completed.returncode != 0 and not args.continue_on_error:
                write_csv(log_dir / f"experiment_results_{start_time}.csv", results)
                raise SystemExit(completed.returncode)

    results_path = log_dir / f"experiment_results_{start_time}.csv"
    write_csv(results_path, results)
    print(f"Matrix results: {results_path}")

    failures = [row for row in results if row["status"] == "failed"]
    if failures:
        print(f"Failed runs: {len(failures)}")
        if not args.continue_on_error:
            raise SystemExit(1)


def discover_configs(args: argparse.Namespace) -> list[Path]:
    configs: list[Path] = []

    for explicit in args.config:
        path = Path(explicit)
        if path.exists():
            configs.append(path)

    config_dirs = args.config_dir or ["configs/experiments"]
    for raw_dir in config_dirs:
        config_dir = Path(raw_dir)
        if not config_dir.exists():
            continue
        configs.extend(sorted(config_dir.glob("*.yaml")))
        configs.extend(sorted(config_dir.glob("*.yml")))

    configs = unique_paths(configs)
    selected: list[Path] = []

    for config_path in configs:
        config = load_yaml(config_path)
        run_id = str(config.get("run", {}).get("run_id", config_path.stem))
        haystack = f"{config_path.stem} {run_id}".lower()

        if args.only and not any(token.lower() in haystack for token in args.only):
            continue
        if args.exclude and any(token.lower() in haystack for token in args.exclude):
            continue
        selected.append(config_path)

    return selected


def resolve_cases(args: argparse.Namespace) -> list[str]:
    cases = list(args.case or [])
    if args.case_file:
        path = Path(args.case_file)
        cases.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return list(dict.fromkeys(cases))


def modify_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    modified = json.loads(json.dumps(config))
    run_cfg = modified.setdefault("run", {})
    base_run_id = str(run_cfg.get("run_id", "run"))
    run_cfg["run_id"] = f"{args.run_id_prefix}{base_run_id}{args.run_id_suffix}"
    if args.output_root:
        run_cfg["output_root"] = args.output_root
    return modified


def build_run_case_command(
    *,
    python_exe: str,
    config_path: str,
    cases: list[str],
    limit_steps: int | None,
    monitor: bool,
    monitor_interval: float | None,
    offline_run: bool,
    overwrite: bool,
) -> list[str]:
    cmd = [python_exe, "scripts/run_case.py", "--config", config_path]
    for case_id in cases:
        cmd.extend(["--case", case_id])
    if limit_steps is not None:
        cmd.extend(["--limit-steps", str(limit_steps)])
    if monitor:
        cmd.append("--monitor")
    if monitor_interval is not None:
        cmd.extend(["--monitor-interval", str(monitor_interval)])
    if offline_run:
        cmd.append("--offline-run")
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def unique_paths(paths: list[Path]) -> list[Path]:
    output: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            output.append(path)
    return output


if __name__ == "__main__":
    main()