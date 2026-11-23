"""
Lightweight resource monitor for local-first experiment runs.

The monitor samples system/process resources in a background thread while the
pipeline runs. It is intentionally best-effort: CPU/RAM metrics work when
psutil is installed; GPU metrics work when nvidia-smi is available; otherwise
those fields are left empty and the run continues.

Network counters are system-wide deltas, not process-isolated traffic. They are
still useful for the local-first claim when combined with an offline execution
check, but they should be reported as system-wide observations.
"""

from __future__ import annotations

import datetime as _dt
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore


@dataclass
class ResourceMonitor:
    """
    Background resource sampler.

    Parameters
    ----------
    run_id:
        Current experiment run identifier.
    sample_interval_seconds:
        Sampling interval in seconds.
    monitor_gpu:
        Whether to attempt nvidia-smi GPU sampling.
    monitor_network:
        Whether to sample system-wide network counters.
    """

    run_id: str
    sample_interval_seconds: float = 1.0
    monitor_gpu: bool = True
    monitor_network: bool = True
    samples: list[dict[str, Any]] = field(default_factory=list)

    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _started_at: float | None = field(default=None, init=False)
    _process: Any = field(default=None, init=False, repr=False)
    _network_start: dict[str, int] | None = field(default=None, init=False)
    _network_end: dict[str, int] | None = field(default=None, init=False)

    def start(self) -> None:
        """Start background resource sampling."""
        self._started_at = time.perf_counter()
        self._stop_event.clear()

        if psutil is not None:
            try:
                self._process = psutil.Process()
                psutil.cpu_percent(interval=None)
                self._process.cpu_percent(interval=None)
            except Exception:
                self._process = None

            if self.monitor_network:
                self._network_start = _network_counters()

        self._thread = threading.Thread(
            target=self._run,
            name=f"ResourceMonitor-{self.run_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        """
        Stop sampling and return an aggregate summary.
        """
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.sample_interval_seconds * 2.0))

        if psutil is not None and self.monitor_network:
            self._network_end = _network_counters()

        return self.summary()

    def summary(self) -> dict[str, Any]:
        """
        Return aggregate metrics from collected samples.
        """
        elapsed_ms = None
        if self._started_at is not None:
            elapsed_ms = (time.perf_counter() - self._started_at) * 1000.0

        summary: dict[str, Any] = {
            "run_id": self.run_id,
            "monitoring_enabled": True,
            "sample_interval_seconds": self.sample_interval_seconds,
            "num_resource_samples": len(self.samples),
            "elapsed_ms_observed": elapsed_ms,
            "psutil_available": psutil is not None,
            "gpu_sampling_available": _nvidia_smi_available(),
            "network_observation_scope": "system_wide_delta" if self.monitor_network else "disabled",
        }

        numeric_fields = [
            "cpu_percent_system",
            "cpu_percent_process",
            "ram_used_mb",
            "ram_percent",
            "process_rss_mb",
            "process_vms_mb",
            "gpu_utilization_percent",
            "gpu_memory_used_mb",
            "gpu_memory_total_mb",
        ]

        for field_name in numeric_fields:
            values = [_as_float(sample.get(field_name)) for sample in self.samples]
            values = [value for value in values if value is not None]

            if values:
                summary[f"{field_name}_mean"] = sum(values) / len(values)
                summary[f"{field_name}_max"] = max(values)

        if self.samples:
            first = self.samples[0]
            summary["cpu_count_logical"] = first.get("cpu_count_logical")
            summary["cpu_count_physical"] = first.get("cpu_count_physical")
            summary["ram_total_mb"] = first.get("ram_total_mb")
            summary["gpu_name"] = first.get("gpu_name")

        if self._network_start and self._network_end:
            summary["network_bytes_sent_delta"] = max(
                0,
                self._network_end.get("bytes_sent", 0)
                - self._network_start.get("bytes_sent", 0),
            )
            summary["network_bytes_recv_delta"] = max(
                0,
                self._network_end.get("bytes_recv", 0)
                - self._network_start.get("bytes_recv", 0),
            )
            summary["network_packets_sent_delta"] = max(
                0,
                self._network_end.get("packets_sent", 0)
                - self._network_start.get("packets_sent", 0),
            )
            summary["network_packets_recv_delta"] = max(
                0,
                self._network_end.get("packets_recv", 0)
                - self._network_start.get("packets_recv", 0),
            )

        return summary

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.samples.append(self._sample())
            self._stop_event.wait(self.sample_interval_seconds)

    def _sample(self) -> dict[str, Any]:
        timestamp_utc = _dt.datetime.now(_dt.UTC).isoformat()

        elapsed_ms = None
        if self._started_at is not None:
            elapsed_ms = (time.perf_counter() - self._started_at) * 1000.0

        sample: dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp_utc": timestamp_utc,
            "elapsed_ms": elapsed_ms,
        }

        if psutil is not None:
            sample.update(
                _sample_psutil(
                    process=self._process,
                    monitor_network=self.monitor_network,
                )
            )

        if self.monitor_gpu:
            sample.update(_sample_gpu())

        return sample


def monitoring_enabled_from_config(config: dict[str, Any], force_monitor: bool = False) -> bool:
    """
    Return whether resource monitoring should be enabled for this run.
    """
    if force_monitor:
        return True
    return bool(config.get("monitoring", {}).get("enabled", False))


def _sample_psutil(process: Any, *, monitor_network: bool) -> dict[str, Any]:
    sample: dict[str, Any] = {}

    try:
        vm = psutil.virtual_memory()
        sample.update(
            {
                "cpu_percent_system": psutil.cpu_percent(interval=None),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "ram_total_mb": vm.total / (1024 * 1024),
                "ram_used_mb": vm.used / (1024 * 1024),
                "ram_percent": vm.percent,
            }
        )
    except Exception:
        pass

    try:
        if process is not None:
            mem = process.memory_info()
            sample.update(
                {
                    "cpu_percent_process": process.cpu_percent(interval=None),
                    "process_rss_mb": mem.rss / (1024 * 1024),
                    "process_vms_mb": mem.vms / (1024 * 1024),
                    "process_num_threads": process.num_threads(),
                }
            )
    except Exception:
        pass

    if monitor_network:
        counters = _network_counters()
        if counters:
            sample.update({f"network_{key}": value for key, value in counters.items()})

    return sample


def _network_counters() -> dict[str, int] | None:
    if psutil is None:
        return None

    try:
        counters = psutil.net_io_counters()
        return {
            "bytes_sent": int(counters.bytes_sent),
            "bytes_recv": int(counters.bytes_recv),
            "packets_sent": int(counters.packets_sent),
            "packets_recv": int(counters.packets_recv),
        }
    except Exception:
        return None


def _sample_gpu() -> dict[str, Any]:
    if not _nvidia_smi_available():
        return {}

    cmd = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return {}

    if completed.returncode != 0 or not completed.stdout.strip():
        return {}

    parsed_rows: list[dict[str, Any]] = []

    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]

        if len(parts) < 4:
            continue

        parsed_rows.append(
            {
                "gpu_name": parts[0],
                "gpu_utilization_percent": _as_float(parts[1]),
                "gpu_memory_used_mb": _as_float(parts[2]),
                "gpu_memory_total_mb": _as_float(parts[3]),
            }
        )

    if not parsed_rows:
        return {}

    return max(parsed_rows, key=lambda row: row.get("gpu_memory_used_mb") or 0.0)


def _nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None