"""resmon.py — lightweight in-process CPU/GPU resource sampler for training jobs.

A daemon thread samples every ``interval`` seconds:
  - GPU utilization %% and memory used (via pynvml / nvidia-ml-py)
  - CPU %% of this process AND its children (DataLoader workers are separate
    processes — sampling only the main process would miss them entirely)
  - RSS of the main process

``epoch_stats()`` drains the samples accumulated since the previous call and
returns per-epoch aggregates (mean for rates, max for memory), so each epoch's
row in the epoch log reflects that epoch only.

Degrades gracefully: if psutil / pynvml are missing or there is no GPU, the
corresponding fields come back as "" (the epoch log stays parseable). Both deps
are tiny pure wrappers — `pip install psutil nvidia-ml-py`.
"""
import threading
import time

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:        # ImportError, or NVML init failure on a GPU-less node
    _HAS_NVML = False


class ResourceMonitor:
    """Background sampler; ``start()`` it once, call ``epoch_stats()`` per epoch."""

    FIELDS = ("gpu_util_pct", "gpu_mem_gb", "cpu_pct", "rss_gb")

    def __init__(self, interval: float = 2.0, gpu_index: int = 0):
        self.interval = interval
        self._samples = []                       # list of (gpu_util, gpu_mem_gb, cpu_pct, rss_gb)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

        self._gpu = None
        if _HAS_NVML:
            try:
                self._gpu = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                self._gpu = None

        self._proc = psutil.Process() if _HAS_PSUTIL else None
        # Persistent child Process objects: cpu_percent() measures the delta since
        # the PREVIOUS call on the same object, so children must be cached across
        # samples (fresh objects would always read 0).
        self._children = {}
        if self._proc is not None:
            self._proc.cpu_percent(None)         # prime the delta

    # ------------------------------------------------------------------ sampling
    def _sample_once(self):
        gpu_util = gpu_mem = cpu = rss = None
        if self._gpu is not None:
            try:
                gpu_util = float(pynvml.nvmlDeviceGetUtilizationRates(self._gpu).gpu)
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu).used / 1e9
            except Exception:
                pass
        if self._proc is not None:
            try:
                cpu = self._proc.cpu_percent(None)
                # Refresh the child cache (workers come and go between epochs) and
                # add their usage; 100%% == one fully-busy core.
                live = {}
                for ch in self._proc.children(recursive=True):
                    obj = self._children.get(ch.pid)
                    if obj is None:
                        obj = ch
                        obj.cpu_percent(None)    # prime; first reading is 0
                    live[ch.pid] = obj
                    try:
                        cpu += obj.cpu_percent(None)
                    except psutil.NoSuchProcess:
                        pass
                self._children = live
                rss = self._proc.memory_info().rss / 1e9
            except psutil.Error:
                pass
        with self._lock:
            self._samples.append((gpu_util, gpu_mem, cpu, rss))

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample_once()

    # ------------------------------------------------------------------- control
    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="resource-monitor")
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    # ----------------------------------------------------------------- reporting
    def epoch_stats(self):
        """Aggregate + clear samples since the last call.

        Returns {field: value-or-""}: mean for gpu_util/cpu (how busy), max for
        gpu_mem/rss (peak footprint). "" when a source is unavailable or no
        sample landed in the window (e.g. a sub-interval epoch).
        """
        with self._lock:
            samples, self._samples = self._samples, []

        def agg(idx, fn):
            vals = [s[idx] for s in samples if s[idx] is not None]
            return round(fn(vals), 2) if vals else ""

        return {
            "gpu_util_pct": agg(0, lambda v: sum(v) / len(v)),
            "gpu_mem_gb": agg(1, max),
            "cpu_pct": agg(2, lambda v: sum(v) / len(v)),
            "rss_gb": agg(3, max),
        }

    @staticmethod
    def describe():
        """One-line availability note for the run log."""
        return ("resource monitor: psutil=%s nvml=%s"
                % ("on" if _HAS_PSUTIL else "OFF (pip install psutil)",
                   "on" if _HAS_NVML else "OFF (pip install nvidia-ml-py / no GPU)"))
