#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path("/home/mmathew23/repos")
UNSLOTH_ZOO_ROOT = REPO_ROOT / "unsloth-zoo"
UNSLOTH_ROOT = REPO_ROOT / "unsloth"
WORKTREE_ROOT = REPO_ROOT / ".worktrees"
NOTEBOOK = REPO_ROOT / "notebooks/python_scripts/qwen3vlcopy.py"
VENV_PYTHON = Path("/mnt/disks/unslothai/mathew/.venv/bin/python")
RESULTS_ROOT = REPO_ROOT / "outputs/qwen3vlcopy_parallel_tracks"


@dataclass(frozen = True)
class Track:
    name: str
    gpu: int
    use_reentrant: bool
    extra_env: dict[str, str]


TRACKS = [
    Track("track0_reentrant_control", 0, True, {}),
    Track("track1_nonreentrant_control", 1, False, {}),
    Track("track2_nr_disable_fast_lora", 2, False, {
        "UNSLOTH_NR_DISABLE_FAST_LORA": "1",
    }),
    Track("track3_nr_disable_text_mlp_compile", 3, False, {
        "UNSLOTH_NR_DISABLE_QWEN3VL_TEXT_MLP_COMPILE": "1",
    }),
    Track("track4_nr_disable_vision_mlp_compile", 4, False, {
        "UNSLOTH_NR_DISABLE_QWEN3VL_VISION_MLP_COMPILE": "1",
    }),
    Track("track5_nr_disable_vision_merger_compile", 5, False, {
        "UNSLOTH_NR_DISABLE_QWEN3VL_VISION_MERGER_COMPILE": "1",
    }),
    Track("track6_nr_disable_all_hot_subpaths", 6, False, {
        "UNSLOTH_NR_DISABLE_QWEN3VL_TEXT_MLP_COMPILE": "1",
        "UNSLOTH_NR_DISABLE_QWEN3VL_VISION_MLP_COMPILE": "1",
        "UNSLOTH_NR_DISABLE_QWEN3VL_VISION_MERGER_COMPILE": "1",
    }),
    Track("track7_nr_disable_fast_lora_plus_all_hot_subpaths", 7, False, {
        "UNSLOTH_NR_DISABLE_FAST_LORA": "1",
        "UNSLOTH_NR_DISABLE_QWEN3VL_TEXT_MLP_COMPILE": "1",
        "UNSLOTH_NR_DISABLE_QWEN3VL_VISION_MLP_COMPILE": "1",
        "UNSLOTH_NR_DISABLE_QWEN3VL_VISION_MERGER_COMPILE": "1",
    }),
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd = cwd, check = True)


def capture(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text = True).strip()


def ensure_worktree(track: Track) -> Path:
    path = WORKTREE_ROOT / f"unsloth-zoo-{track.name}"
    root_head = capture(["git", "-C", str(UNSLOTH_ZOO_ROOT), "rev-parse", "HEAD"])
    if path.exists():
        try:
            worktree_head = capture(["git", "-C", str(path), "rev-parse", "HEAD"])
        except subprocess.CalledProcessError:
            worktree_head = ""
        if worktree_head == root_head:
            return path
        run(["git", "-C", str(UNSLOTH_ZOO_ROOT), "worktree", "remove", "--force", str(path)])
    run([
        "git", "-C", str(UNSLOTH_ZOO_ROOT), "worktree", "add", "--detach",
        str(path), "HEAD",
    ])
    return path


def parse_profile_summary(path: Path) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    if not path.exists():
        return results
    line_re = re.compile(
        r"^(?P<key>.+?)\tcount=(?P<count>\d+)\tself_cpu_ms=(?P<self_cpu>[0-9.]+)\t"
        r"cpu_total_ms=(?P<cpu_total>[0-9.]+)\tself_cuda_ms=(?P<self_cuda>[0-9.]+)\t"
        r"cuda_total_ms=(?P<cuda_total>[0-9.]+)$"
    )
    for line in path.read_text(encoding = "utf-8").splitlines():
        match = line_re.match(line.strip())
        if not match:
            continue
        results[match.group("key")] = {
            "count": int(match.group("count")),
            "self_cpu_ms": float(match.group("self_cpu")),
            "cpu_total_ms": float(match.group("cpu_total")),
            "self_cuda_ms": float(match.group("self_cuda")),
            "cuda_total_ms": float(match.group("cuda_total")),
        }
    return results


def collect_metrics(
    summary: dict[str, dict[str, float | int]],
    *,
    exact_keys: tuple[str, ...] = (),
    prefixes: tuple[str, ...] = (),
) -> dict[str, float | int]:
    aggregate = {
        "count": 0,
        "self_cpu_ms": 0.0,
        "cpu_total_ms": 0.0,
        "self_cuda_ms": 0.0,
        "cuda_total_ms": 0.0,
    }
    for key, value in summary.items():
        if key in exact_keys or any(key.startswith(prefix) for prefix in prefixes):
            aggregate["count"] += int(value["count"])
            aggregate["self_cpu_ms"] += float(value["self_cpu_ms"])
            aggregate["cpu_total_ms"] += float(value["cpu_total_ms"])
            aggregate["self_cuda_ms"] += float(value["self_cuda_ms"])
            aggregate["cuda_total_ms"] += float(value["cuda_total_ms"])
    return aggregate


def get_copy_metric(summary: dict[str, dict[str, float | int]], key_fragment: str) -> dict[str, float | int]:
    aggregate = {
        "count": 0,
        "self_cpu_ms": 0.0,
        "cpu_total_ms": 0.0,
        "self_cuda_ms": 0.0,
        "cuda_total_ms": 0.0,
    }
    for key, value in summary.items():
        if key_fragment in key:
            aggregate["count"] += int(value["count"])
            aggregate["self_cpu_ms"] += float(value["self_cpu_ms"])
            aggregate["cpu_total_ms"] += float(value["cpu_total_ms"])
            aggregate["self_cuda_ms"] += float(value["self_cuda_ms"])
            aggregate["cuda_total_ms"] += float(value["cuda_total_ms"])
    return aggregate


def parse_log_metrics(log_path: Path) -> dict[str, float | str | None]:
    text = log_path.read_text(encoding = "utf-8", errors = "replace")
    metrics: dict[str, float | str | None] = {
        "train_runtime_s": None,
        "peak_reserved_gb": None,
        "peak_reserved_training_gb": None,
        "gpu_name": None,
    }
    runtime_match = re.search(r"'train_runtime':\s*([0-9.]+)", text)
    if runtime_match:
        metrics["train_runtime_s"] = float(runtime_match.group(1))
    else:
        runtime_match = re.search(r"([0-9.]+)\s+seconds used for training\.", text)
        if runtime_match:
            metrics["train_runtime_s"] = float(runtime_match.group(1))
    peak_match = re.search(r"Peak reserved memory = ([0-9.]+) GB\.", text)
    if peak_match:
        metrics["peak_reserved_gb"] = float(peak_match.group(1))
    peak_training_match = re.search(r"Peak reserved memory for training = ([0-9.]+) GB\.", text)
    if peak_training_match:
        metrics["peak_reserved_training_gb"] = float(peak_training_match.group(1))
    gpu_match = re.search(r"GPU = (.+?)\. Max memory =", text)
    if gpu_match:
        metrics["gpu_name"] = gpu_match.group(1)
    return metrics


def launch_track(track: Track) -> tuple[subprocess.Popen[bytes], Path, Path]:
    worktree = ensure_worktree(track)
    track_root = RESULTS_ROOT / track.name
    profile_dir = track_root / "profile"
    compile_dir = track_root / "torch_compile_debug"
    compiled_cache_dir = worktree / "unsloth_compiled_cache"
    inductor_cache_dir = Path("/tmp") / f"torchinductor_{track.name}"
    log_path = track_root / "run.log"
    track_root.mkdir(parents = True, exist_ok = True)
    profile_dir.mkdir(parents = True, exist_ok = True)
    compile_dir.mkdir(parents = True, exist_ok = True)
    compiled_cache_dir.mkdir(parents = True, exist_ok = True)
    inductor_cache_dir.mkdir(parents = True, exist_ok = True)

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(track.gpu),
        "PYTHONPATH": f"{worktree}:{UNSLOTH_ROOT}",
        "USE_REENTRANT": "1" if track.use_reentrant else "0",
        "UNSLOTH_EXPERIMENT_USE_REENTRANT": "1" if track.use_reentrant else "0",
        "UNSLOTH_COMPILE_LOCATION": str(compiled_cache_dir),
        "UNSLOTH_COMPILE_OVERWRITE": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(inductor_cache_dir),
        "TORCH_COMPILE_DEBUG_DIR": str(compile_dir),
        "UNSLOTH_MAX_STEPS": "2",
        "UNSLOTH_TORCH_PROFILE": "1",
        "UNSLOTH_TORCH_PROFILE_USE_SCHEDULE": "1",
        "UNSLOTH_TORCH_PROFILE_WAIT_STEPS": "1",
        "UNSLOTH_TORCH_PROFILE_WARMUP_STEPS": "0",
        "UNSLOTH_TORCH_PROFILE_ACTIVE_STEPS": "1",
        "UNSLOTH_TORCH_PROFILE_DIR": str(profile_dir),
        "TOKENIZERS_PARALLELISM": "false",
    })
    env.update(track.extra_env)

    log_file = open(log_path, "wb")
    process = subprocess.Popen(
        [str(VENV_PYTHON), str(NOTEBOOK)],
        cwd = str(REPO_ROOT),
        env = env,
        stdout = log_file,
        stderr = subprocess.STDOUT,
    )
    return process, track_root, log_path


def build_track_summary(track: Track, track_root: Path, log_path: Path) -> dict[str, object]:
    mode_name = "reentrant" if track.use_reentrant else "nonreentrant"
    log_metrics = parse_log_metrics(log_path)
    compiled_summary = parse_profile_summary(track_root / "profile" / f"{mode_name}_compiled_summary.txt")
    copy_summary = parse_profile_summary(track_root / "profile" / f"{mode_name}_copy_summary.txt")

    summary = {
        "track": track.name,
        "gpu": track.gpu,
        "use_reentrant": track.use_reentrant,
        "extra_env": track.extra_env,
        **log_metrics,
        "compiled_function": collect_metrics(compiled_summary, exact_keys = ("CompiledFunction",)),
        "compiled_function_backward": collect_metrics(compiled_summary, exact_keys = ("CompiledFunctionBackward",)),
        "torch_compiled_region": collect_metrics(compiled_summary, prefixes = ("Torch-Compiled Region",)),
        "compiled_fx_graph": collect_metrics(compiled_summary, prefixes = ("## Call CompiledFxGraph ",)),
        "aten_copy": get_copy_metric(copy_summary, "aten::copy_"),
        "cuda_memcpy_async": get_copy_metric(copy_summary, "cudaMemcpyAsync"),
        "pinned_h2d": get_copy_metric(copy_summary, "Memcpy HtoD (Pinned -> Device)"),
        "pinned_d2h": get_copy_metric(copy_summary, "Memcpy DtoH (Device -> Pinned)"),
        "profile_dir": str(track_root / "profile"),
        "log_path": str(log_path),
    }
    return summary


def write_summary(tracks: list[dict[str, object]]) -> None:
    RESULTS_ROOT.mkdir(parents = True, exist_ok = True)
    results_path = RESULTS_ROOT / "summary.json"
    results_path.write_text(json.dumps(tracks, indent = 2, sort_keys = True), encoding = "utf-8")

    reentrant = next(track for track in tracks if track["track"] == "track0_reentrant_control")
    nr_control = next(track for track in tracks if track["track"] == "track1_nonreentrant_control")
    re_runtime = float(reentrant["train_runtime_s"] or 0.0)
    nr_runtime = float(nr_control["train_runtime_s"] or 0.0)
    baseline_gap = nr_runtime - re_runtime

    lines = [
        "| Track | GPU | Runtime s | Peak GB | CompiledBackward cpu ms | Compiled cpu ms | Pinned H2D ms | Pinned D2H ms | Gap Recovery |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for track in tracks:
        runtime = float(track["train_runtime_s"] or 0.0)
        peak = float(track["peak_reserved_gb"] or 0.0)
        compiled_backward = float(track["compiled_function_backward"]["cpu_total_ms"])  # type: ignore[index]
        compiled_forward = float(track["compiled_function"]["cpu_total_ms"])  # type: ignore[index]
        pinned_h2d = float(track["pinned_h2d"]["cpu_total_ms"])  # type: ignore[index]
        pinned_d2h = float(track["pinned_d2h"]["cpu_total_ms"])  # type: ignore[index]
        if track["use_reentrant"]:
            gap_recovery = 1.0
        elif baseline_gap <= 0:
            gap_recovery = 0.0
        else:
            gap_recovery = (nr_runtime - runtime) / baseline_gap
        lines.append(
            f"| {track['track']} | {track['gpu']} | {runtime:.2f} | {peak:.2f} | "
            f"{compiled_backward:.2f} | {compiled_forward:.2f} | {pinned_h2d:.2f} | {pinned_d2h:.2f} | {gap_recovery:.3f} |"
        )

    markdown_path = RESULTS_ROOT / "summary.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding = "utf-8")


def main() -> int:
    RESULTS_ROOT.mkdir(parents = True, exist_ok = True)
    launched: list[tuple[Track, subprocess.Popen[bytes], Path, Path]] = []

    for track in TRACKS:
        process, track_root, log_path = launch_track(track)
        launched.append((track, process, track_root, log_path))
        print(f"Launched {track.name} on GPU {track.gpu} -> pid {process.pid}")

    failures = []
    results = []
    for track, process, track_root, log_path in launched:
        returncode = process.wait()
        print(f"{track.name} finished with code {returncode}")
        if returncode != 0:
            failures.append((track.name, returncode, str(log_path)))
            continue
        results.append(build_track_summary(track, track_root, log_path))

    write_summary(results)

    if failures:
        print("Failures detected:")
        for name, returncode, log_path in failures:
            print(f"  {name}: rc={returncode} log={log_path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
