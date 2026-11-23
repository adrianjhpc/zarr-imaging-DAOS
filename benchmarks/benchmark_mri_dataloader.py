from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import zarr
from zarr.storage import LocalStore

from zarr_daos import DAOSArrayStore, DAOSStore
from zarr_imaging_daos import ingest_mri_volume

# Per-worker state (initialized once per spawned worker process)
_WORKER_ARRAYS: list[zarr.Array] = []
_WORKER_SHAPES: list[tuple[int, int, int]] = []
_WORKER_PATCH_SIZE: int = 64
_WORKER_RNG: np.random.Generator | None = None


def _make_store(backend: str, pool: str, cont: str, prefix: str, lustre_dir: str) -> Any:
    if backend == "daos_array":
        return DAOSArrayStore(pool=pool, cont=cont, prefix=prefix)
    if backend == "daos":
        return DAOSStore(pool=pool, cont=cont, prefix=prefix)
    if backend == "lustre":
        return LocalStore(Path(lustre_dir) / prefix)
    raise ValueError(f"Unknown backend: {backend}")


def _worker_init(
    backend: str,
    pool: str,
    cont: str,
    run_id: str,
    lustre_dir: str,
    n_subjects: int,
    patch_size: int,
    seed: int,
) -> None:
    global _WORKER_ARRAYS, _WORKER_SHAPES, _WORKER_PATCH_SIZE, _WORKER_RNG
    pid = os.getpid()
    _WORKER_PATCH_SIZE = patch_size
    _WORKER_RNG = np.random.default_rng(seed + pid)
    _WORKER_ARRAYS = []
    _WORKER_SHAPES = []

    for s in range(n_subjects):
        prefix = f"{run_id}/sub-{s:03d}"
        store = _make_store(backend, pool, cont, prefix, lustre_dir)
        # Open full-resolution level 0/image directly via consolidated metadata
        grp = zarr.open_group(store=store, mode="r")
        arr = grp["0/image"]
        _WORKER_ARRAYS.append(arr)
        _WORKER_SHAPES.append(tuple(int(x) for x in arr.shape))


def _fetch_random_patch(_: int) -> tuple[float, int]:
    """Sample one random 3D patch from a random subject and return (latency_s, nbytes)."""
    assert _WORKER_RNG is not None
    p = _WORKER_PATCH_SIZE
    subj_idx = int(_WORKER_RNG.integers(0, len(_WORKER_ARRAYS)))
    arr = _WORKER_ARRAYS[subj_idx]
    dz, dy, dx = _WORKER_SHAPES[subj_idx]

    z0 = int(_WORKER_RNG.integers(0, dz - p + 1))
    y0 = int(_WORKER_RNG.integers(0, dy - p + 1))
    x0 = int(_WORKER_RNG.integers(0, dx - p + 1))

    t0 = time.perf_counter()
    patch = np.asarray(arr[z0 : z0 + p, y0 : y0 + p, x0 : x0 + p])
    dt = time.perf_counter() - t0
    return dt, patch.nbytes


def run_write(args: argparse.Namespace) -> None:
    """Generate a synthetic cohort of 3D MRI NIfTI volumes and ingest them."""
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    nifti_path = tmp_dir / f"template_{args.vol_size}.nii.gz"

    if not nifti_path.exists():
        rng = np.random.default_rng(42)
        vol = rng.standard_normal((args.vol_size, args.vol_size, args.vol_size), dtype=np.float32)
        nib.save(nib.Nifti1Image(vol, np.eye(4)), nifti_path)

    t0 = time.perf_counter()
    for s in range(args.subjects):
        prefix = f"{args.run_id}/sub-{s:03d}"
        store = _make_store(args.backend, args.pool, args.cont, prefix, args.lustre_dir)
        ingest_mri_volume(
            nifti_path,
            store,
            pyramid_levels=1,  # level 0 only for ML patch training
            spatial_chunk=args.chunk_size,
            return_datatree=False,
        )
    dt = time.perf_counter() - t0
    total_mib = (args.subjects * (args.vol_size**3) * 4) / (1024**2)
    print(
        f"[{args.backend}] Ingested {args.subjects} x {args.vol_size}^3 volumes "
        f"({total_mib:.1f} MiB) in {dt:.2f}s ({total_mib / dt:.1f} MiB/s)"
    )


def run_read(args: argparse.Namespace) -> None:
    """Run multi-worker random 3D patch sampling and report throughput + tail latency."""
    worker_counts = [int(w) for w in args.workers.split(",")]
    ctx = mp.get_context("spawn")

    print(
        f"\nBackend: {args.backend} | Cohort: {args.subjects} x {args.vol_size}^3 "
        f"| Chunk: {args.chunk_size}^3 | Patch: {args.patch_size}^3 | Patches/run: {args.patches}"
    )
    print(
        f"{'Workers':>7} | {'Patches/s':>10} | {'Eff. MiB/s':>10} | "
        f"{'p50 (ms)':>9} | {'p95 (ms)':>9} | {'p99 (ms)':>9} | {'Wall (s)':>8}"
    )
    print("-" * 78)

    for nw in worker_counts:
        with ctx.Pool(
            processes=nw,
            initializer=_worker_init,
            initargs=(
                args.backend,
                args.pool,
                args.cont,
                args.run_id,
                args.lustre_dir,
                args.subjects,
                args.patch_size,
                args.seed,
            ),
        ) as pool:
            # Warm up worker pool handles before timing
            _ = list(pool.imap_unordered(_fetch_random_patch, range(nw)))

            t0 = time.perf_counter()
            results = list(pool.imap_unordered(_fetch_random_patch, range(args.patches)))
            wall = time.perf_counter() - t0

        lat_ms = np.array([r[0] * 1e3 for r in results], dtype=np.float64)
        total_mib = sum(r[1] for r in results) / (1024**2)
        pps = args.patches / wall
        mib_s = total_mib / wall
        p50, p95, p99 = np.percentile(lat_ms, [50, 95, 99])

        print(
            f"{nw:>7d} | {pps:>10.1f} | {mib_s:>10.1f} | "
            f"{p50:>9.2f} | {p95:>9.2f} | {p99:>9.2f} | {wall:>8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-worker 3D MRI random-patch DataLoader benchmark")
    parser.add_argument("--mode", choices=["write", "read"], required=True)
    parser.add_argument("--backend", choices=["daos_array", "daos", "lustre"], required=True)
    parser.add_argument("--pool", default="tutorial")
    parser.add_argument("--cont", default="")
    parser.add_argument("--lustre-dir", default="/newlust/zarr_lustre_tmp")
    parser.add_argument("--tmp-dir", default="/tmp/mri_dl_bench")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--subjects", type=int, default=32, help="Number of 3D MRI volumes in cohort")
    parser.add_argument("--vol-size", type=int, default=256, help="Cubic volume dimension (e.g. 256 -> 64 MiB)")
    parser.add_argument("--chunk-size", type=int, default=64, help="Zarr spatial chunk size")
    parser.add_argument("--patch-size", type=int, default=64, help="Random 3D patch size (e.g. 64 or 96)")
    parser.add_argument("--patches", type=int, default=512, help="Total random patches to sample per worker count")
    parser.add_argument("--workers", default="1,4,8,16", help="Comma-separated worker counts to sweep")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.mode == "write":
        run_write(args)
    else:
        run_read(args)


if __name__ == "__main__":
    main()
