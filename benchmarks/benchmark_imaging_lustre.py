from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import xarray as xr
import zarr
from astropy.io import fits
from astropy.wcs import WCS
from zarr.storage import LocalStore

from zarr_imaging_daos import ingest_fits_cube, ingest_mri_volume


def bench_radio_cube(run_dir: Path, tmp_dir: Path, n_freq: int = 256, mode: str = "both") -> dict:
    shape = (1, n_freq, 512, 512)
    size_mib = (np.prod(shape) * 4) / (1024**2)
    store = LocalStore(run_dir / "radio.zarr")

    t_write = None
    if mode in ("both", "write"):
        cube = np.random.default_rng(7).standard_normal(shape).astype(np.float32)
        wcs = WCS(naxis=4)
        wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
        wcs.wcs.crval = [180.0, -30.0, 1.42e9, 1.0]
        wcs.wcs.cdelt = [-0.001, 0.001, 1.0e5, 1.0]
        wcs.wcs.crpix = [256.0, 256.0, 1.0, 1.0]

        fits_path = tmp_dir / f"radio_{n_freq}.fits"
        if not fits_path.exists():
            hdu = fits.PrimaryHDU(data=cube, header=wcs.to_header())
            hdu.header["BUNIT"] = "Jy/beam"
            hdu.writeto(fits_path, overwrite=True)
        del cube

        t0 = time.perf_counter()
        ds = ingest_fits_cube(fits_path, store)
        t_write = time.perf_counter() - t0
    else:
        ds = xr.open_zarr(store, consolidated=True)

    t_read = None
    notes = "write-only"
    if mode in ("both", "read"):
        t0 = time.perf_counter()
        _ = ds["SKY"].isel(stokes=0, frequency=n_freq // 2).values
        t_chan_ms = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        _ = ds["SKY"].isel(stokes=0, dec=256, ra=256).values
        t_spec_ms = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        _ = ds["SKY"].values
        t_read = time.perf_counter() - t0
        _ = ds.radio.wcs
        notes = f"2D chan: {t_chan_ms:.1f} ms | 1D spec: {t_spec_ms:.1f} ms"

    return {
        "name": f"lustre ({mode}) | Radio FITS ({n_freq}x512^2)",
        "size": f"{size_mib:6.1f} M",
        "write": size_mib / t_write if t_write else None,
        "read": size_mib / t_read if t_read else None,
        "notes": notes,
    }


def bench_mri_3d(run_dir: Path, tmp_dir: Path, mode: str = "both") -> dict:
    shape = (256, 256, 256)
    size_mib = (np.prod(shape) * 4) / (1024**2)
    store = LocalStore(run_dir / "mri3d.zarr")

    t_write = None
    if mode in ("both", "write"):
        nifti_path = tmp_dir / "mri_3d_256.nii"
        if not nifti_path.exists():
            vol = np.random.default_rng(11).standard_normal(shape).astype(np.float32)
            nib.save(nib.Nifti1Image(vol, np.diag([1.0, 1.0, 1.0, 1.0])), nifti_path)
            del vol

        t0 = time.perf_counter()
        dt = ingest_mri_volume(nifti_path, store, pyramid_levels=3, spatial_chunk=64, return_datatree=(mode != "write"))
        t_write = time.perf_counter() - t0
    else:
        dt = xr.open_datatree(store, engine="zarr", consolidated=True)

    t_read = None
    notes = "write-only"
    if mode in ("both", "read"):
        ds0 = dt["0"].ds
        t0 = time.perf_counter()
        _ = ds0["image"].isel(z=128).values
        t_ax = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        _ = ds0["image"].isel(y=128).values
        t_cor = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        _ = ds0["image"].isel(x=128).values
        t_sag = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        _ = ds0["image"].values
        t_read = time.perf_counter() - t0
        notes = f"Ax: {t_ax:.1f} ms | Cor: {t_cor:.1f} ms | Sag: {t_sag:.1f} ms"

    return {
        "name": f"lustre ({mode}) | 3D MRI NIfTI (256^3)",
        "size": f"{size_mib:6.1f} M",
        "write": size_mib / t_write if t_write else None,
        "read": size_mib / t_read if t_read else None,
        "notes": notes,
    }


def bench_fmri_4d(run_dir: Path, tmp_dir: Path, mode: str = "both") -> dict:
    shape = (128, 128, 64, 32)
    size_mib = (np.prod(shape) * 4) / (1024**2)
    store = LocalStore(run_dir / "fmri4d.zarr")

    t_write = None
    if mode in ("both", "write"):
        nifti_path = tmp_dir / "fmri_4d.nii"
        if not nifti_path.exists():
            vol = np.random.default_rng(19).standard_normal(shape).astype(np.float32)
            nib.save(nib.Nifti1Image(vol, np.diag([2.0, 2.0, 2.0, 1.0])), nifti_path)
            del vol

        t0 = time.perf_counter()
        dt = ingest_mri_volume(nifti_path, store, pyramid_levels=2, spatial_chunk=64, return_datatree=(mode != "write"))
        t_write = time.perf_counter() - t0
    else:
        dt = xr.open_datatree(store, engine="zarr", consolidated=True)

    t_read = None
    notes = "write-only"
    if mode in ("both", "read"):
        ds0 = dt["0"].ds
        t0 = time.perf_counter()
        _ = ds0["image"].isel(t=16).values
        t_vol_ms = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        _ = ds0["image"].isel(z=32, y=64, x=64).values
        t_ts_ms = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        _ = ds0["image"].values
        t_read = time.perf_counter() - t0
        notes = f"1 vol: {t_vol_ms:.1f} ms | 1 voxel TS: {t_ts_ms:.1f} ms"

    return {
        "name": f"lustre ({mode}) | 4D fMRI (32x64x128^2)",
        "size": f"{size_mib:6.1f} M",
        "write": size_mib / t_write if t_write else None,
        "read": size_mib / t_read if t_read else None,
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark zarr-imaging-DAOS pipelines on Lustre (LocalStore).")
    parser.add_argument("--lustre-dir", required=True, help="Exact directory on Lustre for the Zarr stores")
    parser.add_argument("--mode", choices=["both", "write", "read"], default="both")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--n-freq", type=int, default=256)
    parser.add_argument("--tmp-dir", default="/dev/shm" if Path("/dev/shm").exists() else None)
    parser.add_argument("--cleanup", action="store_true", help="Delete --lustre-dir after finishing")
    args = parser.parse_args()

    zarr.config.set({"async.concurrency": args.concurrency})
    run_dir = Path(args.lustre_dir)
    if args.mode in ("both", "write"):
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Lustre LocalStore Benchmarks | Mode: {args.mode} | Dir: {run_dir}\n")
    rows = []
    try:
        with tempfile.TemporaryDirectory(dir=args.tmp_dir, prefix="zarr_img_bench_") as td:
            tmp_path = Path(td)
            rows.append(bench_radio_cube(run_dir, tmp_path, n_freq=args.n_freq, mode=args.mode))
            rows.append(bench_mri_3d(run_dir, tmp_path, mode=args.mode))
            rows.append(bench_fmri_4d(run_dir, tmp_path, mode=args.mode))
    finally:
        if args.cleanup and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

    hdr = f"{'Benchmark':<42} | {'Size':>8} | {'Ingest (MiB/s)':>14} | {'Read (MiB/s)':>12} | Notes"
    print("\n" + hdr)
    print("-" * len(hdr) + "-------------------------")
    for r in rows:
        w_str = f"{r['write']:14.1f}" if r["write"] is not None else f"{'-':>14}"
        r_str = f"{r['read']:12.1f}" if r["read"] is not None else f"{'-':>12}"
        print(f"{r['name']:<42} | {r['size']:>8} | {w_str} | {r_str} | {r['notes']}")


if __name__ == "__main__":
    main()
