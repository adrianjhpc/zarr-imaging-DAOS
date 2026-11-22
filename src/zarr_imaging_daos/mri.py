import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import dask.array as da
import nibabel as nib
import numpy as np
import xarray as xr
import zarr
from zarr.abc.store import Store
from zarr.codecs import BloscCodec, BytesCodec
from zarr.core.array import Array, AsyncArray
from zarr.core.buffer import default_buffer_prototype
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.sync import sync
from zarr.storage import LocalStore, StorePath

_DEFAULT_COMPRESSOR = object()


def _downsample_spatial_2x_dask(arr: da.Array) -> da.Array:
    """Downsample spatial axes (z, y, x) by 2x via local mean on a Dask array."""
    ndim = arr.ndim
    coarsen_axes = {ndim - 3: 2, ndim - 2: 2, ndim - 1: 2}
    slices = tuple(
        slice(0, (s // 2) * 2) if ax in coarsen_axes else slice(None)
        for ax, s in enumerate(arr.shape)
    )
    trimmed = arr[slices]
    return da.coarsen(np.mean, trimmed, coarsen_axes, trim_excess=True).astype(arr.dtype)


def _downsample_spatial_2x_numpy(arr: np.ndarray) -> np.ndarray:
    """Fast vectorized 2x spatial downsampling (z, y, x) in memory."""
    if arr.ndim == 3:
        z, y, x = ((s // 2) * 2 for s in arr.shape)
        return (
            arr[:z, :y, :x]
            .reshape(z // 2, 2, y // 2, 2, x // 2, 2)
            .mean(axis=(1, 3, 5))
            .astype(arr.dtype, copy=False)
        )
    t = arr.shape[0]
    z, y, x = ((s // 2) * 2 for s in arr.shape[1:])
    return (
        arr[:, :z, :y, :x]
        .reshape(t, z // 2, 2, y // 2, 2, x // 2, 2)
        .mean(axis=(2, 4, 6))
        .astype(arr.dtype, copy=False)
    )


async def _write_coord_array_async(grp: zarr.Group, dim: str, coord_vals: np.ndarray) -> None:
    c_arr = await grp._async_group.create_array(
        name=dim,
        shape=coord_vals.shape,
        overwrite=True,
        chunks=coord_vals.shape,
        dtype=np.float64,
        dimension_names=[dim],
    )
    await c_arr.setitem(slice(None), coord_vals)


async def _set_many_compat(store: Store, kv_pairs: list[tuple[str, Any]]) -> None:
    """Write multiple (key, Buffer) pairs using the fastest batch API supported by the store."""
    if not kv_pairs:
        return
    if hasattr(store, "set_many"):
        await store.set_many(kv_pairs)
    elif hasattr(store, "_set_many"):
        await store._set_many(kv_pairs)
    else:
        await asyncio.gather(*(store.set(k, v) for k, v in kv_pairs))


async def _write_all_coords_async(
    store: Store,
    coord_tasks: list[tuple[str, str, np.ndarray]],
    base_prefix: str = "",
) -> dict[str, dict[str, Any]]:
    """Write all 1D coordinate arrays (zarr.json + c/0) in a single batched store call."""
    if not coord_tasks:
        return {}

    proto = default_buffer_prototype()
    kv_pairs: list[tuple[str, Any]] = []
    coord_meta: dict[str, dict[str, Any]] = {}
    pfx = f"{base_prefix.strip('/')}/" if base_prefix.strip("/") else ""

    for level_str, dim, coord_vals in coord_tasks:
        arr_le = np.ascontiguousarray(coord_vals, dtype="<f8")
        n = int(arr_le.shape[0])
        rel_path = f"{level_str}/{dim}".lstrip("/")

        meta: dict[str, Any] = {
            "shape": [n],
            "data_type": "float64",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [n]},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": 0.0,
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
            ],
            "attributes": {},
            "zarr_format": 3,
            "node_type": "array",
            "dimension_names": [dim],
        }

        coord_meta[rel_path] = meta
        meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        kv_pairs.append((f"{pfx}{rel_path}/zarr.json", proto.buffer.from_bytes(meta_bytes)))
        kv_pairs.append((f"{pfx}{rel_path}/c/0", proto.buffer.from_bytes(arr_le.tobytes())))

    await _set_many_compat(store, kv_pairs)
    return coord_meta


def ingest_mri_volume(
    nifti_path: str | Path,
    store: Store | str,
    *,
    pyramid_levels: int = 3,
    spatial_chunk: int = 64,
    in_memory_limit_bytes: int = 1024**3,
    compressors: Any = _DEFAULT_COMPRESSOR,
    consolidate: bool = True,
    return_datatree: bool = True,
) -> xr.DataTree | None:
    """Ingest a 3D or 4D NIfTI MRI volume into an OME-Zarr multi-scale hierarchy."""
    if compressors is _DEFAULT_COMPRESSOR:
        compressors = [BloscCodec(cname="lz4", clevel=1, shuffle="shuffle")]
    elif compressors is None:
        compressors = []

    _prof = os.environ.get("ZARR_IMAGING_PROFILE") == "1"
    _t0 = time.perf_counter()
    img = nib.load(str(nifti_path))
    proxy = img.dataobj
    raw_shape = proxy.shape

    if len(raw_shape) not in (3, 4):
        raise ValueError(f"Expected 3D or 4D MRI volume, got shape {raw_shape}.")

    if len(raw_shape) == 3:
        perm = (2, 1, 0)
        dim_names = ["z", "y", "x"]
        voxel_sizes = tuple(float(v) for v in img.header.get_zooms()[:3][::-1])
        chunk_shape = (
            min(spatial_chunk, raw_shape[2]),
            min(spatial_chunk, raw_shape[1]),
            min(spatial_chunk, raw_shape[0]),
        )
        stream_chunks = (raw_shape[0], raw_shape[1], min(spatial_chunk, raw_shape[2]))
    else:
        perm = (3, 2, 1, 0)
        dim_names = ["t", "z", "y", "x"]
        zooms = img.header.get_zooms()
        voxel_sizes = (float(zooms[3]), float(zooms[2]), float(zooms[1]), float(zooms[0]))
        chunk_shape = (
            1,
            min(spatial_chunk, raw_shape[2]),
            min(spatial_chunk, raw_shape[1]),
            min(spatial_chunk, raw_shape[0]),
        )
        stream_chunks = (raw_shape[0], raw_shape[1], raw_shape[2], 1)

    raw_dtype = np.dtype(img.get_data_dtype()).newbyteorder("=")
    is_scaled = (getattr(proxy, "slope", 1.0) not in (1.0, None)) or (
        getattr(proxy, "inter", 0.0) not in (0.0, None)
    )
    dtype = np.result_type(raw_dtype, np.float32) if is_scaled else raw_dtype
    fill_val: float | int = 0.0 if np.issubdtype(dtype, np.floating) else 0

    est_bytes = int(np.prod(raw_shape)) * dtype.itemsize
    use_in_memory = est_bytes <= in_memory_limit_bytes

    if use_in_memory:
        current_arr: np.ndarray | da.Array = np.ascontiguousarray(
            np.transpose(np.asarray(proxy, dtype=dtype), perm)
        )
    else:
        current_arr = da.from_array(proxy, chunks=stream_chunks).transpose(perm).astype(dtype)

    _t_load = time.perf_counter() - _t0
    _t_img = 0.0
    _t_down = 0.0

    # Resolve StorePath without issuing open_group() existence probes
    if isinstance(store, StorePath):
        store_obj = store.store
        base_prefix = store.path.strip("/")
    elif isinstance(store, Store):
        store_obj = store
        base_prefix = ""
    else:
        store_obj = LocalStore(store)
        base_prefix = ""
    pfx = f"{base_prefix}/" if base_prefix else ""
    proto = default_buffer_prototype()

    base_affine = np.asarray(img.affine, dtype=np.float64)
    tr_seconds = float(img.header.get_zooms()[3]) if len(raw_shape) == 4 else None

    # 1. Pre-compute all pyramid level group + image metadata in memory (0 RPCs)
    level_specs: list[tuple[int, tuple[int, ...], tuple[int, ...], ArrayV3Metadata]] = []
    meta_kv_pairs: list[tuple[str, Any]] = []
    consolidated_meta: dict[str, Any] = {}
    datasets_meta: list[dict[str, Any]] = []

    shape_iter = tuple(current_arr.shape)
    codecs = [BytesCodec(endian="little"), *compressors]

    for level in range(pyramid_levels):
        if level > 0 and min(shape_iter[-3:]) < 32:
            break

        scale_factor = 2**level
        level_chunks = tuple(min(c, s) for c, s in zip(chunk_shape, shape_iter))
        level_affine = base_affine.copy()
        level_affine[:3, :3] *= scale_factor

        grp_meta: dict[str, Any] = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "nifti_affine": level_affine.tolist(),
                "scale_factor": scale_factor,
            },
        }
        img_meta_dict: dict[str, Any] = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": list(shape_iter),
            "data_type": str(dtype),
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": list(level_chunks)},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": fill_val,
            "codecs": [c.to_dict() for c in codecs],
            "attributes": {},
            "dimension_names": dim_names,
        }
        img_meta_obj = ArrayV3Metadata.from_dict(img_meta_dict)

        consolidated_meta[str(level)] = grp_meta
        consolidated_meta[f"{level}/image"] = img_meta_dict

        meta_kv_pairs.append(
            (
                f"{pfx}{level}/zarr.json",
                proto.buffer.from_bytes(json.dumps(grp_meta, separators=(",", ":")).encode("utf-8")),
            )
        )
        meta_kv_pairs.append(
            (
                f"{pfx}{level}/image/zarr.json",
                proto.buffer.from_bytes(json.dumps(img_meta_dict, separators=(",", ":")).encode("utf-8")),
            )
        )

        scales = [
            base_step * scale_factor if dim in ("z", "y", "x") else base_step
            for dim, base_step in zip(dim_names, voxel_sizes)
        ]
        datasets_meta.append(
            {
                "path": f"{level}/image",
                "coordinateTransformations": [{"type": "scale", "scale": scales}],
            }
        )
        level_specs.append((level, shape_iter, level_chunks, img_meta_obj))

        if len(shape_iter) == 3:
            shape_iter = tuple(s // 2 for s in shape_iter)
        else:
            shape_iter = (shape_iter[0], *(s // 2 for s in shape_iter[1:]))

    # 2. Write all level group + image zarr.json files in a single batch
    _tm0 = time.perf_counter()
    sync(_set_many_compat(store_obj, meta_kv_pairs))
    _t_meta = time.perf_counter() - _tm0

    # 3. Write image chunks directly via in-memory Array wrappers
    coord_tasks: list[tuple[str, str, np.ndarray]] = []
    for idx, (level, _, level_chunks, img_meta_obj) in enumerate(level_specs):
        scale_factor = 2**level
        img_store_path = StorePath(store_obj, f"{pfx}{level}/image")
        z_arr = Array(AsyncArray(metadata=img_meta_obj, store_path=img_store_path))

        for dim, size, base_step in zip(dim_names, current_arr.shape, voxel_sizes):
            step = base_step * scale_factor if dim in ("z", "y", "x") else base_step
            coord_tasks.append((str(level), dim, np.arange(size, dtype=np.float64) * step))

        _ti = time.perf_counter()
        if isinstance(current_arr, np.ndarray):
            z_arr[:] = current_arr
        else:
            if current_arr.chunksize != level_chunks:
                current_arr = current_arr.rechunk(level_chunks)
            da.to_zarr(current_arr, z_arr)
        _t_img += time.perf_counter() - _ti

        if idx + 1 < len(level_specs):
            _td = time.perf_counter()
            if isinstance(current_arr, np.ndarray):
                current_arr = _downsample_spatial_2x_numpy(current_arr)
            else:
                current_arr = _downsample_spatial_2x_dask(da.from_zarr(z_arr))
            _t_down += time.perf_counter() - _td

    # 4. Write all coordinate arrays in a single batch
    _tc = time.perf_counter()
    if coord_tasks:
        coord_meta = sync(_write_all_coords_async(store_obj, coord_tasks, base_prefix=base_prefix))
        consolidated_meta.update(coord_meta)
    _t_coords = time.perf_counter() - _tc

    # 5. Write consolidated root zarr.json in one shot
    root_attrs: dict[str, Any] = {
        "nifti_affine": base_affine.tolist(),
        "voxel_units": "millimeter",
        "ome": {
            "version": "0.5",
            "multiscales": [
                {
                    "name": Path(nifti_path).stem,
                    "axes": [
                        {
                            "name": d,
                            "type": "time" if d == "t" else "space",
                            "unit": "second" if d == "t" else "millimeter",
                        }
                        for d in dim_names
                    ],
                    "datasets": datasets_meta,
                }
            ],
        },
    }
    if tr_seconds is not None:
        root_attrs["tr_seconds"] = tr_seconds

    _t2 = time.perf_counter()
    root_meta: dict[str, Any] = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": root_attrs,
    }
    if consolidate:
        root_meta["consolidated_metadata"] = {
            "kind": "inline",
            "must_understand": False,
            "metadata": consolidated_meta,
        }
    sync(
        store_obj.set(
            f"{pfx}zarr.json",
            proto.buffer.from_bytes(json.dumps(root_meta, separators=(",", ":")).encode("utf-8")),
        )
    )
    _t_cons = time.perf_counter() - _t2

    _t3 = time.perf_counter()
    dt = xr.open_datatree(store, engine="zarr", consolidated=consolidate) if return_datatree else None
    _t_open = time.perf_counter() - _t3

    if _prof:
        _total = time.perf_counter() - _t0
        print(
            f"    [MRI profile] total={_total*1e3:.1f}ms | "
            f"load={_t_load*1e3:.1f}ms | create_meta={_t_meta*1e3:.1f}ms | "
            f"img_write={_t_img*1e3:.1f}ms | coords={_t_coords*1e3:.1f}ms | "
            f"downsample={_t_down*1e3:.1f}ms | consolidate={_t_cons*1e3:.1f}ms | "
            f"open_dt={_t_open*1e3:.1f}ms"
        )
    return dt

@xr.register_dataset_accessor("mri")
class MRIAccessor:
    """Domain accessor for physical-space slicing and NIfTI affine export."""

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj

    @property
    def affine(self) -> np.ndarray:
        raw = self._obj.attrs.get("nifti_affine")
        if raw is None:
            raise AttributeError("Dataset does not contain NIfTI affine metadata.")
        return np.asarray(raw, dtype=np.float64)

    def to_nifti(self, var_name: str = "image") -> nib.Nifti1Image:
        """Reconstruct a nibabel NIfTI image in canonical (x, y, z[, t]) order."""
        data = self._obj[var_name].values
        inv_perm = (2, 1, 0) if data.ndim == 3 else (3, 2, 1, 0)
        return nib.Nifti1Image(np.transpose(data, inv_perm), affine=self.affine)
