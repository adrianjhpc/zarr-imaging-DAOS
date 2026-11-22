from __future__ import annotations

from pathlib import Path
from typing import Any

import asyncio
import json
import dask.array as da
import numpy as np
import xarray as xr
import zarr
from astropy.io import fits
from astropy.wcs import WCS
from zarr.codecs import BloscCodec
from zarr.abc.store import Store
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

def _extract_fits_coordinates(
    header: fits.Header, shape: tuple[int, ...]
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Derive Xarray dimension names and 1D world coordinate arrays from a FITS WCS header."""
    wcs = WCS(header)
    ndim = len(shape)
    fits_ctypes = [header.get(f"CTYPE{i}", f"AXIS{i}").lower() for i in range(ndim, 0, -1)]

    dim_names: list[str] = []
    coords: dict[str, np.ndarray] = {}

    for numpy_axis, (ctype, size) in enumerate(zip(fits_ctypes, shape)):
        fits_axis = ndim - 1 - numpy_axis
        if "ra" in ctype or "glon" in ctype:
            name = "ra"
        elif "dec" in ctype or "glat" in ctype:
            name = "dec"
        elif "freq" in ctype or "vrad" in ctype or "velo" in ctype:
            name = "frequency"
        elif "stokes" in ctype:
            name = "stokes"
        else:
            name = ctype.split("-")[0] or f"dim_{numpy_axis}"

        dim_names.append(name)

        pix = np.zeros((size, ndim), dtype=np.float64)
        for ax in range(ndim):
            pix[:, ax] = header.get(f"CRPIX{ax + 1}", 1.0) - 1.0
        pix[:, fits_axis] = np.arange(size, dtype=np.float64)

        try:
            world = wcs.wcs_pix2world(pix, 0)
            coords[name] = world[:, fits_axis]
        except Exception:
            coords[name] = np.arange(size, dtype=np.float64)

    return dim_names, coords


def _default_radio_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Choose balanced 3D chunks (~8-16 MiB for float32) supporting spectral and spatial slicing."""
    if len(shape) == 4:
        return (1, min(64, shape[1]), min(256, shape[2]), min(256, shape[3]))
    if len(shape) == 3:
        return (min(64, shape[0]), min(256, shape[1]), min(256, shape[2]))
    return tuple(min(512, s) for s in shape)


async def _write_fits_coords_and_root_async(
    store: Store,
    coords: dict[str, np.ndarray],
    img_meta: dict[str, Any],
    var_name: str,
    root_attrs: dict[str, Any],
) -> None:
    proto = default_buffer_prototype()
    kv_pairs: list[tuple[str, Any]] = []
    consolidated: dict[str, Any] = {var_name: img_meta}

    for dim_name, coord_vals in coords.items():
        arr_le = np.ascontiguousarray(coord_vals, dtype=coord_vals.dtype.newbyteorder("<"))
        n = int(arr_le.shape[0])
        meta: dict[str, Any] = {
            "shape": [n],
            "data_type": str(arr_le.dtype),
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [n]},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": 0.0 if np.issubdtype(arr_le.dtype, np.floating) else 0,
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
            ],
            "attributes": {},
            "zarr_format": 3,
            "node_type": "array",
            "dimension_names": [dim_name],
        }
        consolidated[dim_name] = meta
        meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        kv_pairs.append((f"{dim_name}/zarr.json", proto.buffer.from_bytes(meta_bytes)))
        kv_pairs.append((f"{dim_name}/c/0", proto.buffer.from_bytes(arr_le.tobytes())))

    root_meta: dict[str, Any] = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": root_attrs,
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": consolidated,
        },
    }
    kv_pairs.append(
        ("zarr.json", proto.buffer.from_bytes(json.dumps(root_meta, separators=(",", ":")).encode("utf-8")))
    )

    if hasattr(store, "set_many"):
        await store.set_many(kv_pairs)
    elif hasattr(store, "_set_many"):
        await store._set_many(kv_pairs)
    else:
        await asyncio.gather(*(store.set(k, v) for k, v in kv_pairs))


def ingest_fits_cube(
    fits_path: str | Path,
    store: Store | str,
    *,
    hdu_index: int = 0,
    chunks: tuple[int, ...] | None = None,
    var_name: str = "SKY",
    in_memory_limit_bytes: int = 2 * 1024**3,
) -> xr.Dataset:
    """Stream a FITS image cube into DAOSStore and return a lazy Xarray Dataset."""
    with fits.open(fits_path, memmap=True) as hdul:
        hdu = hdul[hdu_index]
        if hdu.data is None:
            raise ValueError(f"HDU {hdu_index} in '{fits_path}' contains no image data.")

        shape = hdu.data.shape
        native_dtype = hdu.data.dtype.newbyteorder("=")
        chunk_shape = chunks or _default_radio_chunks(shape)
        dim_names, coords = _extract_fits_coordinates(hdu.header, shape)

        header_cards: dict[str, Any] = {
            k: v
            for k, v in hdu.header.items()
            if k and k not in ("COMMENT", "HISTORY", "") and isinstance(v, (str, int, float, bool))
        }
        root_attrs: dict[str, Any] = {
            "conventions": "xradio-image-v0.1",
            "fits_header": header_cards,
            "wcs_header_string": hdu.header.tostring(sep="\n"),
        }

        root = zarr.open_group(store=store, mode="a")

        # Write main image cube with attributes in one metadata write
        img_arr = root.create_array(
            name=var_name,
            shape=shape,
            chunks=chunk_shape,
            dtype=native_dtype,
            dimension_names=dim_names,
            attributes={"units": hdu.header.get("BUNIT", "Jy/beam")},
            compressors=[BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")],
            overwrite=False,
        )

        # Write all 1D coordinates + consolidated root zarr.json in one set_many batch
        sync(
            _write_fits_coords_and_root_async(
                root.store,
                coords,
                img_arr.metadata.to_dict(),
                var_name,
                root_attrs,
            )
        )

        # Fast path: use Zarr's native async BatchedCodecPipeline when cube fits in memory
        est_bytes = int(np.prod(shape)) * native_dtype.itemsize
        if est_bytes <= in_memory_limit_bytes:
            img_arr[:] = np.ascontiguousarray(hdu.data, dtype=native_dtype)
        else:
            dask_data = da.from_array(hdu.data, chunks=chunk_shape).astype(native_dtype)
            da.to_zarr(dask_data, img_arr)

    return xr.open_zarr(store=store, consolidated=True)

@xr.register_dataset_accessor("radio")
class RadioAccessor:
    """Domain accessor adding astropy WCS reconstruction to Xarray Datasets on DAOS."""

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj

    @property
    def wcs(self) -> WCS:
        header_str = self._obj.attrs.get("wcs_header_string")
        if not header_str:
            raise AttributeError("Dataset does not contain serialized FITS WCS metadata.")
        return WCS(fits.Header.fromstring(header_str, sep="\n"))
