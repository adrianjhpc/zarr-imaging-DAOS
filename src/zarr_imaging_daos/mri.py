from __future__ import annotations

from pathlib import Path
from typing import Any

import dask.array as da
import nibabel as nib
import numpy as np
import xarray as xr
import zarr
from zarr.codecs import BloscCodec
from zarr.abc.store import Store
from zarr_daos.store import DAOSStore


def _downsample_spatial_2x(arr: da.Array) -> da.Array:
    """Downsample spatial axes (z, y, x) by 2x via local mean while keeping time/coil intact."""
    ndim = arr.ndim
    coarsen_axes = {ndim - 3: 2, ndim - 2: 2, ndim - 1: 2}
    slices = tuple(
        slice(0, (s // 2) * 2) if ax in coarsen_axes else slice(None)
        for ax, s in enumerate(arr.shape)
    )
    trimmed = arr[slices]
    return da.coarsen(np.mean, trimmed, coarsen_axes, trim_excess=True).astype(arr.dtype)


def ingest_mri_volume(
    nifti_path: str | Path,
    store: DAOSStore | Store | str,
    *,
    pyramid_levels: int = 3,
    spatial_chunk: int = 64,
) -> xr.DataTree:
    """Ingest a 3D or 4D NIfTI MRI volume into an OME-Zarr multi-scale hierarchy on DAOS."""
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

    dtype = np.dtype(img.get_data_dtype()).newbyteorder("=")
    base_dask = da.from_array(proxy, chunks="auto").transpose(perm).rechunk(chunk_shape).astype(dtype)

    root = zarr.open_group(store=store, mode="w")
    affine_list = img.affine.tolist()
    tr_seconds = float(img.header.get_zooms()[3]) if len(raw_shape) == 4 else None

    root.attrs["nifti_affine"] = affine_list
    root.attrs["voxel_units"] = "millimeter"
    if tr_seconds is not None:
        root.attrs["tr_seconds"] = tr_seconds

    datasets_meta: list[dict[str, Any]] = []
    current_arr = base_dask

    for level in range(pyramid_levels):
        if level > 0 and min(current_arr.shape[-3:]) < 32:
            break

        scale_factor = 2**level
        level_chunks = tuple(min(c, s) for c, s in zip(chunk_shape, current_arr.shape))

        grp = root.create_group(str(level))
        grp.attrs["nifti_affine"] = affine_list
        grp.attrs["scale_factor"] = scale_factor

        z_arr = grp.create_array(
            name="image",
            shape=current_arr.shape,
            chunks=level_chunks,
            dtype=dtype,
            compressors=[BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")],
        )
        z_arr.attrs["_ARRAY_DIMENSIONS"] = dim_names

        for dim, size, base_step in zip(dim_names, current_arr.shape, voxel_sizes):
            step = base_step * scale_factor if dim in ("z", "y", "x") else base_step
            coord_vals = np.arange(size, dtype=np.float64) * step
            c_arr = grp.create_array(
                name=dim,
                shape=coord_vals.shape,
                chunks=coord_vals.shape,
                dtype=np.float64,
            )
            c_arr[:] = coord_vals
            c_arr.attrs["_ARRAY_DIMENSIONS"] = [dim]

        da.to_zarr(current_arr.rechunk(level_chunks), z_arr)

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

        current_arr = _downsample_spatial_2x(current_arr)

    root.attrs["multiscales"] = [
        {
            "version": "0.4",
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
    ]

    zarr.consolidate_metadata(store)
    return xr.open_datatree(store=store, engine="zarr", consolidated=True)


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
