from __future__ import annotations

import os
import uuid
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from zarr.abc.store import Store
from zarr.storage import MemoryStore

from zarr_imaging_daos import ingest_fits_cube, ingest_mri_volume


@pytest.fixture(params=["memory", "daos_array"])
def store(request: pytest.FixtureRequest) -> Store:
    if request.param == "memory":
        return MemoryStore()

    pool = os.environ.get("DAOS_POOL")
    cont = os.environ.get("DAOS_CONT")
    if not pool or not cont:
        pytest.skip("Set DAOS_POOL and DAOS_CONT to run DAOSArrayStore integration tests.")

    from zarr_daos.libdaos_store import DAOSArrayStore

    return DAOSArrayStore(pool=pool, cont=cont, prefix=f"img_test_{uuid.uuid4().hex}")


def test_fits_cube_ingestion_and_wcs_accessor(tmp_path: Path, store: Store) -> None:
    cube = np.random.default_rng(7).standard_normal((1, 16, 32, 32)).astype(np.float32)

    wcs = WCS(naxis=4)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
    wcs.wcs.crval = [180.0, -30.0, 1.42e9, 1.0]
    wcs.wcs.cdelt = [-0.001, 0.001, 1.0e5, 1.0]
    wcs.wcs.crpix = [16.0, 16.0, 1.0, 1.0]

    fits_file = tmp_path / "synthetic_hi.fits"
    hdu = fits.PrimaryHDU(data=cube, header=wcs.to_header())
    hdu.header["BUNIT"] = "Jy/beam"
    hdu.writeto(fits_file)

    ds = ingest_fits_cube(fits_file, store, chunks=(1, 8, 16, 16))
    assert set(ds["SKY"].dims) == {"stokes", "frequency", "dec", "ra"}
    np.testing.assert_allclose(ds["SKY"].values, cube)
    assert ds.radio.wcs.wcs.ctype[2] == "FREQ"


def test_mri_nifti_ingestion_and_pyramid(tmp_path: Path, store: Store) -> None:
    vol = np.linspace(0, 1000, 64 * 64 * 64, dtype=np.float32).reshape(64, 64, 64)
    affine = np.diag([1.5, 1.5, 2.0, 1.0])
    nifti_file = tmp_path / "sub01_t1w.nii.gz"
    nib.save(nib.Nifti1Image(vol, affine), nifti_file)

    dt = ingest_mri_volume(nifti_file, store, pyramid_levels=2, spatial_chunk=32)

    assert dt["0"].ds["image"].shape == (64, 64, 64)
    assert dt["1"].ds["image"].shape == (32, 32, 32)
    assert dt.attrs["ome"]["version"] == "0.5"

    reconstructed = dt["0"].ds.mri.to_nifti()
    np.testing.assert_allclose(reconstructed.get_fdata(), vol)
    np.testing.assert_allclose(reconstructed.affine, affine)

    expected_l1_affine = affine.copy()
    expected_l1_affine[:3, :3] *= 2.0
    np.testing.assert_allclose(dt["1"].ds.mri.affine, expected_l1_affine)
