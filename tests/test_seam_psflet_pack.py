"""A wave-optics pack carries the stop tilt through detector placement once."""

from dataclasses import replace

import ineedvalidation as vv
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import bands_at, focal_image, sky_grid
from optixstuff.disperser import LensletDisperser
from optixstuff.grids import PixelGrid
from optixstuff.psflet_pack import PsfletPack, load_psflet_pack, save_psflet_pack

from coronachrome import IFSRenderer, build_ir
from coronachrome.build import _resolve_psflet_pack

physicaloptix = pytest.importorskip("physicaloptix")

LAM_REF = 1000.0


@pytest.fixture(scope="module")
def tilted_pack_path(tmp_path_factory):
    """Write a flat-illuminated PSFlet shifted 0.3 px by a linear stop OPD.

    Args:
        tmp_path_factory: Factory for temporary directories shared by this module.

    Returns:
        Path to the serialized wave-optics pack.
    """
    grid = PixelGrid.pupil(64)
    axis = (np.arange(64) - grid.center_x) * grid.step
    pupil = jnp.asarray(axis[:, None] ** 2 + axis[None, :] ** 2 <= 0.25)
    optics = dict(
        pitch_lod_ref=0.5,
        lam_ref_nm=LAM_REF,
        micropupil_px=1.5,
        n_tile=48,
        illumination="flat",
        pinhole_radius=1.5,
    )
    base = physicaloptix.ifs.LensletChain(pupil, grid, **optics)
    stop = base.stop_grid
    x = (np.arange(stop.shape[1]) - stop.center_x) * stop.step
    slope = 0.3 * LAM_REF / base.px_per_diffraction(LAM_REF)
    opd = np.broadcast_to(slope * x, stop.shape)
    tilted = physicaloptix.ifs.LensletChain(pupil, grid, **optics, stop_opd_nm=opd)
    pack = physicaloptix.ifs.psflet_pack(
        tilted,
        [LAM_REF],
        half_extent=6.0,
        step=0.1,
        n_quad=10,
    )

    path = tmp_path_factory.mktemp("psflet") / "tilted.npz"
    save_psflet_pack(path, pack)
    return path


def test_producer_writes_centroid_relative_format_two(tilted_pack_path):
    """The producer declares the centroid convention in the format-2 archive."""
    with np.load(tilted_pack_path) as archive:
        assert archive["format_version"] == 2
        assert archive["template_origin"] == "centroid"


@vv.case("psflet-pack-seam", "code-verification")
@vv.seam("physicaloptix.psflet_pack", "coronachrome.build_ir")
def test_render_tilted_pack(tilted_pack_path):
    """Recover the Fourier shift to 0.005 px and conserve the input lenslet flux."""
    disp = LensletDisperser(
        pitch_m=130e-6,
        pixsize_m=13e-6,
        angle_rad=0.0,
        lam_ref_nm=LAM_REF,
        pix_per_reselt=2.0,
        dispersion_coeffs=jnp.array([0.0]),
        psflet_kind="template",
        psflet_pack_path=str(tilted_pack_path),
        psflet_params=jnp.array([1.0]),
        psflet_ref_nm=LAM_REF,
        grid_kind="square",
        n_lenslets=1,
        detector_shape=(21, 21),
        sky_pitch_arcsec=0.2,
    )
    ir = build_ir(
        disp, bands_at([LAM_REF], 1e-4), sky_grid((2, 2)), half=6, quadrature_order=1
    )
    flux = 7.0
    cube = focal_image(jnp.full((1, 2, 2), flux / 4), ir.fp_grid)
    spot = np.asarray(IFSRenderer(ir).forward_spmv(cube).data)
    geometric_x = (disp.detector_shape[1] - 1) / 2
    shift = spot.sum(axis=0) @ (np.arange(spot.shape[1]) - geometric_x) / spot.sum()
    # Pixel integration and finite support give a measured error of +0.000167 px.
    assert abs(shift - 0.3) < 5e-3
    np.testing.assert_allclose(spot.sum(), flux * ir.det_vals.sum(), rtol=1e-6, atol=0)


@pytest.fixture
def pack_fields():
    """Minimal template arrays shared by the mixed-version cases."""
    return dict(
        templates=np.ones((1, 1, 3, 3)),
        offsets=np.arange(-1, 2),
        field_xy=np.zeros((1, 2)),
        wavelengths_nm=np.array([LAM_REF]),
    )


def test_producer_requires_centroid_origin(pack_fields):
    """Construction rejects producers that omit or misdeclare the origin."""
    with pytest.raises(TypeError, match="template_origin"):
        PsfletPack(**pack_fields)
    with pytest.raises(ValueError, match=r"origin.*regenerate"):
        PsfletPack(**pack_fields, template_origin="origin")


@pytest.mark.parametrize("origin", [None, "origin"])
def test_format_two_refuses_legacy_template(tmp_path, pack_fields, origin):
    """Format 2 requires an explicit centroid origin before loading templates."""
    if origin is not None:
        pack_fields["template_origin"] = origin
    path = tmp_path / "legacy.npz"
    np.savez(path, format_version=np.int64(2), **pack_fields)
    with pytest.raises(ValueError, match=r"origin.*regenerate"):
        load_psflet_pack(path)


def test_explicit_pack_refuses_legacy_template(pack_fields):
    """An in-memory pack must meet the same origin contract as a file."""
    pack = replace(PsfletPack(**pack_fields, template_origin="centroid"))
    object.__setattr__(pack, "template_origin", "origin")
    with pytest.raises(ValueError, match=r"origin.*regenerate"):
        _resolve_psflet_pack(None, pack)


def test_template_weights_refuse_legacy_template(pack_fields):
    """The public weight function applies the same origin contract."""
    from coronachrome.templates import template_weights

    pack = PsfletPack(**pack_fields, template_origin="centroid")
    object.__setattr__(pack, "template_origin", "origin")
    zeros = jnp.zeros((1, 1))
    with pytest.raises(ValueError, match=r"origin.*regenerate"):
        template_weights(pack, zeros, zeros, LAM_REF, jnp.zeros(1, dtype=int), 0.0)
