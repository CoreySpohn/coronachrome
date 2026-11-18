"""Tests for the IFS renderer forward model."""

from itertools import pairwise

import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import bands_at, focal_image, sky_grid, square_rectangle_area
from hwoutils.grids import Measure
from optixstuff.disperser import LensletDisperser

from coronachrome.build import build_ir
from coronachrome.render import IFSRenderer, spatial_sample


def _renderer(n=6, n_wav=5, fp=(64, 64), *, grid_kind="square", step=0.1):
    disp = LensletDisperser(
        pitch_m=174e-6,
        sky_pitch_arcsec=0.2,
        pixsize_m=13e-6,
        angle_rad=float(jnp.arcsin(1.0 / jnp.sqrt(5.0))),
        lam_ref_nm=660.0,
        pix_per_reselt=2.0,
        dispersion_coeffs=jnp.array([100.0, 0.0]),
        psflet_params=jnp.array([0.7]),
        psflet_ref_nm=660.0,
        grid_kind=grid_kind,
        n_lenslets=n,
        psflet_kind="gaussian",
        detector_shape=(256, 256),
    )
    lam = jnp.linspace(640.0, 680.0, n_wav)
    ir = build_ir(disp, fp_grid=sky_grid(fp, step), bands=bands_at(lam))
    return IFSRenderer(ir), ir, fp, n_wav


def test_forward_spmv_shape_and_flux():
    """forward_spmv returns a detector-shaped map with positive total flux."""
    r, ir, fp, n_wav = _renderer()
    cube = jnp.ones((n_wav, fp[0], fp[1]))
    det = r.forward_spmv(focal_image(cube, r.ir.fp_grid)).data
    assert det.shape == ir.det_shape
    assert float(det.sum()) > 0.0


def test_h_mono_matches_spmv():
    """forward_spmv equals the spatial step followed by an H_mono matvec."""
    r, ir, fp, n_wav = _renderer()
    cube = jnp.ones((n_wav, fp[0], fp[1]))
    z = spatial_sample(focal_image(cube, ir.fp_grid), ir)
    direct = (r.H_mono @ z.reshape(-1)).reshape(ir.det_shape)
    assert jnp.allclose(direct, r.forward_spmv(focal_image(cube, r.ir.fp_grid)).data)


def test_streaming_matches_spmv():
    """Streaming scatter-add forward equals the BCOO spmv forward."""
    r, _ir, fp, n_wav = _renderer()
    key_cube = jnp.arange(n_wav * fp[0] * fp[1], dtype=float).reshape(
        n_wav, fp[0], fp[1]
    )
    cube = jnp.sin(key_cube)
    assert jnp.allclose(
        r.forward_streaming(focal_image(cube, r.ir.fp_grid)).data,
        r.forward_spmv(focal_image(cube, r.ir.fp_grid)).data,
        atol=1e-9,
    )


def test_adjoint_dot_product_identity():
    """The adjoint satisfies the inner-product identity <H z, y> == <z, H^T y>."""
    r, ir, _fp, _n_wav = _renderer()
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    z = jax.random.normal(k1, (ir.n_channels, ir.n_wav))
    y = jax.random.normal(k2, ir.det_shape)
    hz = r.H_mono @ z.reshape(-1)
    hty = r.channel_spectra(r.adjoint(r.detector_image(y, "e/s")))
    assert jnp.allclose(
        jnp.dot(hz, y.reshape(-1)),
        jnp.dot(z.reshape(-1), hty.reshape(-1)),
        atol=1e-8,
    )


def test_forward_is_differentiable():
    """forward_spmv has a finite, nonzero gradient w.r.t. the input cube."""
    r, ir, fp, n_wav = _renderer()
    target = jnp.ones(ir.det_shape)

    def misfit(cube):
        return jnp.sum(
            (r.forward_spmv(focal_image(cube, r.ir.fp_grid)).data - target) ** 2
        )

    cube0 = jnp.zeros((n_wav, fp[0], fp[1]))
    g = jax.grad(misfit)(cube0)
    assert g.shape == cube0.shape
    assert float(jnp.linalg.norm(g)) > 0.0


@vv.case("ifs-geometry", "code-verification")
def test_end_to_end_point_source_lands_on_detector():
    """Fails on axis, sky parity, or half-lenslet errors; typed centres, FP floor.

    Sky (E,N)=(-.08,+.12) at PA=90, parity=-1 maps to (x,y)=(+.12,-.08).
    Nearest lenslet (+.1,-.1) is channel 2 (x-major). Its detector trace is
    (29.5+5+2,19.5-5)=(36.5,14.5), with constant +2px dispersion.
    """
    from hwoutils.grids import SkyOrientation
    from hwoutils.spectral import Band

    from coronachrome.build import lenslet_cell_centers

    grid = sky_grid((21, 25), 0.02, orientation=SkyOrientation(90, -1))
    disp = LensletDisperser(
        pitch_m=130e-6,
        pixsize_m=13e-6,
        angle_rad=0.0,
        sky_pitch_arcsec=0.2,
        lam_ref_nm=660.0,
        pix_per_reselt=2.0,
        dispersion_coeffs=jnp.array([2.0]),
        psflet_params=jnp.array([0.15]),
        psflet_ref_nm=660.0,
        grid_kind="square",
        n_lenslets=2,
        psflet_kind="gaussian",
        detector_shape=(40, 60),
    )
    r = IFSRenderer(build_ir(disp, (Band(659, 661),), grid))
    xy = grid.orientation.sky_to_detector(-0.08, 0.12)
    pixel = grid.unit_to_pixel(*xy)
    assert jnp.allclose(jnp.array(pixel), jnp.array([18.0, 6.0]), atol=3e-15, rtol=0)
    cube = focal_image(jnp.zeros((1, 21, 25)).at[0, 6, 18].set(1), grid)
    cx, cy = lenslet_cell_centers(disp, grid)
    assert int(jnp.argmin((cx - 18) ** 2 + (cy - 6) ** 2)) == 2
    assert jnp.allclose(
        spatial_sample(cube, r.ir)[:, 0],
        jnp.array([0.0, 0.0, 1.0, 0.0]),
        atol=1e-15,
        rtol=0,
    )
    detector = r.forward_spmv(cube)
    yy, xx = jnp.indices(detector.grid.shape)
    assert jnp.allclose(detector.data.sum(), 1, atol=2e-15, rtol=0)
    assert jnp.allclose(
        jnp.array([(xx * detector.data).sum(), (yy * detector.data).sum()]),
        jnp.array([36.5, 14.5]),
        atol=1e-12,
        rtol=0,
    )
    assert (
        r.channel_cube(jnp.array([[0.0], [0.0], [1.0], [0.0]]), "e/s").data[0, 0, 1]
        == 1
    )


def test_hex_grid_pipeline_streaming_matches_spmv():
    """A hex-grid disperser builds and renders; streaming equals spmv."""
    disp = LensletDisperser(
        pitch_m=174e-6,
        sky_pitch_arcsec=0.2,
        pixsize_m=13e-6,
        angle_rad=float(jnp.arcsin(1.0 / jnp.sqrt(5.0))),
        lam_ref_nm=660.0,
        pix_per_reselt=2.0,
        dispersion_coeffs=jnp.array([100.0, 0.0]),
        psflet_params=jnp.array([0.7]),
        psflet_ref_nm=660.0,
        grid_kind="hex",
        n_lenslets=6,
        psflet_kind="gaussian",
        detector_shape=(256, 256),
    )
    lam = jnp.linspace(640.0, 680.0, 5)
    r = IFSRenderer(
        build_ir(disp, fp_grid=sky_grid((64, 64), 0.2 / (2.0)), bands=bands_at(lam))
    )
    # Square collection cells retain area .2**2, despite axial output packing.
    density = focal_image(
        jnp.full((5, 64, 64), 100.0), r.ir.fp_grid, measure=Measure.density("grid")
    )
    assert jnp.allclose(spatial_sample(density, r.ir), 4.0, atol=2e-14, rtol=0)
    cube = jnp.sin(jnp.arange(5 * 64 * 64, dtype=float).reshape(5, 64, 64))
    assert jnp.allclose(
        r.forward_streaming(focal_image(cube, r.ir.fp_grid)).data,
        r.forward_spmv(focal_image(cube, r.ir.fp_grid)).data,
        atol=1e-9,
    )


@vv.case("ifs-geometry", "code-verification")
@pytest.mark.parametrize("n,grid_kind", [(6, "square"), (5, "square"), (5, "hex")])
def test_spatial_sample_integrates_cell_area(n, grid_kind):
    """Fails if rotated collection uses deposition instead of aperture area.

    Independent half-plane vertex enumeration checks every clipped edge;
    interior cells have area .04 arcsec^2. Float64 floor: 128 eps per cell.
    Even/odd square origins and staggered square apertures are exercised.
    """
    angle = np.arcsin(1 / np.sqrt(5))
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    # Derive channel centres without production layout/geometry helpers.
    if grid_kind == "square":
        centres = [
            (x - (n - 1) / 2, y - (n - 1) / 2) for x in range(n) for y in range(n)
        ]
    else:
        centres = [
            (
                x - (n - 1) / 2 + (y % 2) / 2 - (n // 2) / (2 * n),
                (y - (n - 1) / 2) * np.sqrt(3) / 2,
            )
            for y in range(n)
            for x in range(n)
        ]
    centres = 0.2 * np.array(centres) @ rotation.T
    for size in (64, 2 * n):
        if size == 64:
            _, ir, _, _ = _renderer(n, 2, (size, size), grid_kind=grid_kind)
        else:
            with pytest.warns(UserWarning, match="extend past"):
                _, ir, _, _ = _renderer(n, 2, (size, size), grid_kind=grid_kind)
        half_width = size * 0.1 / 2
        areas = np.array(
            [
                square_rectangle_area(
                    c, angle, 0.2, (-half_width, half_width, -half_width, half_width)
                )
                for c in centres
            ]
        )
        interior = (
            np.max(np.abs(centres), axis=1) + 0.1 * (np.cos(angle) + np.sin(angle))
            < half_width
        )
        assert interior.any()
        np.testing.assert_allclose(areas[interior], 0.04, rtol=0, atol=1e-16)
        if size != 64:
            assert (~interior).any()
        data = jnp.ones((2, size, size)) * jnp.array([1.0, 2.0])[:, None, None]
        expected = areas[:, None] / 0.01 * np.array([1.0, 2.0])
        tol = 128 * np.finfo(float).eps
        for measure, scale in ((Measure.integral(), 1), (Measure.density("grid"), 100)):
            z = spatial_sample(
                focal_image(data * scale, ir.fp_grid, measure=measure), ir
            )
            np.testing.assert_allclose(z, expected, rtol=tol, atol=tol)
        if size == 64:
            # Restores the original rotated 6x6 fixture's first lenslet to 4.
            np.testing.assert_allclose(z[:, 0], 4.0, rtol=0, atol=tol)


@vv.case("ifs-geometry", "code-verification")
def test_spatial_sample_pixel_overlap_and_covered_flux():
    """Fails if a lit pixel is assigned to centres or edge weights renormalize.

    A rotated 6x6 tiling fully covers this 6x6 focal field. Independently
    intersect the lit pixel with each square; sum arbitrary positive flux
    over the entire covered field. Tolerance is the float64 accumulation floor.
    """
    with pytest.warns(UserWarning, match="extend past"):
        _, ir, _, _ = _renderer(6, 2, (6, 6))
    angle = np.arcsin(1 / np.sqrt(5))
    ca, sa = np.cos(angle), np.sin(angle)
    centres = [
        0.2 * np.array([ca * x - sa * y, sa * x + ca * y])
        for x in np.arange(6) - 2.5
        for y in np.arange(6) - 2.5
    ]
    # Pixel (row 2, col 3): x in [0,.1], y in [-.1,0].
    expected = (
        np.array(
            [square_rectangle_area(c, angle, 0.2, (0, 0.1, -0.1, 0)) for c in centres]
        )
        / 0.01
    )
    assert np.count_nonzero(expected > 1e-14) > 1
    data = jnp.stack(
        [jnp.zeros((6, 6)).at[2, 3].set(1), jnp.arange(1.0, 37.0).reshape(6, 6)]
    )
    z = spatial_sample(focal_image(data, ir.fp_grid), ir)
    np.testing.assert_allclose(
        z[:, 0], expected, rtol=0, atol=128 * np.finfo(float).eps
    )
    np.testing.assert_allclose(
        z.sum(axis=0), [1.0, 666.0], rtol=128 * np.finfo(float).eps, atol=0
    )


@vv.case(
    "ifs-geometry", "solution-verification", refined_parameter="focal pixel spacing"
)
def test_band_limited_cell_integral_error():
    """Fails if the constant-pixel approximation does not converge to the DFT integral.

    A sigma=.2 arcsec Gaussian sampled on a 3.2 arcsec field is zero-padded
    by 2, matching the shared band_limited convention. Integrate every DFT
    mode analytically over a rotated .2 arcsec square using two sinc factors.
    Three spacings measure convergence; refinement must meet a 1% budget,
    not a universal bound for band-limited inputs (especially near Nyquist).
    """
    angle = np.arcsin(1 / np.sqrt(5))
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    centres = (
        0.2
        * np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
        @ rotation.T
    )
    errors = []
    for size in (32, 64, 128):
        step = 3.2 / size
        _, ir, _, _ = _renderer(2, 1, (size, size), step=step)
        coords = (np.arange(size) - (size - 1) / 2) * step
        density = np.exp(-(coords[:, None] ** 2 + coords[None, :] ** 2) / (2 * 0.2**2))
        cube = focal_image(
            jnp.asarray(density[None] * step**2),
            ir.fp_grid,
            reconstruction="band_limited",
        )
        sampled = np.asarray(spatial_sample(cube, ir))[:, 0]
        padded = 2 * size
        coeff = np.fft.fft2(density, s=(padded, padded)) / padded**2
        freq = np.fft.fftfreq(padded, d=step)
        ky, kx = np.meshgrid(freq, freq, indexing="ij")
        # Real cosine Nyquist convention is immaterial here: Gaussian modes
        # at Nyquist are below roundoff at all three spacings.
        aperture = 0.04 * np.sinc(0.2 * (kx * rotation[0, 0] + ky * rotation[1, 0]))
        aperture *= np.sinc(0.2 * (kx * rotation[0, 1] + ky * rotation[1, 1]))
        exact = np.array(
            [
                np.sum(
                    coeff
                    * aperture
                    * np.exp(
                        2j
                        * np.pi
                        * (
                            kx * (c[0] + (size - 1) * step / 2)
                            + ky * (c[1] + (size - 1) * step / 2)
                        )
                    )
                ).real
                for c in centres
            ]
        )
        errors.append(np.max(np.abs(sampled / exact - 1)))
    print("band_limited max relative cell errors:", errors)
    assert errors[1] < 0.01
    assert all(2.5 < a / b < 6 for a, b in pairwise(errors))
