"""Contract tests for coronachrome.viz: geometry drawn is the operator's."""

import subprocess
import sys

import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest
from optixstuff.disperser import LensletDisperser

matplotlib.use("Agg")

# Imported directly, not through importorskip: the test extra carries the viz
# extra, so a missing plotting stack must fail loudly rather than skip.
import eyepiece as ep
import matplotlib.pyplot as plt

from coronachrome import (
    IFSRenderer,
    analytic_psflet_pack,
    build_ir,
    viz,
)
from coronachrome.build import (
    detector_centroids,
    lenslet_cell_centers,
)

FP_SHAPE = (40, 48)  # (ny, nx), deliberately non-square
DET_SHAPE = (96, 128)
LAM = jnp.array([600.0, 630.0, 660.0, 690.0, 720.0])
FP_PX = 5.0


def _disperser(**overrides):
    """A LensletDisperser with test defaults."""
    kwargs = dict(
        pitch_m=174e-6,
        pixsize_m=13e-6,
        angle_rad=float(np.arctan(0.5)),
        lam_ref_nm=660.0,
        pix_per_reselt=2.0,
        dispersion_coeffs=jnp.array([140.0, 0.0]),
        psflet_params=jnp.array([1.0]),
        psflet_ref_nm=660.0,
        grid_kind="square",
        n_lenslets=5,
        psflet_kind="gaussian",
        detector_shape=DET_SHAPE,
    )
    kwargs.update(overrides)
    return LensletDisperser(**kwargs)


@pytest.fixture(scope="module")
def setup():
    """A small square-grid disperser and its IR on non-square planes."""
    disperser = _disperser()
    ir = build_ir(disperser, LAM, FP_SHAPE, fp_px_per_lenslet=FP_PX)
    return disperser, ir


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test opened."""
    yield
    plt.close("all")


def _by_label(artists, label):
    """The single artist carrying ``label``."""
    matches = [a for a in artists if a.get_label() == label]
    assert len(matches) == 1, [a.get_label() for a in artists]
    return matches[0]


# -- import mechanics -------------------------------------------------------


def _run(code):
    """Run a snippet in a fresh interpreter."""
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_base_install_imports_without_eyepiece():
    """The base package and its viz package import with eyepiece blocked."""
    result = _run(
        "import sys; sys.modules['eyepiece'] = None\n"
        "import coronachrome\n"
        "import coronachrome.viz\n"
        "assert 'matplotlib' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stderr


def test_missing_eyepiece_names_the_viz_extra():
    """Touching a plot function without eyepiece names the viz extra."""
    result = _run(
        "import sys; sys.modules['eyepiece'] = None\n"
        "import coronachrome.viz\n"
        "try:\n"
        "    coronachrome.viz.plot_traces\n"
        "except ImportError as err:\n"
        "    assert 'coronachrome[viz]' in str(err), str(err)\n"
        "else:\n"
        "    raise SystemExit('expected ImportError')\n"
    )
    assert result.returncode == 0, result.stderr


def test_top_level_package_does_not_import_viz():
    """Importing coronachrome does not import the viz package."""
    result = _run(
        "import sys, coronachrome; assert 'coronachrome.viz' not in sys.modules"
    )
    assert result.returncode == 0, result.stderr


def test_dir_lists_the_exports():
    """dir() advertises the lazily exported plot functions."""
    assert {
        "plot_channel_covariance",
        "plot_lenslet_cells",
        "plot_traces",
    } <= set(dir(viz))


# -- the ax-first contract --------------------------------------------------


def _calls(setup):
    """One call per view, each drawing into the given axes."""
    disperser, ir = setup
    cov = np.eye(LAM.shape[0]) + 0.3 * np.eye(LAM.shape[0], k=1)
    cov = cov + cov.T
    return [
        lambda ax: viz.plot_lenslet_cells(
            disperser, FP_SHAPE, fp_px_per_lenslet=FP_PX, channels=(12,), ax=ax
        ),
        lambda ax: viz.plot_traces(ir, disperser, LAM, channels=(12, 17), ax=ax),
        lambda ax: viz.plot_channel_covariance(cov, wavelengths_nm=LAM, ax=ax),
    ]


@pytest.mark.parametrize("which", [0, 1, 2])
def test_draws_on_supplied_ax_without_global_state_change(setup, which):
    """A handed-in axes is drawn into and nothing global changes."""
    fig, axes = plt.subplots(1, 2, layout="constrained")
    widths = [a.get_position(original=True).width for a in axes]
    rc_before = dict(matplotlib.rcParams)
    figs_before = plt.get_fignums()
    plt.sca(axes[1])

    result = _calls(setup)[which](axes[0])

    assert isinstance(result, ep.PlotResult)
    assert result.ax is axes[0]
    assert result.fig is fig
    assert set(result.artists) <= ep.ARTIST_KEYS
    assert plt.get_fignums() == figs_before
    assert plt.gca() is axes[1]
    assert dict(matplotlib.rcParams) == rc_before
    assert [a.get_position(original=True).width for a in axes] == widths


@pytest.mark.parametrize("which", [0, 1, 2])
def test_creates_its_own_figure_when_no_ax(setup, which):
    """With no axes, each view creates and returns its own figure."""
    result = _calls(setup)[which](None)
    assert isinstance(result, ep.PlotResult)
    assert result.ax.figure is result.fig


# -- entrance plane: cells and origins ---------------------------------------


def test_lenslet_grid_origin_is_drawn_at_half_the_cube_shape(setup):
    """The grid origin sits at (nx/2, ny/2); the optical center where stated."""
    disperser, _ = setup
    center = (21.25, 17.5)
    result = viz.plot_lenslet_cells(
        disperser, FP_SHAPE, fp_px_per_lenslet=FP_PX, optical_center_px=center
    )
    grid = _by_label(result.artists["lines"], "lenslet-grid origin")
    optical = _by_label(result.artists["lines"], "optical center")
    ny, nx = FP_SHAPE
    assert np.allclose(np.ravel(grid.get_data()), [nx / 2.0, ny / 2.0])
    assert np.allclose(np.ravel(optical.get_data()), center)


def test_no_optical_center_is_invented(setup):
    """Without optical_center_px no optical center is drawn."""
    disperser, _ = setup
    result = viz.plot_lenslet_cells(disperser, FP_SHAPE, fp_px_per_lenslet=FP_PX)
    labels = [a.get_label() for a in result.artists["lines"]]
    assert labels == ["lenslet-grid origin"]


def test_drawn_cells_are_the_cells_the_operator_integrates(setup):
    """Each outlined cell has the area and centroid of its spatial footprint."""
    disperser, ir = setup
    channels = (0, 7, 12, 24)
    result = viz.plot_lenslet_cells(
        disperser, FP_SHAPE, fp_px_per_lenslet=FP_PX, channels=channels
    )
    nx = FP_SHAPE[1]
    for ch, patch in zip(channels, result.artists["ellipse"], strict=True):
        xy = patch.get_xy()[:4]
        x, y = xy[:, 0], xy[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        src = np.asarray(ir.spatial_src[ch])
        w = np.asarray(ir.spatial_w[ch])
        sy, sx = np.divmod(src, nx)
        assert area == pytest.approx(float(w.sum()), rel=1e-6)
        assert xy.mean(axis=0) == pytest.approx(
            [float((w * sx).sum() / w.sum()), float((w * sy).sum() / w.sum())],
            abs=0.15,  # the operator bins supersampled subpoints to pixels
        )
        edge = xy[1] - xy[0]
        assert np.arctan2(edge[1], edge[0]) == pytest.approx(disperser.angle_rad)


def test_every_cell_outline_is_drawn_in_channel_order(setup):
    """One outline per channel, centered on the library's cell centers."""
    disperser, ir = setup
    result = viz.plot_lenslet_cells(disperser, FP_SHAPE, fp_px_per_lenslet=FP_PX)
    paths = result.artists["collection"].get_paths()
    assert len(paths) == ir.n_channels
    cx, cy = lenslet_cell_centers(disperser, FP_SHAPE, FP_PX)
    centers = np.array([p.vertices[:4].mean(axis=0) for p in paths])
    assert np.allclose(centers, np.stack([cx, cy], axis=1))


def test_bare_centers_door_matches_the_disperser_door(setup):
    """Bare cell centers draw the same cells as the descriptor."""
    disperser, _ = setup
    cx, cy = lenslet_cell_centers(disperser, FP_SHAPE, FP_PX)
    a = viz.plot_lenslet_cells(
        disperser, FP_SHAPE, fp_px_per_lenslet=FP_PX, channels=(3,)
    )
    b = viz.plot_lenslet_cells(
        np.stack([cx, cy], axis=1),
        FP_SHAPE,
        fp_px_per_lenslet=FP_PX,
        angle_rad=disperser.angle_rad,
        channels=(3,),
    )
    assert np.allclose(
        a.artists["ellipse"][0].get_xy(), b.artists["ellipse"][0].get_xy()
    )
    with pytest.raises(ValueError, match="angle_rad"):
        viz.plot_lenslet_cells(np.stack([cx, cy], axis=1), FP_SHAPE)


def test_cube_image_is_band_summed_or_one_plane(setup):
    """A cube is band-summed by default, or one plane is drawn."""
    disperser, _ = setup
    cube = np.ones((3, *FP_SHAPE)) * np.array([1.0, 2.0, 4.0])[:, None, None]
    summed = viz.plot_lenslet_cells(
        disperser, FP_SHAPE, fp_px_per_lenslet=FP_PX, image=cube
    )
    plane = viz.plot_lenslet_cells(
        disperser, FP_SHAPE, fp_px_per_lenslet=FP_PX, image=cube, plane=2
    )
    assert np.allclose(summed.artists["image"].get_array(), 7.0)
    assert np.allclose(plane.artists["image"].get_array(), 4.0)


# -- detector plane: traces, centroids, origin --------------------------------


def test_trace_origin_is_drawn_at_half_the_detector_shape(setup):
    """The trace origin sits at (nx/2, ny/2) of the detector."""
    disperser, ir = setup
    result = viz.plot_traces(ir, disperser, LAM, channels=(12,))
    origin = _by_label(result.artists["lines"], "detector trace origin")
    ny, nx = DET_SHAPE
    assert np.allclose(np.ravel(origin.get_data()), [nx / 2.0, ny / 2.0])


def test_drawn_centroids_equal_the_library_centroids(setup):
    """Marked centroids equal detector_centroids for the same inputs."""
    disperser, ir = setup
    channels = (12, 17)
    marked = np.array([0, 2, 4])
    result = viz.plot_traces(ir, disperser, LAM, channels=channels, marked=marked)
    xc, yc = detector_centroids(disperser, LAM)
    expected = np.concatenate(
        [np.stack([xc[ch, marked], yc[ch, marked]], axis=1) for ch in channels]
    )
    assert np.allclose(result.artists["scatter"].get_offsets(), expected)


def test_centroids_follow_the_documented_dispersion_law(setup):
    """Lenslet (0, 0): x = nx/2 + c1 log(lambda/lam_ref), y = ny/2."""
    disperser, ir = setup
    center = 12  # lenslet-index (0, 0) on a 5 x 5 square grid
    result = viz.plot_traces(ir, disperser, LAM, channels=(center,))
    ny, nx = DET_SHAPE
    lam = np.asarray(LAM)
    expected_x = nx / 2.0 + 140.0 * np.log(lam / 660.0)
    offsets = result.artists["scatter"].get_offsets()
    assert np.allclose(offsets[:, 0], expected_x)
    assert np.allclose(offsets[:, 1], ny / 2.0)
    assert np.all(np.diff(offsets[:, 0]) > 0)  # longer wavelength, larger x


def test_template_correction_moves_markers_off_the_geometric_trace():
    """A calibrated centroid correction is applied once, to the markers only."""
    disperser = _disperser(psflet_kind="template")
    shift = np.zeros((1, LAM.shape[0], 2))
    shift[0, :, 1] = 0.75  # +0.75 px along y at every wavelength
    pack = analytic_psflet_pack(
        "gaussian", jnp.array([1.0]), LAM, psflet_ref_nm=660.0, centroids=shift
    )
    ir = build_ir(disperser, LAM, FP_SHAPE, fp_px_per_lenslet=FP_PX, psflet_pack=pack)
    result = viz.plot_traces(ir, disperser, LAM, channels=(12,), psflet_pack=pack)
    trace = _by_label(result.artists["lines"], "lenslet 12 trace")
    markers = result.artists["scatter"].get_offsets()
    assert np.allclose(markers[:, 0], trace.get_xdata())
    assert np.allclose(markers[:, 1] - trace.get_ydata(), 0.75)
    # the markers are where the operator placed the footprints
    xc, yc = detector_centroids(disperser, LAM, psflet_pack=pack)
    rows = np.asarray(ir.det_rows[12])
    center_pixel = rows[:, rows.shape[1] // 2]
    cy, cx = np.divmod(center_pixel, DET_SHAPE[1])
    assert np.array_equal(cx, np.round(np.asarray(xc[12])))
    assert np.array_equal(cy, np.round(np.asarray(yc[12])))


def test_footprint_image_is_the_forward_operator_applied_to_unit_spectra(setup):
    """The detector image is H applied to unit spectra in the channels."""
    disperser, ir = setup
    channels = (12, 13)
    result = viz.plot_traces(
        ir, disperser, LAM, channels=channels, floor=1e-30, vmin=1e-30
    )
    z = jnp.zeros((ir.n_channels, ir.n_wav)).at[jnp.array(channels)].set(1.0)
    expected = (IFSRenderer(ir).H_mono @ z.reshape(-1)).reshape(DET_SHAPE)
    drawn = np.asarray(result.artists["image"].get_array())
    assert np.allclose(drawn, np.clip(np.asarray(expected), 1e-30, None))


def test_bare_image_and_centroids_door(setup):
    """A bare image and centroid pair draw without an IR or descriptor."""
    disperser, _ = setup
    xc, yc = detector_centroids(disperser, LAM)
    image = np.random.default_rng(0).random(DET_SHAPE)
    result = viz.plot_traces(
        image, (xc, yc), LAM, channels=(12,), trace_origin_px=(64.0, 48.0)
    )
    origin = _by_label(result.artists["lines"], "detector trace origin")
    assert np.allclose(np.ravel(origin.get_data()), [64.0, 48.0])
    assert np.allclose(result.artists["scatter"].get_offsets()[:, 0], xc[12])


# -- the wavelength-scan updater --------------------------------------------


def _snapshot(result):
    """Everything the updater must leave unchanged."""
    ax = result.ax
    image = result.artists["image"]
    return {
        "image": np.array(image.get_array()),
        "clim": image.get_clim(),
        "xlim": ax.get_xlim(),
        "ylim": ax.get_ylim(),
        "lines": [np.array(ln.get_xydata()) for ln in result.artists["lines"]],
        "scatter": np.array(result.artists["scatter"].get_offsets()),
        "edge": _by_label(result.artists["ellipse"], "detector edge").get_bbox(),
        "n_children": len(ax.get_children()),
    }


def test_update_moves_only_the_declared_artists(setup):
    """update(k) moves the highlight, footprint box and readout only."""
    disperser, ir = setup
    ch = 12
    result = viz.plot_traces(ir, disperser, LAM, channels=(ch, 17), scan_index=0)
    before = _snapshot(result)
    readout = _by_label(result.artists["text"], "scan readout")
    box = _by_label(result.artists["ellipse"], "scan footprint")
    text_before = readout.get_text()

    result.update(3)

    xc, yc = detector_centroids(disperser, LAM)
    assert np.allclose(
        np.ravel(result.artists["line"].get_data()), [xc[ch, 3], yc[ch, 3]]
    )
    assert readout.get_text() != text_before and "690 nm" in readout.get_text()
    # the box frames the 7 x 7 footprint window (half = 3) around the centroid
    assert box.get_width() == pytest.approx(7.0)
    assert box.get_height() == pytest.approx(7.0)
    assert box.get_x() == pytest.approx(np.round(xc[ch, 3]) - 3.5)
    assert box.get_y() == pytest.approx(np.round(yc[ch, 3]) - 3.5)

    after = _snapshot(result)
    assert np.array_equal(after["image"], before["image"])
    assert after["clim"] == before["clim"]
    assert after["xlim"] == before["xlim"] and after["ylim"] == before["ylim"]
    for a, b in zip(after["lines"], before["lines"], strict=True):
        assert np.array_equal(a, b)
    assert np.array_equal(after["scatter"], before["scatter"])
    assert np.allclose(after["edge"].bounds, before["edge"].bounds)
    assert after["n_children"] == before["n_children"]


def test_scan_box_on_a_clipped_trace_stays_on_the_detector():
    """A footprint cut by the detector edge frames only its on-detector pixels.

    Lenslet-index (-2, 2) on a 60 x 80 detector runs off the left edge: its
    600 nm footprint is wholly off, its 630 nm footprint is cut, and its
    660 nm footprint is whole.
    """
    det_shape = (60, 80)
    disperser = _disperser(detector_shape=det_shape)
    with pytest.warns(UserWarning, match="fell off the detector"):
        ir = build_ir(disperser, LAM, FP_SHAPE, fp_px_per_lenslet=FP_PX)
    ch, half = 4, 3
    result = viz.plot_traces(ir, disperser, LAM, channels=(ch,), scan_index=0)
    box = _by_label(result.artists["ellipse"], "scan footprint")
    assert not box.get_visible()  # 600 nm: nothing reaches the detector

    xc, _ = detector_centroids(disperser, LAM)
    for k in (1, 2):
        result.update(k)
        assert box.get_visible()
        assert box.get_width() <= 2 * half + 1
        assert box.get_height() <= 2 * half + 1
        assert box.get_x() >= -0.5
        assert box.get_x() + box.get_width() <= det_shape[1] - 0.5
        right = np.round(float(xc[ch, k])) + half + 0.5
        assert box.get_x() + box.get_width() == pytest.approx(right)
    assert box.get_width() == 2 * half + 1  # 660 nm is whole
    result.update(0)
    assert not box.get_visible()


def test_no_scan_means_no_updater(setup):
    """Without scan_index there is no highlight and no updater."""
    disperser, ir = setup
    result = viz.plot_traces(ir, disperser, LAM, channels=(12,))
    assert result.update is None
    assert "line" not in result.artists


def test_update_records_through_eyepiece(setup, tmp_path):
    """The updater drives an eyepiece recording frame by frame."""
    disperser, ir = setup
    result = viz.plot_traces(ir, disperser, LAM, channels=(12,), scan_index=0)
    path = tmp_path / "scan.gif"
    with ep.record(result.fig, str(path), fps=5, dpi=40) as rec:
        for k in range(LAM.shape[0]):
            result.update(k)
            rec.frame()
    assert path.stat().st_size > 0


# -- channel covariance -------------------------------------------------------


def test_correlation_view_normalizes_and_pins_the_scale():
    """Correlation is cov / sqrt(d d^T) on a norm fixed to [-1, 1]."""
    cov = np.array([[4.0, -1.0], [-1.0, 1.0]])
    result = viz.plot_channel_covariance(cov, wavelengths_nm=[600.0, 650.0])
    drawn = np.asarray(result.artists["image"].get_array())
    assert np.allclose(drawn, [[1.0, -0.5], [-0.5, 1.0]])
    assert result.artists["image"].get_clim() == (-1.0, 1.0)
    labels = [t.get_text() for t in result.ax.get_xticklabels()]
    assert labels == ["600", "650"]


def test_covariance_view_selects_a_block_and_updates_under_one_norm():
    """A stacked block is selected and updates keep the first norm."""
    stack = np.stack([np.eye(3) * 2.0, np.eye(3) * 5.0])
    result = viz.plot_channel_covariance(stack, block=1, correlation=False)
    assert result.artists["image"].get_clim() == (-5.0, 5.0)
    result.update(stack * 10.0)
    assert result.artists["image"].get_clim() == (-5.0, 5.0)
    assert np.allclose(np.diagonal(result.artists["image"].get_array()), 50.0)
    with pytest.raises(ValueError, match="block"):
        viz.plot_channel_covariance(stack)


def test_multi_lenslet_covariance_draws_block_separators():
    """A multi-lenslet matrix gets block separators and labels."""
    lam = [600.0, 650.0, 700.0]
    cov = np.eye(6) + 0.2 * np.eye(6, k=3) + 0.2 * np.eye(6, k=-3)
    result = viz.plot_channel_covariance(
        cov, wavelengths_nm=lam, channel_labels=["lenslet 12", "lenslet 13"]
    )
    positions = sorted(
        {float(np.ravel(ln.get_xdata())[0]) for ln in result.artists["lines"]}
        | {float(np.ravel(ln.get_ydata())[0]) for ln in result.artists["lines"]}
    )
    assert 2.5 in positions
    assert [t.get_text() for t in result.artists["text"]] == [
        "lenslet 12",
        "lenslet 13",
    ]
    with pytest.raises(ValueError, match="multiple"):
        viz.plot_channel_covariance(np.eye(5), wavelengths_nm=lam)


def test_covariance_orientation_is_row_up_column_right():
    """Entry [i, j] is drawn at x = j, y = i, and each label sits on its block.

    An asymmetric matrix pins the orientation: a transpose or a flipped
    origin would move the one nonzero off-diagonal entry.
    """
    lam = [600.0, 650.0, 700.0]
    cov = np.eye(6)
    cov[4, 1] = 0.5  # row 4 (lenslet B, 650 nm), column 1 (lenslet A, 650 nm)
    result = viz.plot_channel_covariance(
        cov, wavelengths_nm=lam, correlation=False, channel_labels=["A", "B"]
    )
    image = result.artists["image"]
    assert np.array_equal(np.asarray(image.get_array()), cov)
    assert image.origin == "lower"
    x0, x1, y0, y1 = image.get_extent()
    assert x0 < x1 and y0 < y1
    ylim = result.ax.get_ylim()
    assert ylim[0] < ylim[1]
    for b, text in enumerate(result.artists["text"]):
        x, y = text.get_position()
        assert b * 3 - 0.5 < x < (b + 1) * 3 - 0.5
        assert b * 3 - 0.5 < y < (b + 1) * 3 - 0.5


def test_spectrum_covariance_output_plots_directly(setup):
    """spectrum_covariance output plots with unit diagonal correlation."""
    from coronachrome import spectrum_covariance

    _, ir = setup
    cov = spectrum_covariance(IFSRenderer(ir), channels=jnp.array([12]))
    result = viz.plot_channel_covariance(cov, block=0, wavelengths_nm=LAM)
    drawn = np.asarray(result.artists["image"].get_array())
    assert np.allclose(np.diagonal(drawn), 1.0)
