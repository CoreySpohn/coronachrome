"""Lenslet collection cells and detector traces: extract, delegate, decorate.

coronachrome supplies what eyepiece cannot know: where each lenslet's
collection cell sits on the entrance plane, where each (lenslet, wavelength)
PSFlet is centered on the detector, which detector pixels its footprint
covers, and where the three reference points of the geometry lie. The pixels
themselves are drawn by ``eyepiece.imshow_log``.

Coordinates are pixel-center coordinates on both planes: pixel
``(row, column)`` is centered on ``(x, y) = (column, row)``, and images are
drawn with ``origin="lower"`` and edge extents, so a marker at ``(x, y)``
sits on the pixel it names.

The three reference points are kept separate because they are separate
quantities:

- the entrance **optical center** belongs to whoever produced the cube, so it
  is drawn only where the caller states it (``optical_center_px``);
- the **lenslet-grid origin** is where lenslet-index ``(0, 0)`` sits on the
  entrance plane, ``(fp_shape[1] / 2, fp_shape[0] / 2)``;
- the **detector trace origin** is where that origin lands on the detector
  at zero dispersion offset, ``(detector_shape[1] / 2, detector_shape[0] / 2)``.
"""

import jax.numpy as jnp
import numpy as np
from optixstuff.disperser import LensletDisperser

from coronachrome.build import (
    _resolve_fp_px_per_lenslet,
    detector_centroids,
    detector_trace_origin,
    lenslet_cell_centers,
)
from coronachrome.ir import SpatialChannelIR
from coronachrome.viz._require import eyepiece


def _neutral(level):
    """A tone ``level`` of the way from the axes facecolor to the text color.

    Resolved from the active rcParams at call time, so reference marks invert
    with the style mode instead of freezing one gray for both.
    """
    import matplotlib as mpl
    from matplotlib.colors import to_rgb

    face = np.asarray(to_rgb(mpl.rcParams["axes.facecolor"]))
    text = np.asarray(to_rgb(mpl.rcParams["text.color"]))
    return tuple(face + level * (text - face))


def _halo(linewidth=2.5):
    """A background-colored stroke that keeps labels and marks legible.

    Reference marks take the text color, so over the bright end of an image
    they need an outline in the background color to stay visible.
    """
    import matplotlib as mpl
    from matplotlib import patheffects

    face = mpl.rcParams["axes.facecolor"]
    return [patheffects.withStroke(linewidth=linewidth, foreground=face)]


def _source_name(channel):
    """The source name a highlighted lenslet is registered under."""
    return f"lenslet {int(channel)}"


def _styles_for(ep, channels, styles):
    """The caller's ``SourceStyles``, or one declared from ``channels``."""
    if styles is not None:
        return styles
    return ep.SourceStyles([_source_name(c) for c in channels])


def _edge_extent(shape):
    """``(left, right, bottom, top)`` pixel-edge extent of a ``(ny, nx)`` grid."""
    ny, nx = shape
    return (-0.5, nx - 0.5, -0.5, ny - 0.5)


def _cell_corners(cx, cy, cell_px, angle_rad):
    """Corners ``(n, 4, 2)`` of square cells of side ``cell_px`` rotated by angle.

    The same cell :func:`coronachrome.build_ir` integrates the cube over: a
    square about the lenslet center, rotated by the lenslet angle.
    """
    half = 0.5 * cell_px
    local = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    rx = ca * local[:, 0] - sa * local[:, 1]
    ry = sa * local[:, 0] + ca * local[:, 1]
    cx = np.asarray(cx, dtype=float)[:, None]
    cy = np.asarray(cy, dtype=float)[:, None]
    return np.stack([cx + rx, cy + ry], axis=-1)


def _log_image(ep, image, ax, extent, floor, vmin, vmax, cmap, cbar_label, kw, rel):
    """Draw ``image`` with ``imshow_log``, flooring at ``rel`` of its peak."""
    peak = float(np.nanmax(image))
    if floor is None:
        floor = rel * peak if peak > 0 else 1e-20
    return ep.imshow_log(
        image,
        ax=ax,
        extent=extent,
        floor=floor,
        vmin=floor if vmin is None else vmin,
        vmax=vmax,
        cmap=cmap,
        colorbar=kw["colorbar"],
        cbar_label=cbar_label,
        imshow_kw=kw["imshow_kw"],
        cbar_kw=kw["cbar_kw"],
    )


def _origin_marker(ax, xy, marker, label, text_offset_pt, color, artists):
    """Draw one labeled reference point and file its artists."""
    (line,) = ax.plot(
        [xy[0]],
        [xy[1]],
        linestyle="none",
        marker=marker,
        markersize=9,
        markeredgewidth=1.8,
        markerfacecolor="none",
        markeredgecolor=color,
        label=label,
        path_effects=_halo(4.0),
        zorder=6,
    )
    text = ax.annotate(
        label,
        xy,
        xytext=text_offset_pt,
        textcoords="offset points",
        color=color,
        fontsize="small",
        ha="left" if text_offset_pt[0] >= 0 else "right",
        va="bottom" if text_offset_pt[1] >= 0 else "top",
        path_effects=_halo(),
        zorder=6,
    )
    artists.setdefault("lines", []).append(line)
    artists.setdefault("text", []).append(text)


def plot_lenslet_cells(
    disperser_or_centers,
    fp_shape,
    *,
    fp_px_per_lenslet=None,
    fp_pixel_scale_arcsec=None,
    angle_rad=None,
    image=None,
    plane=None,
    channels=(),
    optical_center_px=None,
    grid_origin_px=None,
    styles=None,
    window=None,
    floor=None,
    colorbar=True,
    ax=None,
    imshow_kw=None,
    cbar_kw=None,
    cell_kw=None,
):
    """Draw the lenslet collection cells over the entrance plane.

    Every lenslet's square collection cell is outlined over the focal-plane
    cube grid (optionally over an image of the cube, in the ``intensity``
    colormap on a log norm, raw pixels), the ``channels`` cells are outlined
    in their source colors and labeled, and the reference points are marked
    with distinct shapes and labels: the optical center (an x, only when
    ``optical_center_px`` is given) and the lenslet-grid origin (a square).

    The cells are the ones :func:`coronachrome.build_ir` integrates: squares
    of side ``fp_px_per_lenslet`` rotated by the lenslet angle. On a
    ``grid_kind="hex"`` descriptor those squares do not tile the plane, and
    the drawing shows that rather than hiding it.

    Args:
        disperser_or_centers: A ``LensletDisperser`` (cell centers and the
            grid origin are derived with
            :func:`coronachrome.build.lenslet_cell_centers`), or bare
            ``(n, 2)`` cell centers ``(x, y)`` in cube pixels, which then need
            ``fp_px_per_lenslet`` and ``angle_rad``.
        fp_shape: Focal-plane cube ``(ny, nx)``.
        fp_px_per_lenslet: Cube pixels per lenslet pitch (the override, as in
            ``build_ir``). With a disperser, pass this or
            ``fp_pixel_scale_arcsec``.
        fp_pixel_scale_arcsec: Cube plate scale; the sampling is then derived
            from ``disperser.sky_pitch_arcsec`` exactly as ``build_ir`` does.
        angle_rad: Cell rotation for bare centers (ignored with a disperser).
        image: Optional entrance image, ``(ny, nx)`` or a cube
            ``(n_wav, ny, nx)`` of bin-integrated rates.
        plane: Wavelength index to draw from a cube; None sums the bins
            (the band-integrated rate).
        channels: Channel indices of lenslets to highlight.
        optical_center_px: The cube's optical center ``(x, y)`` in cube
            pixels, as its producer defines it; None draws no optical center.
        grid_origin_px: Lenslet-grid origin ``(x, y)`` for bare centers; with
            a disperser it is derived and this is ignored.
        styles: An ``eyepiece.SourceStyles`` holding ``"lenslet <k>"`` names;
            None declares one from ``channels`` in order. Pass the same object
            to :func:`plot_traces` so a cell and its trace share a color.
        window: ``(x0, x1, y0, y1)`` axis limits in cube pixels; None shows the
            whole cube.
        floor: Log floor for ``image``; None uses ``1e-4`` of its peak.
        colorbar: Colorbar placement for ``image``, as in
            ``eyepiece.imshow_log`` (True for an inset, ``"figure"``, or False).
        ax: Axes to draw into. None creates a figure.
        imshow_kw: Routed to ``ax.imshow`` through ``eyepiece.imshow_log``.
        cbar_kw: Routed to the colorbar.
        cell_kw: Routed to the ``PolyCollection`` of all cell outlines.

    Returns:
        An ``eyepiece.PlotResult``. Artists: ``"collection"`` (every cell
        outline, in channel order), ``"ellipse"`` (a list of the highlighted
        cell ``Polygon`` patches, in ``channels`` order), ``"lines"`` (the
        reference markers, labeled ``"optical center"`` and
        ``"lenslet-grid origin"``), ``"text"`` (their labels and the lenslet
        labels), plus ``"image"`` when ``image`` is given and ``"cbar"``
        when it also carries a colorbar.
        ``update`` is None.
    """
    ep = eyepiece()
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Polygon

    if isinstance(disperser_or_centers, LensletDisperser):
        disperser = disperser_or_centers
        cell_px = _resolve_fp_px_per_lenslet(
            disperser, fp_px_per_lenslet, fp_pixel_scale_arcsec
        )
        angle = float(disperser.angle_rad)
        cx, cy = lenslet_cell_centers(disperser, fp_shape, cell_px)
        ox, oy = lenslet_cell_centers(
            disperser, fp_shape, cell_px, positions=jnp.zeros((1, 2))
        )
        grid_origin = (float(ox[0]), float(oy[0]))
    else:
        if fp_px_per_lenslet is None or angle_rad is None:
            raise ValueError("bare cell centers need fp_px_per_lenslet and angle_rad")
        centers = np.asarray(disperser_or_centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError(f"bare cell centers must be (n, 2), got {centers.shape}")
        cx, cy = centers[:, 0], centers[:, 1]
        cell_px = float(fp_px_per_lenslet)
        angle = float(angle_rad)
        grid_origin = grid_origin_px
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    channels = [int(c) for c in channels]

    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    extent = _edge_extent(fp_shape)
    artists = {}
    if image is not None:
        img = np.asarray(image, dtype=float)
        if img.ndim == 3:
            img = img.sum(axis=0) if plane is None else img[plane]
        if img.shape != tuple(fp_shape):
            raise ValueError(
                f"image shape {img.shape} does not match fp_shape {tuple(fp_shape)}"
            )
        drawn = _log_image(
            ep,
            img,
            ax,
            extent,
            floor,
            None,
            None,
            None,
            "entrance rate per cube pixel",
            {"imshow_kw": imshow_kw, "cbar_kw": cbar_kw, "colorbar": colorbar},
            1e-4,
        )
        artists.update(drawn.artists)

    corners = _cell_corners(cx, cy, cell_px, angle)
    kw = {
        "facecolors": "none",
        "edgecolors": [_neutral(0.55)],
        "linewidths": 0.6,
        "zorder": 3,
        **(cell_kw or {}),
    }
    cells = PolyCollection(corners, **kw)
    ax.add_collection(cells)
    artists["collection"] = cells

    styles = _styles_for(ep, channels, styles)
    patches = []
    texts = []
    for ch in channels:
        style = styles[_source_name(ch)]
        patch = Polygon(
            corners[ch],
            closed=True,
            fill=False,
            edgecolor=style["color"],
            linewidth=2.2,
            label=_source_name(ch),
            zorder=5,
        )
        ax.add_patch(patch)
        patches.append(patch)
        top = corners[ch][np.argmax(corners[ch][:, 1])]
        texts.append(
            ax.annotate(
                _source_name(ch),
                (float(top[0]), float(top[1])),
                xytext=(0, 4),
                textcoords="offset points",
                color=style["color"],
                fontsize="small",
                ha="center",
                va="bottom",
                path_effects=_halo(),
                zorder=6,
            )
        )
    if patches:
        artists["ellipse"] = patches
    if texts:
        artists["text"] = texts

    ref = _neutral(1.0)
    if optical_center_px is not None:
        _origin_marker(
            ax, optical_center_px, "x", "optical center", (-8, 8), ref, artists
        )
    if grid_origin is not None:
        _origin_marker(
            ax, grid_origin, "s", "lenslet-grid origin", (8, -8), ref, artists
        )

    x0, x1, y0, y1 = extent if window is None else window
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("entrance-plane $x$ [cube px]")
    ax.set_ylabel("entrance-plane $y$ [cube px]")
    return ep.PlotResult(ax=ax, artists=artists)


def _footprint_image(ir, channels):
    """Detector image of unit-flux-per-bin spectra in ``channels`` (None: all)."""
    rows = np.asarray(ir.det_rows)
    vals = np.asarray(ir.det_vals)
    if channels is not None:
        rows = rows[channels]
        vals = vals[channels]
    ny, nx = ir.det_shape
    flat = np.zeros(ny * nx)
    np.add.at(flat, rows.reshape(-1), vals.reshape(-1))
    return flat.reshape(ny, nx)


def _footprint_box(ir, channel, index):
    """Pixel-edge box ``(x0, y0, width, height)`` of one PSFlet footprint.

    Read from the IR's own footprint: the detector pixels that carry weight
    for that (lenslet, wavelength). Footprint pixels that fall off the
    detector are stored with zero weight at a clipped index, so only entries
    with positive weight count. Returns None when the whole footprint is off
    the detector.
    """
    nx = ir.det_shape[1]
    rows = np.asarray(ir.det_rows[channel, index])
    vals = np.asarray(ir.det_vals[channel, index])
    rows = rows[vals > 0]
    if rows.size == 0:
        return None
    ys, xs = np.divmod(rows, nx)
    return (
        float(xs.min()) - 0.5,
        float(ys.min()) - 0.5,
        float(xs.max() - xs.min() + 1),
        float(ys.max() - ys.min() + 1),
    )


def _default_window(xs, ys, margin):
    """Axis limits spanning the drawn centroids plus ``margin`` pixels."""
    return (
        float(np.min(xs)) - margin,
        float(np.max(xs)) + margin,
        float(np.min(ys)) - margin,
        float(np.max(ys)) + margin,
    )


def plot_traces(
    ir_or_image,
    disperser_or_centroids,
    wavelengths_nm,
    *,
    channels,
    psflet_pack=None,
    marked=None,
    scan_index=None,
    scan_channel=None,
    footprints="channels",
    trace_origin_px=None,
    show_trace_origin=True,
    show_detector_edge=True,
    styles=None,
    window=None,
    floor=None,
    vmin=None,
    vmax=None,
    colorbar=True,
    ax=None,
    imshow_kw=None,
    cbar_kw=None,
):
    """Draw detector traces, PSFlet centroids, and one scanned PSFlet.

    The detector pixel grid is drawn raw (nearest, ``readouts`` colormap, log
    norm). With an IR the image is the operator's own response to a unit
    bin-integrated flux in every wavelength bin of the lenslets named by
    ``footprints``, so overlapping footprints of neighboring traces add
    exactly as they do in the forward model. The image is static: it shows
    every bin at once, and a wavelength scan only moves the highlight. On top:

    - each ``channels`` trace as a thin line through its geometric dispersion
      centroids (the dispersion model alone, before any template-pack
      correction), in the lenslet's source color; on the bare-centroids
      route there is no separate geometric trace, and the line runs through
      the centroids passed in;
    - the PSFlet centroids at the ``marked`` wavelengths, as footprints are
      actually placed (a template pack's centroid correction included), so a
      calibrated correction shows as markers leaving the geometric line;
    - the detector trace origin (a hollow circle);
    - the detector edge (a dashed outline), which matters only when
      ``window`` reaches past it;
    - with ``scan_index``, one highlighted PSFlet: its centroid, its
      footprint box from the IR (the on-detector pixels carrying weight;
      hidden while the whole footprint is off the detector), and a
      wavelength readout.

    Dispersion runs along detector x; for a positive leading dispersion
    coefficient longer wavelengths sit at larger x. The end wavelengths of
    the first channel are labeled so the direction is read off the figure.

    Args:
        ir_or_image: A built :class:`~coronachrome.SpatialChannelIR` (the
            footprint image is derived from it), or a bare detector image
            ``(ny, nx)`` drawn as given.
        disperser_or_centroids: A ``LensletDisperser`` (centroids from
            :func:`coronachrome.build.detector_centroids`, the origin from
            :func:`coronachrome.build.detector_trace_origin`), or a bare
            ``(xc, yc)`` pair of ``(n_channels, n_wav)`` detector centroids.
        wavelengths_nm: ``(n_wav,)`` bin centers the IR was built on.
        channels: Channel indices of the traces to draw.
        psflet_pack: Template pack, as passed to ``build_ir``.
        marked: Wavelength indices whose centroids are marked; None marks
            every bin.
        scan_index: Wavelength index of the highlighted PSFlet; None draws
            no highlight and returns no ``update``.
        scan_channel: The lenslet the highlight follows; None uses
            ``channels[0]``.
        footprints: ``"channels"`` (default) draws the footprints of the
            lenslets in ``channels`` only; ``"all"`` draws every lenslet's, the
            full interleaved detector. IR door only.
        trace_origin_px: Trace origin ``(x, y)`` for bare centroids; derived
            with a disperser.
        show_trace_origin: Mark the detector trace origin.
        show_detector_edge: Outline the detector boundary.
        styles: An ``eyepiece.SourceStyles`` holding ``"lenslet <k>"`` names;
            None declares one from ``channels`` in order.
        window: ``(x0, x1, y0, y1)`` axis limits in detector pixels; None
            frames the drawn centroids with a 6 pixel margin.
        floor: Log floor; None uses ``1e-3`` of the image peak.
        vmin: Norm lower bound; None uses the floor.
        vmax: Norm upper bound; None uses the image peak.
        colorbar: Colorbar placement, as in ``eyepiece.imshow_log`` (True for
            an inset, ``"figure"``, or False).
        ax: Axes to draw into. None creates a figure.
        imshow_kw: Routed to ``ax.imshow`` through ``eyepiece.imshow_log``.
        cbar_kw: Routed to the colorbar.

    Returns:
        An ``eyepiece.PlotResult``. Artists: ``"image"``, ``"cbar"`` (unless
        ``colorbar=False``), ``"lines"`` (one trace line per channel in
        ``channels`` order, then the trace-origin marker labeled
        ``"detector trace origin"``),
        ``"scatter"`` (one collection of marked centroids, channel-major:
        every marked wavelength of ``channels[0]``, then of ``channels[1]``,
        and so on), ``"ellipse"`` (the ``"detector edge"`` rectangle and,
        with a scan on the IR door, the ``"scan footprint"`` rectangle),
        ``"text"`` (labels; the readout is labeled ``"scan readout"``) and,
        with a scan, ``"line"`` (the highlighted centroid). ``update(k)``
        moves the highlight to wavelength index ``k``: it changes only
        ``"line"``, the scan footprint rectangle (its extent, and its
        visibility when a footprint lies wholly off the detector), and the
        readout text; the image, its color scale, and the axis limits never
        change.
    """
    ep = eyepiece()
    import hwostyle
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    lam = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
    channels = [int(c) for c in channels]
    if not channels:
        raise ValueError("channels must name at least one lenslet")

    ir = ir_or_image if isinstance(ir_or_image, SpatialChannelIR) else None
    if ir is not None:
        if ir.n_wav != lam.shape[0]:
            raise ValueError(
                f"wavelengths_nm has {lam.shape[0]} bins, the IR {ir.n_wav}"
            )
        if footprints not in ("all", "channels"):
            raise ValueError('footprints must be "all" or "channels"')
        image = _footprint_image(ir, None if footprints == "all" else channels)
        det_shape = ir.det_shape
        cbar_label = "response to unit flux per bin"
    else:
        image = np.asarray(ir_or_image, dtype=float)
        det_shape = image.shape
        cbar_label = None

    if isinstance(disperser_or_centroids, LensletDisperser):
        disperser = disperser_or_centroids
        xc, yc = detector_centroids(disperser, lam, psflet_pack=psflet_pack)
        gx, gy = detector_centroids(disperser, lam, corrected=False)
        trace_origin = detector_trace_origin(disperser)
    else:
        xc, yc = (np.asarray(a, dtype=float) for a in disperser_or_centroids)
        gx, gy = xc, yc
        trace_origin = trace_origin_px
    xc, yc = np.asarray(xc), np.asarray(yc)
    gx, gy = np.asarray(gx), np.asarray(gy)
    marked = np.arange(lam.shape[0]) if marked is None else np.asarray(marked)

    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    drawn = _log_image(
        ep,
        image,
        ax,
        _edge_extent(det_shape),
        floor,
        vmin,
        vmax,
        hwostyle.cmaps.readouts,
        cbar_label,
        {"imshow_kw": imshow_kw, "cbar_kw": cbar_kw, "colorbar": colorbar},
        1e-3,
    )
    artists = dict(drawn.artists)

    styles = _styles_for(ep, channels, styles)
    lines, texts, colors = [], [], []
    for ch in channels:
        color = styles[_source_name(ch)]["color"]
        (line,) = ax.plot(
            gx[ch],
            gy[ch],
            color=color,
            linewidth=1.0,
            alpha=0.9,
            label=f"{_source_name(ch)} trace",
            zorder=4,
        )
        lines.append(line)
        colors.extend([color] * marked.shape[0])
    mx = np.concatenate([xc[ch, marked] for ch in channels])
    my = np.concatenate([yc[ch, marked] for ch in channels])
    artists["scatter"] = ax.scatter(
        mx,
        my,
        s=22,
        facecolors="none",
        edgecolors=colors,
        linewidths=1.2,
        zorder=5,
        label="PSFlet centroids",
    )

    first = channels[0]
    text_color = _neutral(1.0)
    for k, ha in ((0, "right"), (lam.shape[0] - 1, "left")):
        texts.append(
            ax.annotate(
                f"{lam[k]:.0f} nm",
                (float(xc[first, k]), float(yc[first, k])),
                xytext=(-7 if ha == "right" else 7, 7),
                textcoords="offset points",
                color=styles[_source_name(first)]["color"],
                fontsize="small",
                ha=ha,
                va="bottom",
                path_effects=_halo(),
                zorder=6,
            )
        )

    ellipses = []
    if show_detector_edge:
        ny, nx = det_shape
        edge = Rectangle(
            (-0.5, -0.5),
            nx,
            ny,
            fill=False,
            edgecolor=_neutral(0.8),
            linestyle="--",
            linewidth=1.0,
            label="detector edge",
            zorder=4,
        )
        ax.add_patch(edge)
        ellipses.append(edge)

    if show_trace_origin and trace_origin is not None:
        _origin_marker(
            ax,
            trace_origin,
            "o",
            "detector trace origin",
            (8, -8),
            text_color,
            {"lines": lines, "text": texts},
        )

    update = None
    if scan_index is not None:
        sc = first if scan_channel is None else int(scan_channel)
        color = styles[_source_name(sc)]["color"] if sc in channels else text_color
        (marker,) = ax.plot(
            [float(xc[sc, scan_index])],
            [float(yc[sc, scan_index])],
            linestyle="none",
            marker="+",
            markersize=14,
            markeredgewidth=2.0,
            color=text_color,
            label="scan centroid",
            path_effects=_halo(4.0),
            zorder=7,
        )
        artists["line"] = marker
        box = None
        if ir is not None:
            box = Rectangle(
                (0.0, 0.0),
                0.0,
                0.0,
                fill=False,
                edgecolor=color,
                linewidth=1.6,
                label="scan footprint",
                zorder=6,
            )
            ax.add_patch(box)
            ellipses.append(box)

        def place_box(index):
            extent = _footprint_box(ir, sc, index)
            box.set_visible(extent is not None)
            if extent is not None:
                bx, by, bw, bh = extent
                box.set_xy((bx, by))
                box.set_width(bw)
                box.set_height(bh)

        if box is not None:
            place_box(scan_index)
        readout = ax.text(
            0.02,
            0.97,
            rf"{_source_name(sc)}, $\lambda$ = {lam[scan_index]:.0f} nm",
            transform=ax.transAxes,
            color=text_color,
            fontsize="small",
            ha="left",
            va="top",
            label="scan readout",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": _neutral(0.0),
                "edgecolor": "none",
                "alpha": 0.8,
            },
            zorder=7,
        )
        texts.append(readout)

        def update(index):
            """Move the highlight to wavelength index ``index``."""
            marker.set_data([float(xc[sc, index])], [float(yc[sc, index])])
            if box is not None:
                place_box(index)
            readout.set_text(rf"{_source_name(sc)}, $\lambda$ = {lam[index]:.0f} nm")

    artists["lines"] = lines
    artists["text"] = texts
    if ellipses:
        artists["ellipse"] = ellipses

    if window is None:
        sel = np.concatenate([xc[channels].ravel(), gx[channels].ravel()])
        sel_y = np.concatenate([yc[channels].ravel(), gy[channels].ravel()])
        window = _default_window(sel, sel_y, 6.0)
    ax.set_xlim(window[0], window[1])
    ax.set_ylim(window[2], window[3])
    ax.set_aspect("equal")
    ax.set_xlabel("detector $x$ [px]")
    ax.set_ylabel("detector $y$ [px]")
    return ep.PlotResult(ax=ax, artists=artists, update=update)
