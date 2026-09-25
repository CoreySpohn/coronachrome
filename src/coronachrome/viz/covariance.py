"""Spectral covariance of extracted lenslet-bin fluxes.

Takes a covariance array rather than an extraction call, so it draws any
estimate: a within-lenslet block from
:func:`coronachrome.spectrum_covariance`, or a larger matrix spanning several
lenslets in the flattened ``k = channel * n_wav + wavelength`` order, where
overlapping traces put covariance between neighboring lenslets.
"""

import numpy as np

from coronachrome.viz._require import eyepiece


def _correlation(cov):
    """``cov / sqrt(diag diag^T)``; rows with zero variance become NaN."""
    sd = np.sqrt(np.clip(np.diagonal(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        return cov / np.outer(sd, sd)


def _text_color():
    """The active text color, resolved at call time."""
    import matplotlib as mpl

    return mpl.rcParams["text.color"]


def _face_color():
    """The active axes facecolor, resolved at call time."""
    import matplotlib as mpl

    return mpl.rcParams["axes.facecolor"]


def _mid_neutral():
    """A gray halfway from the axes facecolor to the text color.

    Visible both on the pale center of a diverging map and on the background,
    in either style mode.
    """
    import matplotlib as mpl
    from matplotlib.colors import to_rgb

    face = np.asarray(to_rgb(mpl.rcParams["axes.facecolor"]))
    text = np.asarray(to_rgb(mpl.rcParams["text.color"]))
    return tuple(0.5 * (face + text))


def _tick_positions(n, max_ticks):
    """Every ``ceil(n / max_ticks)``-th index in ``range(n)``, from zero."""
    step = max(1, -(-n // max_ticks))
    return np.arange(0, n, step)


def plot_channel_covariance(
    cov,
    *,
    wavelengths_nm=None,
    block=None,
    correlation=True,
    channel_labels=None,
    max_ticks=6,
    ax=None,
    imshow_kw=None,
    cbar_kw=None,
):
    """Draw a spectral covariance or correlation matrix over wavelength bins.

    One cell per (bin, bin) pair on a symmetric diverging norm, raw pixels.
    Cells are indexed by bin, not placed on a wavelength axis, because
    channel grids are log-spaced; ticks carry the bin-center wavelengths.
    A matrix larger than ``len(wavelengths_nm)`` is read as several lenslets
    in the flattened ``channel * n_wav + wavelength`` order, and the lenslet
    blocks are separated by lines and labeled with ``channel_labels``. Entry
    ``[i, j]`` is drawn at ``x = j``, ``y = i`` with the origin at the lower
    left, so lenslet ``b``'s own block is the ``b``-th diagonal block counted
    from the lower left; each label sits in the upper-left corner of its
    lenslet's diagonal block.

    Args:
        cov: A square covariance ``(m * n_wav, m * n_wav)``, or a stack
            ``(n_sel, n_wav, n_wav)`` of within-lenslet blocks (as returned by
            :func:`coronachrome.spectrum_covariance`) with ``block`` selecting
            one.
        wavelengths_nm: ``(n_wav,)`` bin centers for the tick labels. None
            labels bins by index and treats the matrix as one lenslet.
        block: Index into a stacked ``cov``.
        correlation: Draw the correlation coefficient (norm fixed to
            ``[-1, 1]``) instead of the covariance (norm symmetric about zero
            at the largest magnitude).
        channel_labels: One name per lenslet block, in order.
        max_ticks: Most wavelength ticks per lenslet block.
        ax: Axes to draw into. None creates a figure.
        imshow_kw: Routed to ``ax.imshow`` through
            ``eyepiece.imshow_diverging``.
        cbar_kw: Routed to the colorbar.

    Returns:
        An ``eyepiece.PlotResult``. Artists: ``"image"``, ``"cbar"``, and for
        a multi-lenslet matrix ``"lines"`` (block separators) and ``"text"``
        (one label per lenslet, on its diagonal block). ``update(new_cov)``
        redraws a matrix of the same shape (after the same ``block`` selection
        and correlation transform) under the first draw's norm.
    """
    ep = eyepiece()

    arr = np.asarray(cov, dtype=float)
    if arr.ndim == 3:
        if block is None:
            raise ValueError("a stacked (n_sel, n_wav, n_wav) cov needs block=")
        arr = arr[block]
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"cov must be square, got shape {arr.shape}")
    n = arr.shape[0]
    n_wav = n if wavelengths_nm is None else int(np.size(wavelengths_nm))
    if n % n_wav:
        raise ValueError(
            f"cov size {n} is not a multiple of the {n_wav} wavelength bins"
        )
    n_blocks = n // n_wav
    if channel_labels is not None and len(channel_labels) != n_blocks:
        raise ValueError(
            f"{len(channel_labels)} channel_labels for {n_blocks} lenslet blocks"
        )

    def transform(values):
        values = np.asarray(values, dtype=float)
        if values.ndim == 3:
            values = values[block]
        return _correlation(values) if correlation else values

    data = transform(arr)
    vlim = 1.0 if correlation else float(np.nanmax(np.abs(data)))
    drawn = ep.imshow_diverging(
        data,
        ax=ax,
        extent=(-0.5, n - 0.5, -0.5, n - 0.5),
        vlim=vlim,
        cbar_label="correlation" if correlation else "covariance",
        imshow_kw=imshow_kw,
        cbar_kw=cbar_kw,
    )
    ax = drawn.ax
    artists = dict(drawn.artists)

    local = _tick_positions(n_wav, max_ticks)
    ticks = np.concatenate([b * n_wav + local for b in range(n_blocks)])
    if wavelengths_nm is None:
        labels = [str(int(t % n_wav)) for t in ticks]
        axis_label = "wavelength bin"
    else:
        lam = np.asarray(wavelengths_nm, dtype=float)
        labels = [f"{lam[t % n_wav]:.0f}" for t in ticks]
        axis_label = "bin-center wavelength [nm]"
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set_xlabel(axis_label)
    ax.set_ylabel(axis_label)

    if n_blocks > 1:
        edge = _mid_neutral()
        lines = []
        for b in range(1, n_blocks):
            pos = b * n_wav - 0.5
            lines.append(ax.axhline(pos, color=edge, linewidth=1.0))
            lines.append(ax.axvline(pos, color=edge, linewidth=1.0))
        artists["lines"] = lines
        if channel_labels is not None:
            # Label each lenslet on its own diagonal block, so the label names
            # both the row block and the column block it sits in and the
            # orientation cannot be misread.
            from matplotlib import patheffects

            halo = [patheffects.withStroke(linewidth=2.5, foreground=_face_color())]
            texts = []
            for b, name in enumerate(channel_labels):
                texts.append(
                    ax.text(
                        b * n_wav - 0.5 + 0.03 * n_wav,
                        (b + 1) * n_wav - 0.5 - 0.03 * n_wav,
                        str(name),
                        ha="left",
                        va="top",
                        fontsize="small",
                        color=_text_color(),
                        path_effects=halo,
                        zorder=5,
                    )
                )
            artists["text"] = texts

    image_update = drawn.update

    def update(new_cov):
        """Redraw ``new_cov`` under the first draw's norm."""
        new = transform(new_cov)
        if new.shape != data.shape:
            raise ValueError(f"expected shape {data.shape}, got {new.shape}")
        image_update(new)

    return ep.PlotResult(ax=ax, artists=artists, update=update)
