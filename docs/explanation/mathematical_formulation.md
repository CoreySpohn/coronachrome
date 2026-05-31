# Mathematical formulation

This page derives the IFS forward model and the extraction inverse problem from
first principles. It is the theory that the [model](model) page implements. The
[model](model) page is the gentler onramp; read it first if the terms here are
unfamiliar.

## The continuous forward model

The input to the IFS is a focal-plane spectral cube, $f(u, \lambda)$, the flux at
focal-plane position $u$ and wavelength $\lambda$ (already through the coronagraph,
for example from coronagraphoto). The IFS turns this into a detector image $d(x)$,
the intensity at detector position $x$. Two linear stages connect them.

**Stage 1: spatial sampling.** Lenslet $\ell$, centered at focal-plane position
$u_\ell$, collects the light over its aperture. Its flux at wavelength $\lambda$ is

$$ z_\ell(\lambda) = \int a_\ell(u)\, f(u, \lambda)\, du, $$

where $a_\ell$ is the lenslet aperture (in coronachrome, a bilinear footprint at the
lenslet center).

**Stage 2: dispersion and placement.** The disperser spreads lenslet $\ell$ at
wavelength $\lambda$ to a centroid $c_\ell(\lambda)$ on the detector and stamps a
small image there, the PSFlet $\phi$. Summing over lenslets and integrating over
wavelength,

$$ d(x) = \sum_\ell \int z_\ell(\lambda)\, \phi\!\left(x - c_\ell(\lambda);\, \lambda\right) d\lambda. $$

Both stages are linear in the input. Doubling $f$ doubles $d$, and contributions add.

## Discretization into channels

Bin the wavelength axis into $n_\lambda$ bins $\lambda_w$ and the detector into
$n_\mathrm{det}$ pixels $x_i$. The per-lenslet spectrum becomes a finite array
$z_{\ell w} = z_\ell(\lambda_w)$, and the detector becomes a vector $y_i = d(x_i)$.

A **channel** is one (lenslet, wavelength) pair $(\ell, w)$. There are
$n_\mathrm{ch} = n_\mathrm{lenslets}\cdot n_\lambda$ of them. Stage 2 becomes a finite
sum,

$$ y_i = \sum_{(\ell, w)} H_{i,\,(\ell w)}\, z_{\ell w}, \qquad
   H_{i,\,(\ell w)} = \phi\!\left(x_i - c_\ell(\lambda_w);\, \lambda_w\right), $$

that is, $H_{i,c}$ is the value of channel $c$'s PSFlet at detector pixel $i$.
Flattening the channels into a vector $z$ and the pixels into $y$,

$$ \boxed{\,y = H z\,}, \qquad H \in \mathbb{R}^{n_\mathrm{det}\times n_\mathrm{ch}}. $$

Stage 1 is a second linear map $z = S f$ from the discretized cube to the per-lenslet
spectra. coronachrome applies $S$ (the spatial footprints) and then $H$ (the
dispersion). Extraction inverts $H$ to recover $z$, the per-lenslet spectra.

## The structure of H

Each **column** of $H$ is one channel: the flattened PSFlet footprint of lenslet
$\ell$ at wavelength $\lambda_w$, centered at $c_\ell(\lambda_w)$. PSFlets are
spatially compact, a few pixels across, so each column has only $O(k^2)$ nonzero
entries for a $k\times k$ footprint. $H$ is therefore sparse, with roughly
$n_\mathrm{ch}\,k^2$ nonzeros out of $n_\mathrm{det}\,n_\mathrm{ch}$ entries. This is
why coronachrome stores $H$ in a sparse format and never forms it densely.

The **normal matrix** $A = H^\top W H$ (with the weight $W$ defined below) is
$n_\mathrm{ch}\times n_\mathrm{ch}$. Its entry

$$ A_{cc'} = \sum_i H_{ic}\, W_{ii}\, H_{ic'} $$

is the noise-weighted overlap between the footprints of channels $c$ and $c'$. It is
nonzero only when the two channels land on shared detector pixels, which happens for
neighboring wavelengths of one lenslet and for adjacent lenslets. $A$ is therefore
banded, near-diagonal. Those off-diagonal entries are exactly the **cross-talk**
between micro-spectra.

## The inverse problem

The measurement model adds detector noise,

$$ y = H z + n, \qquad \mathrm{Cov}(n) = N = \mathrm{diag}(N_i), $$

where $N_i$ is the per-pixel variance (shot, dark, clock-induced charge, and read
noise, from the optixstuff detector). Define the inverse-variance weight
$W = N^{-1}$.

Under Gaussian noise, the log-likelihood of $z$ given $y$ is, up to a constant,
$-\tfrac{1}{2}(Hz - y)^\top W (Hz - y)$. Maximizing it is weighted least squares,

$$ \hat z = \arg\min_z\ (Hz - y)^\top W (Hz - y). $$

Setting the gradient to zero gives the **normal equations** and their solution,

$$ (H^\top W H)\,\hat z = H^\top W y, \qquad
   \hat z = (H^\top W H)^{-1} H^\top W y. $$

This estimator is unbiased, $\mathbb{E}[\hat z] = z$. By the Gauss-Markov theorem it
is the best linear unbiased estimator: among all unbiased linear estimators it has
the smallest variance. Its covariance is

$$ \boxed{\,\mathrm{Cov}(\hat z) = (H^\top W H)^{-1}\,}. $$

When $W = N^{-1}$ is the true inverse noise covariance, this also equals the
Cramer-Rao bound, so no unbiased estimator does better. The per-wavelength
one-sigma error bars are the square root of the diagonal of the per-spaxel
$n_\lambda\times n_\lambda$ block of this matrix. Because $N$ contains a shot-noise
term proportional to the source rate, the error bars grow where the dispersed
spectrum is bright.

Inverting $A = H^\top W H$ is what removes the cross-talk bias: the off-diagonal
entries tell the estimator how much of each detector pixel belongs to which channel.
The price is that variance is inflated where channels overlap strongly, which is why
the covariance, not just per-pixel photon counting, is needed for honest error bars.

## The matched filter as the diagonal approximation

If the channels did not overlap, $A$ would be diagonal and each channel would be
recovered independently. The matched filter makes exactly that approximation: it
keeps only the diagonal $D = \mathrm{diag}(A)$,

$$ \hat z_\mathrm{MF} = D^{-1} H^\top W y. $$

It equals the least-squares solution only when the micro-spectra do not overlap.
Otherwise it is biased by the flux that leaks in from neighboring channels, but it
costs a single adjoint $H^\top y$ rather than a linear solve, so it is useful as a
fast, differentiable first estimate.

## How this maps to the code

| Symbol | coronachrome |
|---|---|
| $H$ | `H_mono` (BCOO sparse matrix on the {class}`~coronachrome.IFSRenderer`) |
| $y = Hz$ | {meth}`IFSRenderer.forward_spmv <coronachrome.IFSRenderer.forward_spmv>` |
| $z = Sf$ (spatial sampling) | {func}`~coronachrome.spatial_sample` |
| $\hat z = (H^\top W H)^{-1}H^\top W y$ | {func}`~coronachrome.lstsq` |
| $D^{-1}H^\top W y$ | {func}`~coronachrome.matched_filter` |
| $(H^\top W H)^{-1}$ block | {func}`~coronachrome.spectrum_covariance` |
| $\sqrt{\mathrm{diag}}$ of the block | {func}`~coronachrome.spectrum_errorbars` |
| $W = 1/N$ | `1 / optixstuff` {meth}`detector.noise_variance <optixstuff.AbstractDetector.noise_variance>` |
