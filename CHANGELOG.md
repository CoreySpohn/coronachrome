# Changelog

## [2.0.0](https://github.com/CoreySpohn/coronachrome/compare/v1.1.0...v2.0.0) (2026-11-28)


### ⚠ BREAKING CHANGES

* **coronachrome:** close the IFS build and render forwards at their working precision
* **coronachrome:** decide extraction support identically in both lanes, keep zero-weight derivatives finite, scale the matched-filter adjoint
* **extract:** Columns with squared norm below the dtype's smallest normal number return zero flux and infinite variance instead of a floored value; faint clipped columns are solved exactly.
* **extract:** detector_statistics raises for photon-carrier, spectral or n_pix records and sums a supplied variance record over bands with the signal.
* **render:** Band cubes use the leading axis ("band",); ("wavelength",) input is refused. SpatialChannelIR.n_wav is now n_bands with no alias.
* **build:** Footprints clipped by the detector edge keep their clipped fraction instead of being renormalized to unit flux. An explicit pack must declare template_origin centroid.
* **templates:** analytic packs carry template_origin and the documented pack format is 2; older packs must be regenerated.
* **ir:** SpatialChannelIR.collection_overlap axes and sparse entries change from NumPy arrays to jax.Array (indices stay int32).
* **templates:** a one-wavelength pack used for a wider band is refused.
* name every wavelength interval end low_nm and high_nm
* build detector statistics on the single optixstuff Rates type
* read named quadrature weights
* import grids and spectral types from optixstuff
* take PsfletPack and its npz io from optixstuff
* render through shared footprint overlaps and apply the steradian factor to radiance inputs
* build lenslet quadratures from declared spectral breaks
* remove rebin_channels; band rebinning is hwoutils.spectral.rebin_bands
* **viz:** take colours from eyepiece
* channels come from Band, dispersion from the disperser and lenslet layout from the geometric centre only
* render and extract on shared grids, channel bands and the detector contract, with exact lenslet cell integrals

### Features

* render and extract on shared grids, channel bands and the detector contract, with exact lenslet cell integrals ([ddd6997](https://github.com/CoreySpohn/coronachrome/commit/ddd6997c84aa143b99d08b9a5b72603207159b60))
* **render:** name the IFS cube axis band and rename n_wav to n_bands ([884010f](https://github.com/CoreySpohn/coronachrome/commit/884010f76ce33bfc322489ef928650e18a6383ea))


### Bug Fixes

* **build:** normalize PSFlet footprints before the detector mask and refuse legacy-origin packs ([a0219ed](https://github.com/CoreySpohn/coronachrome/commit/a0219ed1d77ce591a1468875317b9e3e9cccbac4))
* **coronachrome:** close the IFS build and render forwards at their working precision ([bc2b6cd](https://github.com/CoreySpohn/coronachrome/commit/bc2b6cd0d59cc7b19d7eb3f2495f45b022152219))
* **coronachrome:** decide extraction support identically in both lanes, keep zero-weight derivatives finite, scale the matched-filter adjoint ([ecf14a5](https://github.com/CoreySpohn/coronachrome/commit/ecf14a5077698a9c8532e7ec1fe6c5af2a6e28f2))
* **extract:** refuse non-electron, spectral and aperture records in detector_statistics ([9d1e8b2](https://github.com/CoreySpohn/coronachrome/commit/9d1e8b21412efe84d2ad196aec4b785c2c54a139))
* **extract:** scale solvers by the exact column norm and remove unsupported columns ([c94cd65](https://github.com/CoreySpohn/coronachrome/commit/c94cd65811174b49d30ec4b52740759a323e7012))
* **ir:** store collection overlap weights as strong leaves ([5b50e61](https://github.com/CoreySpohn/coronachrome/commit/5b50e618655924ed687efa8275a0fe87906a981b))
* render through shared footprint overlaps and apply the steradian factor to radiance inputs ([fcebc3d](https://github.com/CoreySpohn/coronachrome/commit/fcebc3dbb954ab0d596e40b1cbb4ca5bc297b7e5))
* **templates:** accept a one-wavelength PSFlet pack only for a band at its wavelength ([6a90d99](https://github.com/CoreySpohn/coronachrome/commit/6a90d99ed79e7751a61c3cf36a008a59bd51b3f4))
* **templates:** declare the template origin in analytic packs and document pack format 2 ([32d4fb9](https://github.com/CoreySpohn/coronachrome/commit/32d4fb9d768c58bff0c48cb0df6950981b298f13))
* **templates:** read the PSFlet pack grid through its accessor and allow one-wavelength packs; widen the monochromatic test bands ([b525a3e](https://github.com/CoreySpohn/coronachrome/commit/b525a3e7472a833b40d1fa34789f9be037345aca))


### Code Refactoring

* build detector statistics on the single optixstuff Rates type ([3659072](https://github.com/CoreySpohn/coronachrome/commit/36590720c42814fc0aad21484fae91105321b089))
* build lenslet quadratures from declared spectral breaks ([ccdeb40](https://github.com/CoreySpohn/coronachrome/commit/ccdeb408d1e0c0b4fa5bfbe63f238c1be048a048))
* channels come from Band, dispersion from the disperser and lenslet layout from the geometric centre only ([a95d381](https://github.com/CoreySpohn/coronachrome/commit/a95d38108cdbaffb975574902e941d0c50cdc7c9))
* import grids and spectral types from optixstuff ([609a0e5](https://github.com/CoreySpohn/coronachrome/commit/609a0e58b867b842d0a44b5480db2fe292c7effc))
* name every wavelength interval end low_nm and high_nm ([1eda5d8](https://github.com/CoreySpohn/coronachrome/commit/1eda5d857a9b28545e481d74003e595e840cb9c4))
* read named quadrature weights ([0a0c10f](https://github.com/CoreySpohn/coronachrome/commit/0a0c10f57b8b62d7c084fd8aee5ea01990ec1088))
* remove rebin_channels; band rebinning is hwoutils.spectral.rebin_bands ([4eac93f](https://github.com/CoreySpohn/coronachrome/commit/4eac93fbe959ec8dba268500c380d5f0991273e5))
* take PsfletPack and its npz io from optixstuff ([6f6246a](https://github.com/CoreySpohn/coronachrome/commit/6f6246a2f6dc5bd123b55122641456b3a41dfc9d))
* **viz:** take colours from eyepiece ([58f8f67](https://github.com/CoreySpohn/coronachrome/commit/58f8f6762e4d4df95970ca7c9fa224c6d1253874))

## [1.1.0](https://github.com/CoreySpohn/coronachrome/compare/v1.0.0...v1.1.0) (2026-09-29)


### Features

* **viz:** add footprint_box and cmap, decades, halo_marks view options ([3cf9dc4](https://github.com/CoreySpohn/coronachrome/commit/3cf9dc45d145c0c69c6b5a9cd8726b67b462bdb9))
* **viz:** add show parts to plot_lenslet_cells and plot_traces ([464e30f](https://github.com/CoreySpohn/coronachrome/commit/464e30faf4aa4077de7285e6b10f46e9b5140755))

## [1.0.0](https://github.com/CoreySpohn/coronachrome/compare/v0.3.0...v1.0.0) (2026-09-29)


### ⚠ BREAKING CHANGES

* **build:** angular sampling units (sky_pitch_arcsec over fp_pixel_scale_arcsec) and exact wavelength_edges bin smear
* **build:** derive fp_px_per_lenslet from the descriptor sky pitch, with sampling diagnostics

### Features

* **build:** angular sampling units (sky_pitch_arcsec over fp_pixel_scale_arcsec) and exact wavelength_edges bin smear ([75651ec](https://github.com/CoreySpohn/coronachrome/commit/75651ec459d65635c61ddf99b3f4a2905a0fe2af))
* **build:** derive fp_px_per_lenslet from the descriptor sky pitch, with sampling diagnostics ([ec1af1c](https://github.com/CoreySpohn/coronachrome/commit/ec1af1c80f71d4d0df3e1451742c2aa4080ce6fc))
* **build:** expose the lenslet geometry build_ir places cells and footprints with ([2652a69](https://github.com/CoreySpohn/coronachrome/commit/2652a69c8ae5438b137fa28da3b0304a9052ca0a))
* **extract:** add marginal and unmodeled measurement covariance over a nuisance basis, and an amplitude basis for one free amplitude per wavelength ([aee115d](https://github.com/CoreySpohn/coronachrome/commit/aee115d3794fe9241bdafa64dfee169608045ca9))
* **spectral:** Nyquist channel grid helpers and flux-conserving rebin ([74af7a2](https://github.com/CoreySpohn/coronachrome/commit/74af7a2a1ba74c132b06e0179c6c74eb00d1d2cb))
* **templates:** frozen PSFlet template packs and template-mode IR build ([13dc5f6](https://github.com/CoreySpohn/coronachrome/commit/13dc5f621d4aced8cfd2cc900244d2cff86d8d41))
* **viz:** add lenslet layout, trace and channel covariance views ([9734750](https://github.com/CoreySpohn/coronachrome/commit/9734750a932f56ee7fa5c054a992387344750d59))


### Bug Fixes

* **viz:** frame only on-detector footprint pixels and label covariance blocks on the diagonal ([a31bf16](https://github.com/CoreySpohn/coronachrome/commit/a31bf16297a1a1fbb032643a316bdbc189f6ca49))

## [0.3.0](https://github.com/CoreySpohn/coronachrome/compare/v0.2.0...v0.3.0) (2026-06-20)


### Features

* Add psflet reference wavelength for accurate dispersion ([6d32de3](https://github.com/CoreySpohn/coronachrome/commit/6d32de3af63a56837304dc50495523687e1930d2))
* **build:** apply disperser throughput per wavelength in build_ir ([1bc8e30](https://github.com/CoreySpohn/coronachrome/commit/1bc8e308bafe145e3c5c83beceaeab43da8b4156))
* **psflet:** pixel-integrate the Gaussian PSFlet via erf ([5b9a90a](https://github.com/CoreySpohn/coronachrome/commit/5b9a90af20bf5c23c513f91a43897a06cacbe23b))
* **psflet:** pixel-integrate the Moffat PSFlet via sub-pixel quadrature ([3338063](https://github.com/CoreySpohn/coronachrome/commit/3338063824be97f46d11fb6f76e0a64a74a62903))

## [0.2.0](https://github.com/CoreySpohn/coronachrome/compare/v0.1.0...v0.2.0) (2026-05-31)


### Features

* **build:** add flux-conserving rotated lenslet footprint builder ([3d0a430](https://github.com/CoreySpohn/coronachrome/commit/3d0a43072dc4d4991c7a1b2f654d75436c0e28ba))
* **build:** flux-conserving rotated spatial sampling in build_ir ([c23c4fe](https://github.com/CoreySpohn/coronachrome/commit/c23c4fedc1fb9de828fddbd3e6c88b9ed74c9488))
* **extract:** optional Tikhonov damping in lstsq + precision docs ([d6700da](https://github.com/CoreySpohn/coronachrome/commit/d6700da0590ea9968510a49d786921d6d5294a13))


### Performance Improvements

* **extract:** lax.map over channels in spectrum_covariance ([ae1655f](https://github.com/CoreySpohn/coronachrome/commit/ae1655fcff136558a681483f24d6e878d6d6b7e3))

## [0.1.0](https://github.com/CoreySpohn/coronachrome/compare/v0.0.1...v0.1.0) (2026-05-30)


### Features

* **build:** add build_ir compiling a disperser into the Spatial Channel IR ([310ba32](https://github.com/CoreySpohn/coronachrome/commit/310ba32887186345adac91bb0c29a46c89b7377b))
* **coronachrome:** add public API exports ([ec77618](https://github.com/CoreySpohn/coronachrome/commit/ec77618326591b0eac3280fe16b0edf172d3021f))
* **dispersion:** add lenslet dispersion model and detector centroids ([f1cfedb](https://github.com/CoreySpohn/coronachrome/commit/f1cfedb48270f16e9d414340f573bba5ac83ef22))
* **extract:** add lineax NormalCG weighted least-squares extraction ([36c8701](https://github.com/CoreySpohn/coronachrome/commit/36c87014b8b1786886701a0e193cc688a4b07f1c))
* **extract:** add matched-filter spectral estimate ([6beeac2](https://github.com/CoreySpohn/coronachrome/commit/6beeac2f2464e29b53b946ab731df23cef3ef43c))
* **extract:** add O2-dip recovery test and export extractors ([b86b155](https://github.com/CoreySpohn/coronachrome/commit/b86b1559c59afec89a427f55f5f0fe29186809f5))
* **extract:** add per-channel GLS covariance and per-wavelength error bars ([b48be01](https://github.com/CoreySpohn/coronachrome/commit/b48be01ae540fe4e829ad76311736f4729604ef8))
* **grids:** add square and hex lenslet position grids ([5c3e254](https://github.com/CoreySpohn/coronachrome/commit/5c3e254d31200757f5a634be0abcefba71014c08))
* **ir:** add SpatialChannelIR data model with validation ([431f93f](https://github.com/CoreySpohn/coronachrome/commit/431f93f4b07a8017278873673a5d08690d4965a0))
* **psflet:** add analytic PSFlet profiles with LSF smearing ([b7e503e](https://github.com/CoreySpohn/coronachrome/commit/b7e503e6e3de5c266f9685ea8d00b38d1ea76e1c))
* **render:** add H_mono adjoint and verify differentiable forward ([05d7421](https://github.com/CoreySpohn/coronachrome/commit/05d742162f251e274449f75aade8feee66815e19))
* **render:** add IFSRenderer with BCOO forward_spmv ([a8c5260](https://github.com/CoreySpohn/coronachrome/commit/a8c5260a9bc86f7d8eb0bdfc5f63816e0a519a08))
* **render:** add streaming scatter-add forward, equivalent to spmv ([78bfd09](https://github.com/CoreySpohn/coronachrome/commit/78bfd09a1b88b0da1f84b594a8d36442fa19c6c3))


### Bug Fixes

* **extract:** column-equilibrate lstsq for float32 NormalCG robustness ([6a52b09](https://github.com/CoreySpohn/coronachrome/commit/6a52b09524957b05bc281423177148760e1040b6))


### Performance Improvements

* **build:** vectorize build_ir over channels and wavelengths ([fc02808](https://github.com/CoreySpohn/coronachrome/commit/fc0280840a0463aaf8bb81ebbdcc1fd7aac47b36))

## 0.0.1 (2026-01-27)


### Features

* Initial setup ([689071e](https://github.com/CoreySpohn/coronachrome/commit/689071e8bcc5663192cb9a76f126e643dfda2535))


### Miscellaneous Chores

* release 0.0.1 ([d97c9aa](https://github.com/CoreySpohn/coronachrome/commit/d97c9aaa5d2b9a9f45f86a3128736157f70c005f))
