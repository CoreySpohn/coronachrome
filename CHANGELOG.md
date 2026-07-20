# Changelog

## [1.0.0](https://github.com/CoreySpohn/coronachrome/compare/v0.3.0...v1.0.0) (2026-07-20)


### ⚠ BREAKING CHANGES

* **build:** angular sampling units (sky_pitch_arcsec over fp_pixel_scale_arcsec) and exact wavelength_edges bin smear
* **build:** derive fp_px_per_lenslet from the descriptor sky pitch, with sampling diagnostics

### Features

* **build:** angular sampling units (sky_pitch_arcsec over fp_pixel_scale_arcsec) and exact wavelength_edges bin smear ([75651ec](https://github.com/CoreySpohn/coronachrome/commit/75651ec459d65635c61ddf99b3f4a2905a0fe2af))
* **build:** derive fp_px_per_lenslet from the descriptor sky pitch, with sampling diagnostics ([ec1af1c](https://github.com/CoreySpohn/coronachrome/commit/ec1af1c80f71d4d0df3e1451742c2aa4080ce6fc))
* **spectral:** Nyquist channel grid helpers and flux-conserving rebin ([74af7a2](https://github.com/CoreySpohn/coronachrome/commit/74af7a2a1ba74c132b06e0179c6c74eb00d1d2cb))
* **templates:** frozen PSFlet template packs and template-mode IR build ([13dc5f6](https://github.com/CoreySpohn/coronachrome/commit/13dc5f621d4aced8cfd2cc900244d2cff86d8d41))

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
