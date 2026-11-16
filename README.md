# GRMHDSnapshots.jl

Load snapshots from GRMHD simulations – for now, supports KORAL-style, iharm3d/KHARMA (FMKS) and H-AMR snapshots, more can be added as needed. Used in [PlasmaScope.jl](https://github.com/aplavin/PlasmaScope.jl) for dataset access.

Supports the native dump formats and, for KORAL, the equivalent zarr representation. H-AMR dumps load as their own AMR blocks; `resample` turns them into a uniform grid.
See `load_snapshot()` (which auto-detects the format) for details.
