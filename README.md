# GRMHDSnapshots.jl

Load snapshots from GRMHD simulations – for now, supports KORAL-style and iharm3d/KHARMA (FMKS) snapshots, more can be added as needed. Used in [PlasmaScope.jl](https://github.com/aplavin/PlasmaScope.jl) for dataset access.

Supports both the native HDF5 dump formats and the equivalent zarr representation.
See `load_snapshot()` (which auto-detects the format) for details.
