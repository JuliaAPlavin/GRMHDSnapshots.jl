module GRMHDSnapshots

using HDF5
using StaticArrays
using LinearAlgebra: Diagonal
using IntervalSets
using Unitful, UnitfulAstro
using NamedDims: NamedDimsArray
using DataManipulation: filtermap, flatmap, group
using OhMyThreads: tmap, @tasks, @local, chunks
import StructArrays
import ForwardDiff
import Adapt
import Mmap
using Zarr: zopen, ZGroup
using ZarrZfp

export GRMHDSnapshot, load_snapshot, load_koral, load_iharm, load_hamr, load_grid, r_min, r_max,
    plasma_state, map_plasma, fluid_ucon, fluid_velocity, flow_speed, proper_velocity, lapse, grav_redshift, volume_ratio, ks_to_bl, lunit, rhounit, bunit,
    comoving_bsq, comoving_B_gauss, bfield_magnitude, magnetization

include("coordinates.jl")

# `M_unit` (code mass unit, g) and `MBH` (black-hole mass, M☉ — fits Float32, unlike grams) each carry their own
# type parameter (`Um`, `Ub`): a number when known, `Missing` when not — scale-free dumps (e.g. iharm3d/KHARMA,
# H-AMR) store neither, and each can be supplied independently.
# The unit scalars needing a missing one (`lunit`: MBH; `rhounit`/`bunit`: both) throw rather than return `missing`.
# `r_extent` = r_prof[1]..r_prof[end], stored: readable without indexing r_prof, which may live on a GPU.
# `fluid` is a `(:φ,:θ,:r)` array of per-cell records with the keys `ρ` (density), `ũ` (relative 4-velocity ũ^i) and `B`
# (lab field B^i), vectors in Kerr-Schild components; a partial load holds only the requested keys, in that order.
# One interleaved record per cell: an interpolation reads each stencil corner from one place.
# `R` turns grid Cartesian coordinates into the spin frame (black-hole spin along +z): the spin-frame point p is the grid
# point Rᵀp. It is a `Val` of an integer `SMatrix`, so every kernel gets its entries as compile-time constants.
struct GRMHDSnapshot{F, Rp, Re, Tg, T, Rt, Um, Ub}
    fluid::F
    r_prof::Rp
    r_extent::Re
    th_grid::Tg
    φ0::T
    dφ::T
    R::Rt
    spin::T
    gam::T
    t::T
    M_unit::Um
    MBH::Ub
end

"""
    load_snapshot(path; kwargs...)

Load a single GRMHD snapshot, auto-detecting the on-disk format:
- `.h5` with a `prims` dataset → iharm3d/KHARMA FMKS ([`load_iharm`](@ref));
- `.h5` with a `quants/` group → koral ([`load_koral`](@ref));
- a `.../RUN.zarr/snapNNNN` path → koral Zarr ([`load_koral`](@ref));
- a directory with `parameters` and `new_dump0` → H-AMR, either `parameters` layout ([`load_hamr`](@ref); needs its `rcyl` kwarg).

`kwargs...` go to the selected loader (e.g. `M_unit`/`MBH` for iharm); use the explicit loaders for full control.
"""
function load_snapshot(path::AbstractString; kwargs...)
    p = rstrip(path, '/')
    fmt = snapshot_format(p)
    fmt === :iharm && return load_iharm(p; kwargs...)
    fmt === :hamr && return load_hamr(p; kwargs...)
    load_koral(p; kwargs...)
end

"""
    snapshot_format(path) -> Symbol

Detect a snapshot's on-disk format: `:iharm` for an iharm3d/KHARMA `.h5` dump (a `prims` dataset),
`:hamr` for an H-AMR dump directory (`parameters`, `new_dump0`), or `:koral` for the koral
layout (a `.h5` with a `quants/` group, or a Zarr snapshot path).
"""
function snapshot_format(path::AbstractString)
    p = rstrip(path, '/')
    all(f -> isfile(joinpath(p, f)), ("parameters", "new_dump0")) && return :hamr
    endswith(p, ".h5") || return :koral        # Zarr snapshot/run path
    h5open(p, "r") do h5
        haskey(h5, "quants") ? :koral :
        haskey(h5, "prims")  ? :iharm :
        throw(ArgumentError("unrecognized HDF5 snapshot format (no `quants/` or `prims`): $p"))
    end
end

"""
    load_koral(path; kwargs...)

Load a raw `.h5` snapshot or a Zarr snapshot path.

# Arguments
- `path`: `.h5` file, or `.../RUN.zarr/snapNNNN` store snapshot path.
- `kwargs...`: Forwarded to the selected loader.

# Returns
- `GRMHDSnapshot`.
"""
function load_koral(path::AbstractString; kwargs...)
    p = rstrip(path, '/')
    endswith(p, ".h5") ?
        load_koral_h5(p; kwargs...) :
        load_koral(zopen(dirname(p), "r")::ZGroup, basename(p); kwargs...)
end

function load_koral_h5(path::AbstractString)
    h5open(path, "r") do h5
        rd(name) = read(h5["quants/$name"])
        fluid = _interleave(map(names -> _koral_column(Float64, map(rd, names)), _KORAL_COMPONENTS))

        r = read(h5["grid_out/r"]); th = read(h5["grid_out/th"]); ph = read(h5["grid_out/ph"])
        n1 = size(r, 3); n2 = size(r, 2); n3 = size(r, 1)
        r_prof  = Float64[r[1, 1, i] for i in 1:n1]
        th_grid = Float64[th[1, j, i] for i in 1:n1, j in 1:n2]
        φ0_file = Float64(ph[1, 1, 1]); dφ = (Float64(ph[n3, 1, 1]) - φ0_file)/(n3 - 1)
        # ipole convention: file φ + π.
        φ0 = φ0_file + π

        rdscalar(name) = Float64(read(h5[name]))
        GRMHDSnapshot(fluid, r_prof, first(r_prof)..last(r_prof), th_grid, φ0, dφ, Val(one(SMatrix{3,3,Int})),
            rdscalar("header/bhspin"), rdscalar("header/gam"), rdscalar("t"),
            rdscalar("header/units/M_unit"), rdscalar("header/units/M_bh"))
    end
end

const _FLUID_KEYS = (:ρ, :ũ, :B)

# The requested fluid fields in the canonical record order.
function _fluid_keys(fields)
    fields ⊆ _FLUID_KEYS || throw(ArgumentError("unknown fluid field(s) in $fields"))
    isempty(fields) && throw(ArgumentError("no fluid fields requested"))
    filter(in(fields), _FLUID_KEYS)
end

# koral datasets of each fluid field
const _KORAL_COMPONENTS = (ρ = ("rho",), ũ = ("U1", "U2", "U3"), B = ("B1", "B2", "B3"))

_koral_column(::Type, (a,)::Tuple{AbstractArray}) = a
_koral_column(::Type{T}, cs::NTuple{3,AbstractArray}) where {T} = StructArrays.StructArray{SVector{3,T}}(cs)

# Field arrays → one dense `(:φ,:θ,:r)` array of records, in a single pass; the StructArray is only a zero-copy view.
_interleave(cols::NamedTuple) = NamedDimsArray(Array(StructArrays.StructArray(cols)), (:φ, :θ, :r))

"""
    load_grid(store)

Load shared grid arrays from a Zarr run store.

# Arguments
- `store`: Open Zarr run store containing `r`, `th`, and `ph`.

# Returns
- Named tuple with `r_prof`, `th_grid`, `φ0`, and `dφ`.
"""
function load_grid(store::ZGroup)
    rd(name) = store[name][:, :, :]
    r = rd("r"); th = rd("th"); ph = rd("ph")
    n1 = size(r, 3); n2 = size(r, 2); n3 = size(r, 1)
    φ0_file = Float64(ph[1, 1, 1])
    (; r_prof  = Float64[r[1, 1, i] for i in 1:n1],
       th_grid = Float64[th[1, j, i] for i in 1:n1, j in 1:n2],
       φ0 = φ0_file + π,
       dφ = (Float64(ph[n3, 1, 1]) - φ0_file)/(n3 - 1))
end

"""
    load_koral(store, snapname; fields=(:ρ, :ũ, :B), grid=load_grid(store))

Load one snapshot from an open Zarr run store.

# Arguments
- `store`: Open Zarr run store.
- `snapname`: Snapshot group name, e.g. `"snap002000"`.
- `fields`: Fluid fields to read; a nonempty subset of `(:ρ, :ũ, :B)`.
- `grid`: Cached grid from `load_grid(store)`.

# Returns
- `GRMHDSnapshot` whose `fluid` records hold the requested fields, in the order `(ρ, ũ, B)`; `Float32`.
"""
function load_koral(store::ZGroup, snapname::AbstractString;
                    fields = (:ρ, :ũ, :B), grid = load_grid(store))
    ks = _fluid_keys(fields)
    snap = store[snapname]::ZGroup
    rd(name) = Threads.@spawn snap[name][:, :, :]                # each zfp chunk decompresses independently
    # launch every requested component up front so the reads decompress concurrently, then interleave
    tasks = NamedTuple{ks}(map(k -> map(rd, _KORAL_COMPONENTS[k]), ks))
    fluid = _interleave(map(ts -> _koral_column(Float32, map(fetch, ts)), tasks))

    sc(x) = Float64(x)
    hdr = store.attrs; units = hdr["units"]
    GRMHDSnapshot(fluid, grid.r_prof, first(grid.r_prof)..last(grid.r_prof), grid.th_grid, grid.φ0, grid.dφ,
        Val(one(SMatrix{3,3,Int})), sc(hdr["bhspin"]), sc(hdr["gam"]), sc(snap.attrs["t"]),
        sc(units["M_unit"]), sc(units["M_bh"]))
end

"""
    load_iharm(path; fields=(:ρ, :ũ, :B), M_unit=missing, MBH=missing)

Load an iharm3d/KHARMA `.h5` dump in FMKS coordinates (`header/metric` == `"FMKS"`/`"MMKS"`).

The packed `prims` array holds the primitives in *native* FMKS components; the relative velocity
`ũ^i` and lab field `B^i` are transformed to Kerr-Schild `(r, θ, φ)` components at load via the
spatial Jacobian `J₃(r, θ)` (matching ipole's reconstruction), so the result is an ordinary
[`GRMHDSnapshot`] that all downstream physics/interpolation handles unchanged.

# Arguments
- `fields`: fluid fields to read; a nonempty subset of `(:ρ, :ũ, :B)`. The `fluid` records hold these, in the order `(ρ, ũ, B)`.
- `M_unit`, `MBH`: physical units, `M_unit` in grams and `MBH` in M☉, each optional (default `missing`). These
  dumps are scale-free (no units stored): `MBH` enables `lunit`, both together `rhounit`/`bunit`; those missing a unit throw.

# Returns
- `GRMHDSnapshot`.
"""
function load_iharm(path::AbstractString; fields = (:ρ, :ũ, :B), M_unit = missing, MBH = missing)
    ks = _fluid_keys(fields)
    h5open(path, "r") do h5
        rd(name) = read(h5[name])
        metric = strip(c -> c == '\0' || c == ' ', rd("header/metric"))
        metric in ("FMKS", "MMKS") ||
            throw(ArgumentError("only FMKS iharm/KHARMA dumps are supported, got metric = $(repr(metric))"))
        n1 = Int(rd("header/n1")); n2 = Int(rd("header/n2")); n3 = Int(rd("header/n3"))
        # grid start/step under header/geom/; FMKS shape params under header/geom/mmks/ (ipole's dir for FMKS).
        geom(name) = rd("header/geom/$name"); fmks(name) = rd("header/geom/mmks/$name")
        sx1, sx2, sx3 = geom("startx1"), geom("startx2"), geom("startx3")
        dx1, dx2, dx3 = geom("dx1"), geom("dx2"), geom("dx3")
        coords = FMKS(startx1 = sx1, hslope = fmks("hslope"), mks_smooth = fmks("mks_smooth"),
                      poly_xt = fmks("poly_xt"), poly_alpha = fmks("poly_alpha"))

        # zone-centered native coords (Julia index i → 0-based i-1). r depends on X1 only, θ on (X1, X2).
        X1(i) = sx1 + (i - 0.5)*dx1
        X2(j) = sx2 + (j - 0.5)*dx2
        r_prof  = Float64[first(bl_coord(coords, X1(i), X2(1))) for i in 1:n1]
        th_grid = Float64[last(bl_coord(coords, X1(i), X2(j)))  for i in 1:n1, j in 1:n2]

        # prims: HDF5 (n1,n2,n3,n_prim) row-major → Julia (n_prim,n3,n2,n1). Slots: 1=RHO, 3-5=U1-3, 6-8=B1-3.
        prims = rd("prims")
        # native contravariant 3-vector → KS: v_KS = J₃·v_nat, J₃(r,θ) built once per (i,j) and reused over φ.
        J = [jacobian_spatial(coords, X1(i), X2(j)) for i in 1:n1, j in 1:n2]
        tovec(s, k, j, i) = J[i, j] * SVector(prims[s, k, j, i], prims[s + 1, k, j, i], prims[s + 2, k, j, i])
        cellf = (ρ = (k, j, i) -> prims[1, k, j, i], ũ = (k, j, i) -> tovec(3, k, j, i), B = (k, j, i) -> tovec(6, k, j, i))
        fluid = NamedDimsArray(_cellmap(cellf[ks], (n3, n2, n1)), (:φ, :θ, :r))

        sc(x) = Float64(x)   # heterogeneous header scalars → uniform Float64 struct fields
        # φ = X3 (zone-centered); no +π offset (that is specific to the koral converter).
        GRMHDSnapshot(fluid, r_prof, first(r_prof)..last(r_prof), th_grid, sc(sx3 + 0.5*dx3), sc(dx3),
            Val(one(SMatrix{3,3,Int})), sc(fmks("a")), sc(rd("header/gam")), sc(rd("t")), M_unit, MBH)
    end
end

# The records `(; key = fs[key](k, j, i), …)` of every cell (k, j, i) of a grid of size `n`.
_cellmap(fs::NamedTuple, n) = [map(f -> f(k, j, i), fs) for k in 1:n[1], j in 1:n[2], i in 1:n[3]]

"""
    load_hamr(dir; rcyl, rmin=nothing, rmax=nothing, maxlevel=(;), fields=(:ρ, :ũ, :B), M_unit=missing, MBH=missing)

Load an H-AMR native dump directory (`parameters`, `new_dump0`, `new_dump1`, …), restricted to the radial cells
whose centers lie in `[rmin, rmax]`. Only the blocks overlapping that range are read, and only their radial slabs within it.

Both `parameters` layouts are supported, told apart by the file size: the old one (refinement in φ only, block
metadata in `dir/grid`) and the new one (refinement in each direction, block metadata in the run's
`gdumps/grid`, `grid1`, … next to `dir`, i.e. `joinpath(dirname(dir), "gdumps", "grid")`).

The result is an ordinary uniform [`GRMHDSnapshot`] at one refinement level per direction: the finest level among the
blocks whose radial extent overlaps `[rmin, rmax]`, capped by `maxlevel`. Each output cell is the average of the
stored cells it overlaps, weighted by the overlap in native coordinates: a coarser stored cell is copied into every
output cell it covers, finer ones are averaged. Without caps every output cell is a copy of one stored cell.

The dump stores the 4-velocity `u^μ` and lab field `B^i` in native components; each stored cell is transformed to
Kerr-Schild components at its own center via the analytic spatial Jacobian, and `u^μ` is converted to the relative
velocity `ũ^i`, before averaging. Stored values are used without correction, placed at the standard cell centers
`startx + (i + 1/2)·h` (`i` the 0-based cell index, `h` the cell spacing at the block's level). The `fluid` records are
`Float32` (the dump precision); transforms and averages are computed in `Float64`.

# Arguments
- `rcyl`: the run's θ coordinates, required since they are not stored in the dump files. A number is the
  cylindrification radius ([`HAMRCyl`](@ref); HARMPI `global_x10 = log(rcyl)`; 24.32 for the EHT code-comparison
  run), assuming cylindrified Gammie θ with hslope = 1 and HARMPI's default corner `x20 = -1 + 1/n2` (`n2` the
  θ cell count); old layout only. `nothing` is plain θ = π/2·(1 + X2) ([`HAMRSph`](@ref)).
- `rmin`, `rmax`: radial range of cell centers; `nothing` extends to the grid edge.
- `maxlevel`: per-direction caps on the output refinement level, a `NamedTuple` with any of the keys `r`, `θ`, `φ`;
  each a nonnegative integer, or `nothing` for no cap.
- `fields`: fluid fields to read; a nonempty subset of `(:ρ, :ũ, :B)`. The `fluid` records hold these, in the order `(ρ, ũ, B)`.
- `M_unit`, `MBH`: physical units (g, M☉), each optional, as for [`load_iharm`](@ref) (the dumps store none).

# Returns
- `GRMHDSnapshot`.
"""
function load_hamr(dir::AbstractString; rcyl, rmin = nothing, rmax = nothing, maxlevel::NamedTuple = (;),
                   fields = (:ρ, :ũ, :B), M_unit = missing, MBH = missing)
    ks = _fluid_keys(fields)
    keys(maxlevel) ⊆ (:r, :θ, :φ) || throw(ArgumentError("maxlevel keys must be among (:r, :θ, :φ), got $(keys(maxlevel))"))
    all(l -> isnothing(l) || (l isa Integer && l ≥ 0), maxlevel) ||
        throw(ArgumentError("maxlevel values must be nonnegative integers or nothing, got $maxlevel"))
    par = _hamr_parameters(joinpath(dir, "parameters"))
    coords = _hamr_coords(par.layout, rcyl, par)
    blocks = _hamr_blocks(dir, par)
    (; bs, nb, startx, dx) = par

    # output level per direction: the finest among the blocks whose radial extent overlaps the range, capped;
    # uncapped, the output is then at least as fine as every block holding an output cell
    rsel = something(rmin, 0)..something(rmax, Inf)
    nocells() = ArgumentError("no radial cell centers in [$rmin, $rmax]; the grid spans r = $(exp(startx[1])) … $(exp(startx[1] + nb[1]*bs[1]*dx[1]))")
    redge(b, c) = exp(startx[1] + c*bs[1]*dx[1]/2^b.level[1])
    near = filter(b -> !isempty((redge(b, b.coord[1])..redge(b, b.coord[1] + 1)) ∩ rsel), blocks)
    isempty(near) && throw(nocells())
    finest = ntuple(d -> maximum(b -> b.level[d], near), 3)
    tl = map((l, k) -> min(something(get(maxlevel, k, nothing), l), l), finest, (:r, :θ, :φ))
    X(d, i) = _hamr_center(par, d, tl[d], i)
    n = nb .* bs .* 2 .^ tl
    # 0-based output radial cells; contiguous since r grows with the index
    c = filter(i -> exp(X(1, i)) ∈ rsel, 0:n[1]-1)
    isempty(c) && throw(nocells())
    ir = first(c):last(c)

    r_prof  = map(i -> exp(X(1, i)), ir)
    th_grid = [last(bl_coord(coords, X(1, i), X(2, j))) for i in ir, j in 0:n[2]-1]

    zero32 = (ρ = 0f0, ũ = zero(SVector{3,Float32}), B = zero(SVector{3,Float32}))
    outs = Array{typeof(zero32[ks])}(undef, n[3], n[2], length(ir))
    rel = filter(b -> !isempty(_hamr_outcells(b, bs, tl) ∩ ir), blocks)    # every block contributing to the output
    geo = :ũ in ks || :B in ks ? _hamr_cell_geometry(rel, par, coords, tl, ir) : nothing
    # Float64 sums for averaged cells; weights add to 1 per output cell since blocks tile, `w > 0` marks them
    z = merge((; w = 0.0), (ρ = 0.0, ũ = zero(SVector{3,Float64}), B = zero(SVector{3,Float64}))[ks])
    # each task owns whole radial chunks of the output (a few per thread, for balance), so no two tasks write the same cell
    @tasks for cr in chunks(ir; n = 4Threads.nthreads())
        @local row = Vector{typeof(z)}(undef, bs[3])
        cbs = filter(b -> !isempty(_hamr_outcells(b, bs, tl) ∩ cr), rel)
        ro = first(cr) - first(ir)
        sums = all(b -> _hamr_iscopy(b, tl), cbs) ? nothing : fill(z, n[3], n[2], length(cr))
        foreach(b -> _hamr_scatter!(_hamr_iscopy(b, tl) ? outs : sums, ro, row, b, par, geo, tl, cr), cbs)
        isnothing(sums) || foreach(I -> iszero(sums[I].w) || _hamr_put!(outs, ro, Tuple(I)..., sums[I], 1), CartesianIndices(sums))
    end

    # φ = X3 (zone-centered); no +π offset (that is specific to the koral converter).
    GRMHDSnapshot(NamedDimsArray(outs, (:φ, :θ, :r)), r_prof, first(r_prof)..last(r_prof), th_grid,
        X(3, 0), dx[3]/2^tl[3], Val(one(SMatrix{3,3,Int})), par.a, par.gam, par.t, M_unit, MBH)
end

# One stored cell: rho, ug, u^0..3, B^1..3, little-endian float32.
const HAMRCell = SVector{9,Float32}

# Native coordinate in direction d of the center of 0-based cell i at refinement level l.
_hamr_center(par, d, l, i) = par.startx[d] + (i + 1/2)*par.dx[d]/2^l

# 0-based cells of block `b` in direction d, at its own level.
_hamr_cells(b, bs, d) = b.coord[d]*bs[d] .+ (0:bs[d]-1)

# 0-based output radial cells, at target levels `tl`, overlapping block `b`.
_hamr_outcells(b, bs, tl) = _relevel(_hamr_cells(b, bs, 1), tl[1] - b.level[1])

# 0-based cells at level L+s covering the 0-based cells `rng` at level L (dyadic grids aligned at startx).
_relevel(rng::AbstractUnitRange, s::Integer) = s ≥ 0 ? ((first(rng) << s):((last(rng) + 1) << s) - 1) : ((first(rng) >> -s):(last(rng) >> -s))

# Spatial Jacobian and KS metric at the stored cell centers, computed once per (r level, θ level, i, j) the blocks need:
# for each (r level, θ level), the 0-based first cells (i0, j0) and the matrix of (; J, g) from there.
function _hamr_cell_geometry(blocks, par, coords, tl, ir)
    span(d, bl) = minimum(b -> first(_hamr_cells(b, par.bs, d)), bl):maximum(b -> last(_hamr_cells(b, par.bs, d)), bl)
    map(group(b -> b.level[1:2], blocks)) do bl
        (l1, l2) = first(bl).level
        ii = _relevel(ir, l1 - tl[1]) ∩ span(1, bl)
        jj = span(2, bl)
        M = tmap(CartesianIndices((ii, jj))) do I
            X1 = _hamr_center(par, 1, l1, I[1]); X2 = _hamr_center(par, 2, l2, I[2])
            (; J = jacobian_spatial(coords, X1, X2), g = ks_gcov(bl_coord(coords, X1, X2)..., par.a))
        end
        (; i0 = first(ii), j0 = first(jj), M)
    end
end

# block no finer than the output: each output cell lies in one stored cell, written directly
_hamr_iscopy(b, tl) = all(≥(0), tl .- b.level)

# Into the output records (radial offset `ro`): the cell without its weight, in the output precision.
Base.@propagate_inbounds _hamr_put!(outs::AbstractArray{R}, ro, ko, jo, it, c, w) where {R} = outs[ko, jo, it + ro] = convert(R, NamedTuple{fieldnames(R)}(c))
# Into the sums, which have the cell's own element type: weighted accumulation.
Base.@propagate_inbounds _hamr_put!(sums::AbstractArray{E}, ro, ko, jo, it, c::E, w) where {E} = sums[ko, jo, it] = map((x, y) -> x + w*y, sums[ko, jo, it], c)

# Put block `b`'s stored cells overlapping the output radial cells `cr` into `sink` (radial index local to `cr`), each
# converted to KS at its own center and weighted by its overlap. Stored cells: φ fastest, then θ, then r.
function _hamr_scatter!(sink, ro, row, b, par, geo, tl, cr)
    (; bs) = par
    s = tl .- b.level
    w = exp2(sum(min.(s, 0)))
    o = b.coord .* bs
    ii = _relevel(cr, -s[1]) ∩ _hamr_cells(b, bs, 1)
    cells = view(b.cells, :, :, ii .- (o[1] - 1))
    G = isnothing(geo) ? nothing : geo[b.level[1:2]]
    # output φ cells of stored φ cell kl: k0(kl) .+ (1:nk)
    k0(kl) = first(_relevel((o[3] + kl - 1):(o[3] + kl - 1), s[3]))
    nk = 1 << max(s[3], 0)
    for (il, i) in enumerate(ii), jl in 1:bs[2]
        j = o[2] + jl - 1
        gc = isnothing(G) ? nothing : G.M[i - G.i0 + 1, j - G.j0 + 1]
        # convert each stored cell of the φ row once, then put it into every output cell it overlaps
        map!(kl -> _hamr_ks(ltoh.(cells[kl, jl, il]), gc, eltype(row)), row, 1:bs[3])
        for it in _relevel(i:i, s[1]) ∩ cr .- (first(cr) - 1), jo in _relevel(j:j, s[2]) .+ 1, kl in 1:bs[3], q in 1:nk
            @inbounds _hamr_put!(sink, ro, k0(kl) + q, jo, it, row[kl], w)
        end
    end
end

# Stored cell `p` with KS components at its center (geometry `gc`), as the sums element type `E`: unit weight `w`, then the
# requested fields. Only the selected thunks run (`gc` is `nothing` for ρ alone); selecting by the static `K` keeps it
# type-stable. u^t is unchanged: the native↔KS Jacobian has a trivial t row/column.
function _hamr_ks(p::HAMRCell, gc, ::Type{E}) where {K, E<:NamedTuple{K}}
    fs = (w = () -> 1.0, ρ = () -> p[1],
          ũ = () -> gammie_uprim(SVector(p[3], (gc.J*p[SVector(4, 5, 6)])...), gc.g),
          B = () -> gc.J*p[SVector(7, 8, 9)])
    E(map(f -> f(), NamedTuple{K}(fs)))
end

struct HAMROldLayout end
struct HAMRNewLayout end

# H-AMR `parameters` (little-endian): a 264-byte header shared by both layouts, then either (old) i32[nmax] REFINED and
# i32[nmax] ACTIVE, or (new) an i32 triple (block id, time level, node) per active block in dump order.
# `dx` is the level-0 cell spacing recomputed from the grid extent, as H-AMR's own readers do.
function _hamr_parameters(path)
    open(path) do io
        rd(T) = ltoh(read(io, T))
        rdi() = Int(rd(Int32))
        t = rd(Float64)
        skip(io, 4); n_active_total = rdi(); skip(io, 4)    # n_active, nstep
        skip(io, 3*8 + 2*4 + 8 + 4)                         # Dtd, Dtl, Dtr, dump_cnt, rdump_cnt, dt, failed
        bs = ntuple(_ -> rdi(), 3); nmax = rdi(); nb = ntuple(_ -> rdi(), 3)
        startx = ntuple(_ -> rd(Float64), 3); dxstored = ntuple(_ -> rd(Float64), 3)
        skip(io, 8)                                         # tf
        a = rd(Float64); gam = rd(Float64); skip(io, 8)     # cour
        Rin = rd(Float64); Rout = rd(Float64); R0 = rd(Float64)
        @assert position(io) == 196

        old = filesize(path) == 264 + 8nmax
        old || filesize(path) == 264 + 12n_active_total || throw(ArgumentError(
            "$path: size $(filesize(path)) matches no H-AMR `parameters` layout: old 264 + 8·nmax = $(264 + 8nmax), new 264 + 12·n_active_total = $(264 + 12n_active_total)"))
        # the loader assumes r = exp(X1) (R0 = 0) starting at Rin, and an equatorially symmetric X2 grid inside [-1, 1]
        R0 == 0 || throw(ArgumentError("$path: R0 = $R0, only R0 = 0 (r = exp(X1)) is supported"))
        exp(startx[1]) ≈ Rin || throw(ArgumentError("$path: grid start X1 = $(startx[1]) is inconsistent with Rin = $Rin"))
        -1 < startx[2] < 0 || throw(ArgumentError("$path: X2 grid start $(startx[2]) is not in (-1, 0)"))
        dx = ((log(Rout) - log(Rin))/(bs[1]*nb[1]), -2startx[2]/(bs[2]*nb[2]), 2π/(bs[3]*nb[3]))
        # the new layout stores half the level-0 spacing
        all(map((s, d) -> s ≈ (old ? d : d/2), dxstored, dx)) ||
            throw(ArgumentError("$path: stored dx = $dxstored is inconsistent with the grid extent (level-0 spacing $dx)"))
        common = (; t, bs, nb, nmax, startx, dx, a, gam)

        if old
            # no physics flags here: the old layout's i32 at byte 256 is something else, e.g. 1, which would read as RAD_M1
            seek(io, 264 + 4nmax)
            active = ltoh.(read!(io, Vector{Int32}(undef, nmax))) .== 1
            count(active) == n_active_total || throw(ArgumentError("$path: $(count(active)) ACTIVE blocks ≠ n_active_total = $n_active_total"))
            (; common..., layout = HAMROldLayout(), ids = findall(active) .- 1)
        else
            # physics flags (RAD_M1, RESISTIVE, …): any set adds per-cell variables, which the loader does not read
            seek(io, 256); flags = rdi()
            flags == 0 || throw(ArgumentError("$path: physics flags $flags (extra per-cell variables) are not supported"))
            seek(io, 264)
            trip = Int.(ltoh.(read!(io, Matrix{Int32}(undef, 3, n_active_total))))
            (; common..., layout = HAMRNewLayout(), ids = trip[1, :], nodes = trip[3, :])
        end
    end
end

# Native coordinates from `rcyl`; cylindrification is supported for the old layout only (its corner x20 assumes one θ level).
_hamr_coords(_, ::Nothing, par) = HAMRSph()
_hamr_coords(::HAMROldLayout, rcyl::Real, par) = HAMRCyl(startx1 = par.startx[1], x10 = log(rcyl), x20 = -1 + 1/(par.nb[2]*par.bs[2]))
_hamr_coords(::HAMRNewLayout, rcyl::Real, par) = throw(ArgumentError("rcyl = $rcyl: cylindrified θ is not supported for the new H-AMR layout, pass rcyl = nothing"))

# The numbers N of the files `<prefix>N` in `dir`, sorted.
function _numbered(dir, prefix)
    re = Regex("^$prefix(\\d+)\$")
    sort(filtermap(f -> (m = match(re, f); isnothing(m) ? nothing : parse(Int, m[1])), readdir(dir)))
end

# Active blocks in dump order: refinement levels and coordinates per direction (r, θ, φ), and stored cells (memory-mapped,
# indexed φ, θ, r).
function _hamr_blocks(dir, par)
    (; bs, nb, ids) = par
    geom = _hamr_block_geometry(par.layout, dir, par)

    # dump files new_dump0, new_dump1, …: each holds a whole number of blocks, consecutive in dump order
    nums = _numbered(dir, "new_dump")
    nums == 0:length(nums) - 1 || throw(ArgumentError("$dir: new_dump files are not numbered 0, 1, … contiguously: $nums"))
    blockbytes = prod(bs)*sizeof(HAMRCell)
    locs = flatmap(nums) do u
        path = joinpath(dir, "new_dump$u")
        nblk, rem = divrem(filesize(path), blockbytes)
        rem == 0 || throw(ArgumentError("$path: size $(filesize(path)) is not a multiple of the block size $blockbytes"))
        cells = Mmap.mmap(path, Array{HAMRCell,4}, (reverse(bs)..., nblk))
        map(q -> (node = u, cells = view(cells, :, :, :, q)), 1:nblk)
    end
    length(locs) == length(ids) || throw(ArgumentError("$dir: new_dump files hold $(length(locs)) blocks ≠ $(length(ids)) active"))
    _hamr_check_nodes(par.layout, par, locs, dir)

    blocks = StructArrays.StructArray(map((id, g, loc) -> (; id, g.level, g.coord, loc.cells), ids, geom, locs))
    all(b -> all(d -> b.level[d] ≥ 0 && 0 ≤ b.coord[d] < nb[d]*2^b.level[d], 1:3), blocks) ||
        throw(ArgumentError("$dir: block coordinates out of range for their refinement levels"))
    # paint every block onto the finest-level grid, in block units: each slot must be covered exactly once
    L = ntuple(d -> maximum(b -> b.level[d], blocks), 3)
    cover = zeros(Int, nb .* 2 .^ L)
    foreach(b -> cover[ntuple(d -> _relevel(b.coord[d]:b.coord[d], L[d] - b.level[d]) .+ 1, 3)...] .+= 1, blocks)
    all(isone, cover) || throw(ArgumentError("$dir: active blocks do not tile the grid exactly once"))
    blocks
end

# New layout: `parameters` lists each block's node, i.e. its new_dump file.
_hamr_check_nodes(::HAMROldLayout, par, locs, dir) = nothing
_hamr_check_nodes(::HAMRNewLayout, par, locs, dir) = par.nodes == map(l -> l.node, locs) ||
    throw(ArgumentError("$dir: block nodes in `parameters` disagree with the new_dump file sizes"))

# Old layout: metadata in `dir/grid`, 36 i32 per block: 0 ACTIVE, 1 LEVEL, 2 REFINED, 3-5 COORD1..3, …
function _hamr_block_geometry(::HAMROldLayout, dir, par)
    gpath = joinpath(dir, "grid")
    recs = _hamr_grid_records(gpath, par.nmax, par.ids)
    all(isone, recs[1, :]) || throw(ArgumentError("$gpath: ACTIVE flags disagree with parameters"))
    map(g -> (level = (0, 0, g[2]), coord = (g[4], g[5], g[6])), eachcol(recs))
end

# New layout: metadata in the run's `gdumps/grid`, `grid1`, …: i32 3-5 COORD1..3, 110-112 LEVEL1..3 per block.
# Its ACTIVE flags are those of the time the grid was written, not of this dump.
function _hamr_block_geometry(::HAMRNewLayout, dir, par)
    gpath = joinpath(dirname(rstrip(dir, '/')), "gdumps", "grid")
    recs = _hamr_grid_records(gpath, par.nmax, par.ids)
    map(g -> (level = (g[111], g[112], g[113]), coord = (g[4], g[5], g[6])), eachcol(recs))
end

# The records `ids` (0-based) of an H-AMR grid file `gpath`, continued in the parts `gpath1`, `gpath2`, …: an i32 nmax
# header, then nmax records of nv i32 each, every part holding whole records. Returns the nv × length(ids) Int matrix.
function _hamr_grid_records(gpath, nmax, ids)
    isfile(gpath) || throw(ArgumentError("H-AMR grid file $gpath not found"))
    nums = _numbered(dirname(gpath), basename(gpath))
    nums == 1:length(nums) || throw(ArgumentError("$gpath: its parts 1, 2, … are not numbered contiguously: $nums"))
    paths = [gpath; map(u -> "$gpath$u", nums)]
    head = [4; zeros(Int, length(nums))]                 # bytes before the first record of each part
    sizes = filesize.(paths) .- head
    nv, rem = divrem(sum(sizes), 4nmax)
    rem == 0 || throw(ArgumentError("$gpath: total size $(sum(sizes) + 4) ≠ 4 + 4·nv·nmax for nmax = $nmax"))
    all(s -> s % 4nv == 0, sizes) || throw(ArgumentError("$gpath: its parts of sizes $sizes do not each hold whole records of $nv i32"))
    n = open(io -> ltoh(read(io, Int32)), gpath)
    n == nmax || throw(ArgumentError("$gpath: nmax = $n ≠ $nmax from parameters"))
    firstid = cumsum([0; sizes[1:end-1] .÷ 4nv])         # first record of each part
    recs = map((path, h, sz) -> Mmap.mmap(path, Matrix{Int32}, (nv, sz ÷ 4nv), h), paths, head, sizes)
    stack(ids) do id
        p = searchsortedlast(firstid, id)
        Int.(ltoh.(view(recs[p], :, id - firstid[p] + 1)))
    end
end

"""
    r_min(snapshot)

Minimum radial grid coordinate.

# Arguments
- `snapshot`: Loaded `GRMHDSnapshot`.
"""
r_min(s::GRMHDSnapshot) = leftendpoint(s.r_extent)

"""
    r_max(snapshot)

Maximum radial grid coordinate.

# Arguments
- `snapshot`: Loaded `GRMHDSnapshot`.
"""
r_max(s::GRMHDSnapshot) = rightendpoint(s.r_extent)

# GPU-safe searchsortedlast.
@inline function bsearchlast(a, x)
    lo = 1; hi = length(a)
    @inbounds while lo < hi
        mid = (lo + hi + 1) >> 1
        a[mid] ≤ x ? (lo = mid) : (hi = mid - 1)
    end
    return lo
end

# Avoids a device view.
@inline function bsearchlast_row(a, ir, x, n)
    lo = 1; hi = n
    @inbounds while lo < hi
        mid = (lo + hi + 1) >> 1
        a[ir, mid] ≤ x ? (lo = mid) : (hi = mid - 1)
    end
    return lo
end

# Trilinear stencil at Cartesian (x,y,z): bracket indices + weights, field-independent.
# `n3` = φ count (same for every field). Returns `nothing` outside the radial grid.
@inline function _sample_stencil(s::GRMHDSnapshot, n3, x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    (r < r_min(s) || r > r_max(s)) && return nothing
    θ = acos(clamp(z/r, -one(r), one(r)))
    φ = atan(y, x)
    n2 = size(s.th_grid, 2)

    i = clamp(bsearchlast(s.r_prof, r), 1, length(s.r_prof) - 1)
    ti = (r - s.r_prof[i])/(s.r_prof[i+1] - s.r_prof[i])
    # Keep NamedDims indices homogeneous for Metal.
    fk = (φ - s.φ0)/s.dφ
    k0 = unsafe_trunc(Int, floor(fk)); tk = fk - k0
    k = mod(k0, n3) + 1; kp = mod(k0 + 1, n3) + 1

    θbracket(ir) = @inbounds begin
        j = clamp(bsearchlast_row(s.th_grid, ir, θ, n2), 1, n2 - 1)
        a = s.th_grid[ir, j]
        (j, clamp((θ - a)/(s.th_grid[ir, j+1] - a), zero(θ), one(θ)))
    end
    ji, tji = θbracket(i); jp, tjp = θbracket(i + 1)
    return (; i, ti, k, kp, tk, ji, tji, jp, tjp)
end

# Elements are numbers, SVectors, or NamedTuple records of those (interpolated field by field).
@inline _lerp(a, b, t) = a*(1 - t) + b*t
@inline _lerp(a::NamedTuple{K}, b::NamedTuple{K}, t) where {K} = map((x, y) -> _lerp(x, y, t), a, b)

# Not `zero`: Base has no `zero(::Type{<:NamedTuple})`, and defining one would be piracy.
_zero(::Type{T}) where {T} = zero(T)
_zero(::Type{T}) where {T<:NamedTuple} = T(map(_zero, fieldtypes(T)))

# Apply a precomputed stencil to a field array.
@inline function _apply_stencil(st, A)
    face(ir, jj, tj) = @inbounds _lerp(_lerp(A[φ=st.k, θ=jj, r=ir], A[φ=st.k, θ=jj+1, r=ir], tj),
                                       _lerp(A[φ=st.kp, θ=jj, r=ir], A[φ=st.kp, θ=jj+1, r=ir], tj), st.tk)
    _lerp(face(st.i, st.ji, st.tji), face(st.i + 1, st.jp, st.tjp), st.ti)
end

"""
    sample(snapshot, x, y, z)

Trilinearly interpolate the fluid record at Cartesian position `(x, y, z)`, field by field.

# Arguments
- `snapshot`: Loaded `GRMHDSnapshot`, with any subset of the fluid fields.
- `x`, `y`, `z`: Cartesian coordinates in gravitational radii.

# Returns
- Record with the keys of `snapshot.fluid`, e.g. `(; ρ, ũ, B)`, in the grid precision (`eltype(snapshot.r_prof)`); zeros
  outside the grid.
"""
@inline function sample(s::GRMHDSnapshot, x, y, z)
    A = s.fluid
    st = _sample_stencil(s, size(A, :φ), x, y, z)
    # `_lerp` with a weight promotes the out-of-grid zero to the interpolated type (Float32 field → Float64).
    z0 = _zero(eltype(A))
    st === nothing ? _lerp(z0, z0, one(eltype(s.r_prof))) : _apply_stencil(st, A)
end

include("physics.jl")

Adapt.adapt_structure(to, s::GRMHDSnapshot) = GRMHDSnapshot(
    Adapt.adapt(to, s.fluid), Adapt.adapt(to, s.r_prof), s.r_extent, Adapt.adapt(to, s.th_grid),
    s.φ0, s.dφ, s.R, s.spin, s.gam, s.t, s.M_unit, s.MBH)

end
