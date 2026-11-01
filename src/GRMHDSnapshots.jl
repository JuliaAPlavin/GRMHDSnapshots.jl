module GRMHDSnapshots

using HDF5
using StaticArrays
using Unitful, UnitfulAstro
using NamedDims: NamedDimsArray
using DataManipulation: mapview, filtermap, flatmap
using OhMyThreads: tmap, tforeach, @tasks, @local
import StructArrays
import ForwardDiff
import Adapt
using Zarr: zopen, ZGroup
using ZarrZfp

export GRMHDSnapshot, load_snapshot, load_koral, load_iharm, load_hamr, load_grid, r_min, r_max,
    plasma_state, map_plasma, fluid_ucon, fluid_velocity, flow_speed, proper_velocity, lapse, grav_redshift, volume_ratio, ks_to_bl, lunit, rhounit, bunit,
    comoving_bsq, comoving_B_gauss, bfield_magnitude, magnetization

include("coordinates.jl")

# `M_unit`/`MBH` each carry their own type parameter (`Um`, `Ub`): `Float64` when known, `Missing` when
# not — scale-free dumps (e.g. iharm3d/KHARMA, H-AMR) store neither, and each can be supplied independently.
# The unit scalars needing a missing one (`lunit`: MBH; `rhounit`/`bunit`: both) throw rather than return `missing`.
struct GRMHDSnapshot{Ar, Av, Ab, Rp, Tg, T, Um, Ub}
    rho::Ar
    velrel::Av
    bfield::Ab
    r_prof::Rp
    th_grid::Tg
    φ0::T
    dφ::T
    spin::T
    gam::T
    t::T
    M_unit::Um
    MBH::Ub
end

"""
    load_snapshot(path; kwargs...)

Load a single GRMHD snapshot, auto-detecting the on-disk format:
- `.h5` with a `prims` dataset → iharm3d/KHARMA FMKS ([`load_iharm`](@ref)), read in memory;
- `.h5` with a `quants/` group → koral ([`load_koral`](@ref)), memory-mapped;
- a `.../RUN.zarr/snapNNNN` path → koral Zarr ([`load_koral`](@ref));
- a directory with `parameters`, `grid` and `new_dump0` → H-AMR ([`load_hamr`](@ref); needs its `rcyl` kwarg).

Memory-mapping is chosen for the caller: koral `.h5` is mmapped, iharm reads in memory (its
velocity/field components are reconstructed, not stored, so there is nothing to map). `kwargs...`
go to the selected loader (e.g. `M_unit`/`MBH` for iharm); use the explicit loaders for full control.
"""
function load_snapshot(path::AbstractString; kwargs...)
    p = rstrip(path, '/')
    fmt = snapshot_format(p)
    fmt === :iharm && return load_iharm(p; kwargs...)
    fmt === :hamr && return load_hamr(p; kwargs...)
    endswith(p, ".h5") ? load_koral(p; mmap = true, kwargs...) : load_koral(p; kwargs...)
end

"""
    snapshot_format(path) -> Symbol

Detect a snapshot's on-disk format: `:iharm` for an iharm3d/KHARMA `.h5` dump (a `prims` dataset),
`:hamr` for an H-AMR dump directory (`parameters`, `grid`, `new_dump0`), or `:koral` for the koral
layout (a `.h5` with a `quants/` group, or a Zarr snapshot path).
"""
function snapshot_format(path::AbstractString)
    p = rstrip(path, '/')
    all(f -> isfile(joinpath(p, f)), ("parameters", "grid", "new_dump0")) && return :hamr
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

function load_koral_h5(path::AbstractString; mmap=false)
    h5open(path, "r") do h5
        nd(A) = NamedDimsArray(A, (:φ, :θ, :r))
        if mmap
            # mmap bytes keep file endianness.
            order = HDF5.API.h5t_get_order(HDF5.datatype(h5["quants/rho"]).id)
            swap = order == HDF5.API.H5T_ORDER_BE ? ntoh : ltoh
            mm(name) = HDF5.readmmap(h5["quants/$name"])
            rho = nd(mapview(swap, mm("rho")))
            svf(a, b, c) = nd(mapview(v -> swap.(v),
                StructArrays.StructArray{SVector{3,Float64}}((mm(a), mm(b), mm(c)))))
            velrel = svf("U1", "U2", "U3"); bfield = svf("B1", "B2", "B3")
        else
            rd(name) = read(h5["quants/$name"])
            rho = nd(rd("rho"))
            sv3(a, b, c) = nd(SVector{3,Float64}.(a, b, c))
            velrel = sv3(rd("U1"), rd("U2"), rd("U3")); bfield = sv3(rd("B1"), rd("B2"), rd("B3"))
        end

        r = read(h5["grid_out/r"]); th = read(h5["grid_out/th"]); ph = read(h5["grid_out/ph"])
        n1 = size(r, 3); n2 = size(r, 2); n3 = size(r, 1)
        r_prof  = Float64[r[1, 1, i] for i in 1:n1]
        th_grid = Float64[th[1, j, i] for i in 1:n1, j in 1:n2]
        φ0_file = Float64(ph[1, 1, 1]); dφ = (Float64(ph[n3, 1, 1]) - φ0_file)/(n3 - 1)
        # ipole convention: file φ + π.
        φ0 = φ0_file + π

        rdscalar(name) = Float64(read(h5[name]))
        GRMHDSnapshot(rho, velrel, bfield, r_prof, th_grid, φ0, dφ,
            rdscalar("header/bhspin"), rdscalar("header/gam"), rdscalar("t"),
            rdscalar("header/units/M_unit"), ustrip(u"g", rdscalar("header/units/M_bh") * u"Msun"))
    end
end

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
    load_koral(store, snapname; fields=(:rho, :velrel, :bfield), grid=load_grid(store))

Load one snapshot from an open Zarr run store.

# Arguments
- `store`: Open Zarr run store.
- `snapname`: Snapshot group name, e.g. `"snap002000"`.
- `fields`: Fluid fields to read; subset of `(:rho, :velrel, :bfield)`.
- `grid`: Cached grid from `load_grid(store)`.

# Returns
- `GRMHDSnapshot`; unselected fields are `nothing`.
"""
function load_koral(store::ZGroup, snapname::AbstractString;
                    fields = (:rho, :velrel, :bfield), grid = load_grid(store))
    fields ⊆ (:rho, :velrel, :bfield) || throw(ArgumentError("unknown fluid field(s) in $fields"))
    snap = store[snapname]::ZGroup
    nd(A) = NamedDimsArray(A, (:φ, :θ, :r))
    rd(name) = Threads.@spawn snap[name][:, :, :]                # each zfp chunk decompresses independently
    svf(ts) = nd(StructArrays.StructArray{SVector{3,Float32}}(map(fetch, ts)))
    # launch every requested component up front so the reads decompress concurrently, then assemble
    trho = :rho    in fields ? rd("rho")                   : nothing
    tvel = :velrel in fields ? map(rd, ("U1", "U2", "U3")) : nothing
    tbf  = :bfield in fields ? map(rd, ("B1", "B2", "B3")) : nothing
    rho    = isnothing(trho) ? nothing : nd(fetch(trho))
    velrel = isnothing(tvel) ? nothing : svf(tvel)
    bfield = isnothing(tbf)  ? nothing : svf(tbf)

    sc(x) = Float64(x)
    hdr = store.attrs; units = hdr["units"]
    GRMHDSnapshot(rho, velrel, bfield, grid.r_prof, grid.th_grid, grid.φ0, grid.dφ,
        sc(hdr["bhspin"]), sc(hdr["gam"]), sc(snap.attrs["t"]),
        sc(units["M_unit"]), ustrip(u"g", sc(units["M_bh"]) * u"Msun"))
end

"""
    load_iharm(path; fields=(:rho, :velrel, :bfield), M_unit=missing, MBH=missing)

Load an iharm3d/KHARMA `.h5` dump in FMKS coordinates (`header/metric` == `"FMKS"`/`"MMKS"`).

The packed `prims` array holds the primitives in *native* FMKS components; the relative velocity
`ũ^i` and lab field `B^i` are transformed to Kerr-Schild `(r, θ, φ)` components at load via the
spatial Jacobian `J₃(r, θ)` (matching ipole's reconstruction), so the result is an ordinary
[`GRMHDSnapshot`] that all downstream physics/interpolation handles unchanged.

# Arguments
- `fields`: fluid fields to read; subset of `(:rho, :velrel, :bfield)`. Unselected are `nothing`.
- `M_unit`, `MBH`: physical units in grams, each optional (default `missing`). These dumps are scale-free
  (no units stored): `MBH` enables `lunit`, both together `rhounit`/`bunit`; those missing a unit throw.

# Returns
- `GRMHDSnapshot`.
"""
function load_iharm(path::AbstractString; fields = (:rho, :velrel, :bfield), M_unit = missing, MBH = missing)
    fields ⊆ (:rho, :velrel, :bfield) || throw(ArgumentError("unknown fluid field(s) in $fields"))
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
        nd(A) = NamedDimsArray(A, (:φ, :θ, :r))
        rho = :rho in fields ? nd([prims[1, k, j, i] for k in 1:n3, j in 1:n2, i in 1:n1]) : nothing
        # native contravariant 3-vector → KS: v_KS = J₃·v_nat, J₃(r,θ) built once per (i,j) and reused over φ.
        J = [jacobian_spatial(coords, X1(i), X2(j)) for i in 1:n1, j in 1:n2]
        tovec(s1, s2, s3) = nd([J[i, j] * SVector(prims[s1, k, j, i], prims[s2, k, j, i], prims[s3, k, j, i])
                                for k in 1:n3, j in 1:n2, i in 1:n1])
        velrel = :velrel in fields ? tovec(3, 4, 5) : nothing
        bfield = :bfield in fields ? tovec(6, 7, 8) : nothing

        sc(x) = Float64(x)   # heterogeneous header scalars → uniform Float64 struct fields
        # φ = X3 (zone-centered); no +π offset (that is specific to the koral converter).
        GRMHDSnapshot(rho, velrel, bfield, r_prof, th_grid, sc(sx3 + 0.5*dx3), sc(dx3),
            sc(fmks("a")), sc(rd("header/gam")), sc(rd("t")), M_unit, MBH)
    end
end

"""
    load_hamr(dir; rcyl, rmin=nothing, rmax=nothing, fields=(:rho, :velrel, :bfield), M_unit=missing, MBH=missing)

Load an H-AMR native dump directory (`parameters`, `grid`, `new_dump0`, `new_dump1`, …) at native resolution,
restricted to the radial cells whose centers lie in `[rmin, rmax]`. Only the blocks overlapping that range
are read, and only their radial slabs within it.

H-AMR refines blocks in φ only, coarser towards the poles. The result is an ordinary uniform
[`GRMHDSnapshot`] on the finest φ grid: each coarser θ row is upsampled by periodic linear interpolation in
φ between its coarse cell centers; finest rows are copied as-is.

The dump stores the 4-velocity `u^μ` and lab field `B^i` in native [`HAMRCyl`](@ref) components; they are
transformed to Kerr-Schild components via the analytic spatial Jacobian, and `u^μ` is converted to the
relative velocity `ũ^i`. Fluid arrays are `Float32` (the dump precision); transforms are computed in `Float64`.

# Arguments
- `rcyl`: the run's cylindrification radius (HARMPI `global_x10 = log(rcyl)`; 24.32 for the EHT
  code-comparison run). Required: it sets θ near the poles but is not stored in the dump files.
  The other unstored geometry is assumed: cylindrified Gammie θ with hslope = 1, and HARMPI's default
  corner `x20 = -1 + 1/n2` (`n2` the total θ cell count).
- `rmin`, `rmax`: radial range of cell centers; `nothing` extends to the grid edge.
- `fields`: fluid fields to read; subset of `(:rho, :velrel, :bfield)`. Unselected are `nothing`.
- `M_unit`, `MBH`: physical units in grams, each optional, as for [`load_iharm`](@ref) (the dumps store none).

# Returns
- `GRMHDSnapshot`.
"""
function load_hamr(dir::AbstractString; rcyl, rmin = nothing, rmax = nothing,
                   fields = (:rho, :velrel, :bfield), M_unit = missing, MBH = missing)
    fields ⊆ (:rho, :velrel, :bfield) || throw(ArgumentError("unknown fluid field(s) in $fields"))
    isempty(fields) && throw(ArgumentError("no fluid fields requested"))
    par = _hamr_parameters(joinpath(dir, "parameters"))
    (; blocks, lmax, rowlevel) = _hamr_blocks(dir, par)
    (; bs, nb, startx, dx) = par
    n1, n2, n3 = nb[1]*bs[1], nb[2]*bs[2], nb[3]*bs[3]*2^lmax
    coords = HAMRCyl(startx1 = startx[1], x10 = log(rcyl), x20 = -1 + 1/n2)

    # zone-centered native coords (Julia index i → 0-based i-1); r = exp(X1) depends on X1 only
    X1 = startx[1] .+ ((1:n1) .- 1/2) .* dx[1]
    X2 = startx[2] .+ ((1:n2) .- 1/2) .* dx[2]
    sel = findall(x -> something(rmin, -Inf) ≤ exp(x) ≤ something(rmax, Inf), X1)
    isempty(sel) && throw(ArgumentError(
        "no radial cell centers in [$rmin, $rmax]; the grid spans r = $(exp(first(X1))) … $(exp(last(X1)))"))
    ir = first(sel):last(sel)
    r_prof  = exp.(X1[ir])
    th_grid = [last(bl_coord(coords, x1, x2)) for x1 in X1[ir], x2 in X2]
    J       = [jacobian_spatial(coords, x1, x2) for x1 in X1[ir], x2 in X2]

    sz = (n3, n2, length(ir))
    svf() = StructArrays.StructArray{SVector{3,Float32}}(ntuple(_ -> Array{Float32}(undef, sz), 3))
    rho    = :rho    in fields ? Array{Float32}(undef, sz) : nothing
    velrel = :velrel in fields ? svf() : nothing
    bfield = :bfield in fields ? svf() : nothing

    # pass 1: every coarse cell m of a block lands at its first fine φ slot m·s
    outs = (; rho, velrel, bfield)
    todo = filter(b -> !isempty(intersect(ir, b.c1*bs[1] .+ (1:bs[1]))), blocks)
    @tasks for b in todo
        @local buf = Vector{SVector{9,Float32}}(undef, prod(bs))
        _hamr_read_block!(outs, buf, b, bs, 2^(lmax - b.level), ir, r_prof, th_grid, J, par.a)
    end
    # pass 2: fill the coarse rows in place
    srow = [2^(lmax - rowlevel[cld(j, bs[2])]) for j in 1:n2]
    coarse = findall(>(1), srow)
    comps = (_components(rho)..., _components(velrel)..., _components(bfield)...)
    tforeach(axes(r_prof, 1)) do i
        for j in coarse, A in comps
            _upsample_row!(view(A, :, j, i), srow[j])
        end
    end

    nd(A) = isnothing(A) ? nothing : NamedDimsArray(A, (:φ, :θ, :r))
    dφ = dx[3]/2^lmax
    # φ = X3 (zone-centered); no +π offset (that is specific to the koral converter).
    GRMHDSnapshot(nd(rho), nd(velrel), nd(bfield), r_prof, th_grid, startx[3] + dφ/2, dφ,
        par.a, par.gam, par.t, M_unit, MBH)
end

_components(::Nothing) = ()
_components(A::Array) = (A,)
_components(A::StructArrays.StructArray) = Tuple(StructArrays.components(A))

# H-AMR `parameters` (little-endian), old layout: a 264-byte header, then i32[nmax] REFINED and i32[nmax] ACTIVE.
# The newer layout (active-block triples/pairs after the header) is not supported.
function _hamr_parameters(path)
    open(path) do io
        rd(T) = ltoh(read(io, T))
        rdi() = Int(rd(Int32))
        t = rd(Float64)
        skip(io, 4); n_active_total = rdi(); skip(io, 4)    # n_active, nstep
        skip(io, 3*8 + 2*4 + 8 + 4)                         # Dtd, Dtl, Dtr, dump_cnt, rdump_cnt, dt, failed
        bs = ntuple(_ -> rdi(), 3); nmax = rdi(); nb = ntuple(_ -> rdi(), 3)
        startx = ntuple(_ -> rd(Float64), 3); dx = ntuple(_ -> rd(Float64), 3)
        skip(io, 8)                                         # tf
        a = rd(Float64); gam = rd(Float64); skip(io, 8)     # cour
        Rin = rd(Float64); Rout = rd(Float64); R0 = rd(Float64)
        @assert position(io) == 196

        filesize(path) == 264 + 8nmax || throw(ArgumentError(
            "$path: size $(filesize(path)) ≠ 264 + 8·nmax = $(264 + 8nmax); only the old H-AMR `parameters` layout is supported"))
        seek(io, 264 + 4nmax)
        active = ltoh.(read!(io, Vector{Int32}(undef, nmax))) .== 1
        count(active) == n_active_total || throw(ArgumentError("$path: $(count(active)) ACTIVE blocks ≠ n_active_total = $n_active_total"))

        # the loader assumes r = exp(X1) (R0 = 0) spanning [Rin, Rout], and a full periodic φ circle
        R0 == 0 || throw(ArgumentError("$path: R0 = $R0, only R0 = 0 (r = exp(X1)) is supported"))
        (exp(startx[1]) ≈ Rin && exp(startx[1] + nb[1]*bs[1]*dx[1]) ≈ Rout) ||
            throw(ArgumentError("$path: grid X1 ∈ [$(startx[1]), +$(nb[1]*bs[1])·$(dx[1])] is inconsistent with r ∈ [$Rin, $Rout]"))
        nb[3]*bs[3]*dx[3] ≈ 2π || throw(ArgumentError("$path: φ grid spans $(nb[3]*bs[3]*dx[3]) ≠ 2π"))
        # θ mirroring assumes an equatorially symmetric X2 grid inside [-1, 1]
        (startx[2] > -1 && startx[2] + nb[2]*bs[2]*dx[2] ≈ -startx[2]) ||
            throw(ArgumentError("$path: X2 grid [$(startx[2]), +$(nb[2]*bs[2])·$(dx[2])] is not symmetric inside [-1, 1]"))
        (; t, n_active_total, bs, nb, nmax, startx, dx, a, gam, active)
    end
end

# Active blocks in dump order (ascending `id`) with their file and offset. φ-only refinement: each θ row has one level.
function _hamr_blocks(dir, par)
    (; bs, nb, nmax, active, n_active_total) = par
    gpath = joinpath(dir, "grid")
    nv = 36                                            # i32 per block: 0 ACTIVE, 1 LEVEL, 2 REFINED, 3-5 COORD1..3, …
    filesize(gpath) == 4 + 4*nv*nmax || throw(ArgumentError("$gpath: size $(filesize(gpath)) ≠ 4 + 4·$nv·nmax"))
    grid = open(gpath) do io
        n = ltoh(read(io, Int32))
        n == nmax || throw(ArgumentError("$gpath: nmax = $n ≠ $nmax from parameters"))
        ltoh.(read!(io, Matrix{Int32}(undef, nv, nmax)))
    end
    (grid[1, :] .== 1) == active || throw(ArgumentError("$gpath: ACTIVE flags disagree with parameters"))
    ids = findall(active)

    # dump files new_dump0, new_dump1, …: each holds a whole number of blocks, consecutive in dump order
    nums = sort(filtermap(f -> (m = match(r"^new_dump(\d+)$", f); isnothing(m) ? nothing : parse(Int, m[1])), readdir(dir)))
    nums == 0:length(nums) - 1 || throw(ArgumentError("$dir: new_dump files are not numbered 0, 1, … contiguously: $nums"))
    blockbytes = 9*prod(bs)*sizeof(Float32)
    locs = flatmap(nums) do u
        path = joinpath(dir, "new_dump$u")
        nblk, rem = divrem(filesize(path), blockbytes)
        rem == 0 || throw(ArgumentError("$path: size $(filesize(path)) is not a multiple of the block size $blockbytes"))
        map(q -> (file = path, offset = q*blockbytes), 0:nblk-1)
    end
    length(locs) == n_active_total || throw(ArgumentError("$dir: new_dump files hold $(length(locs)) blocks ≠ $n_active_total active"))

    blocks = StructArrays.StructArray(map(ids, locs) do n, loc
        (; id = n - 1, level = Int(grid[2, n]), c1 = Int(grid[4, n]), c2 = Int(grid[5, n]), c3 = Int(grid[6, n]), loc...)
    end)
    lmax = maximum(blocks.level)
    rowlevel = map(0:nb[2]-1) do c2
        ls = unique(blocks.level[blocks.c2 .== c2])
        length(ls) == 1 || throw(ArgumentError("$gpath: θ row $c2 has blocks at levels $ls; expected a single level"))
        only(ls)
    end
    all(b -> 0 ≤ b.c1 < nb[1] && 0 ≤ b.c2 < nb[2] && 0 ≤ b.c3 < nb[3]*2^b.level, blocks) && allunique(zip(blocks.c1, blocks.c2, blocks.c3)) &&
        length(blocks) == nb[1]*sum(l -> nb[3]*2^l, rowlevel) ||
        throw(ArgumentError("$gpath: active blocks do not tile the grid with φ-only refinement"))
    (; blocks, lmax, rowlevel)
end

# Read the radial slab of block `b` overlapping the selected global radial cells `ir`, transform it to KS and
# store each coarse cell m at fine φ slot m·s. Cells are 9 interleaved float32 (rho, ug, u^0..3, B^1..3), k fastest.
function _hamr_read_block!(outs, buf, b, bs, s, ir, r_prof, th_grid, J, a)
    (; rho, velrel, bfield) = outs
    bs1, bs2, bs3 = bs
    ii = intersect(ir, b.c1*bs1 .+ (1:bs1))
    slab = view(buf, 1:length(ii)*bs2*bs3)
    open(b.file) do io
        seek(io, b.offset + (first(ii) - b.c1*bs1 - 1)*bs2*bs3*sizeof(eltype(buf)))
        read!(io, slab)
    end
    cells = reshape(slab, bs3, bs2, length(ii))
    for (il, i) in enumerate(ii), jj in 1:bs2
        io = i - first(ir) + 1; j = b.c2*bs2 + jj
        Jc = J[io, j]; g = ks_gcov(r_prof[io], th_grid[io, j], a)
        for kk in 1:bs3
            p = ltoh.(cells[kk, jj, il])
            k = (b.c3*bs3 + kk - 1)*s + 1
            isnothing(rho)    || (rho[k, j, io] = p[1])
            # u^t is unchanged: the native↔KS Jacobian has a trivial t row/column
            isnothing(velrel) || (velrel[k, j, io] = gammie_uprim(SVector(p[3], (Jc*SVector(p[4], p[5], p[6]))...), g))
            isnothing(bfield) || (bfield[k, j, io] = Jc*SVector(p[7], p[8], p[9]))
        end
    end
end

# In-place periodic linear upsampling of one φ row whose coarse cells (each spanning `s` fine cells) sit at fine
# slots 1, 1+s, …: every fine center is interpolated between its two neighbouring coarse centers.
function _upsample_row!(row::AbstractVector, s::Integer)
    coarse = row[1:s:end]
    nc = length(coarse)
    map!(row, eachindex(row)) do k
        x = (k - 1/2)/s - 1/2           # fine center k in coarse units: 0-based coarse cell m is centered at x = m
        m = floor(Int, x); t = x - m
        (1 - t)*coarse[mod(m, nc) + 1] + t*coarse[mod(m + 1, nc) + 1]
    end
end

"""
    r_min(snapshot)

Minimum radial grid coordinate.

# Arguments
- `snapshot`: Loaded `GRMHDSnapshot`.
"""
r_min(s::GRMHDSnapshot) = @inbounds s.r_prof[1]

"""
    r_max(snapshot)

Maximum radial grid coordinate.

# Arguments
- `snapshot`: Loaded `GRMHDSnapshot`.
"""
r_max(s::GRMHDSnapshot) = @inbounds s.r_prof[end]

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

# Apply a precomputed stencil to a single field array.
@inline function _apply_stencil(st, A)
    face(ir, jj, tj) = @inbounds (A[φ=st.k, θ=jj, r=ir]*(1-tj) + A[φ=st.k, θ=jj+1, r=ir]*tj)*(1-st.tk) +
                                 (A[φ=st.kp, θ=jj, r=ir]*(1-tj) + A[φ=st.kp, θ=jj+1, r=ir]*tj)*st.tk
    face(st.i, st.ji, st.tji)*(1 - st.ti) + face(st.i + 1, st.jp, st.tjp)*st.ti
end

"""
    sample_field(snapshot, field_array, x, y, z)

Interpolate a single fluid field array (e.g. `snap.velrel`) at Cartesian `(x, y, z)`.

# Arguments
- `snapshot`: Loaded `GRMHDSnapshot`.
- `field_array`: One `(φ, θ, r)` field array, e.g. `snap.rho`, `snap.velrel`, or `snap.bfield`.
- `x`, `y`, `z`: Cartesian coordinates in gravitational radii.

# Returns
- Interpolated value; `zero(eltype(field_array))` outside the grid.
"""
@inline function sample_field(s::GRMHDSnapshot, A, x, y, z)
    st = _sample_stencil(s, size(A, :φ), x, y, z)
    # `* one(weight)` promotes the out-of-grid zero to the interpolated type (Float32 field → Float64).
    st === nothing ? zero(eltype(A)) * one(eltype(s.r_prof)) : _apply_stencil(st, A)
end

"""
    sample(snapshot, x, y, z)

Interpolate `(rho, velrel, bfield)` at Cartesian position `(x, y, z)`.

# Arguments
- `snapshot`: Loaded `GRMHDSnapshot` with all fluid fields present.
- `x`, `y`, `z`: Cartesian coordinates in gravitational radii.

# Returns
- Tuple `(rho, velrel, bfield)`.
"""
@inline function sample(s::GRMHDSnapshot, x, y, z)
    st = _sample_stencil(s, size(s.rho, :φ), x, y, z)
    if st === nothing
        # Match the interpolation result type.
        T = promote_type(eltype(s.rho), eltype(s.r_prof)); z3 = zero(SVector{3,T})
        return (zero(T), z3, z3)
    end
    return (_apply_stencil(st, s.rho), _apply_stencil(st, s.velrel), _apply_stencil(st, s.bfield))
end

include("physics.jl")

Adapt.adapt_structure(to, s::GRMHDSnapshot) = GRMHDSnapshot(
    Adapt.adapt(to, s.rho), Adapt.adapt(to, s.velrel), Adapt.adapt(to, s.bfield),
    Adapt.adapt(to, s.r_prof), Adapt.adapt(to, s.th_grid),
    s.φ0, s.dφ, s.spin, s.gam, s.t, s.M_unit, s.MBH)

end
