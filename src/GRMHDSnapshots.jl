module GRMHDSnapshots

using HDF5
using StaticArrays
using Unitful, UnitfulAstro
using AxisKeys: NamedDimsArray, dimnames
using DataManipulation: mapview
using OhMyThreads: tmap
import StructArrays
import Adapt
using Zarr: zopen, ZGroup
using ZarrZfp

export KoralSnapshot, load_koral, load_grid, r_min, r_max,
    plasma_state, map_plasma, fluid_ucon, fluid_velocity, flow_speed, proper_velocity, lapse, grav_redshift, volume_ratio, lunit, rhounit, bunit,
    comoving_bsq, comoving_B_gauss, bfield_magnitude, magnetization

struct KoralSnapshot{Ar, Av, Ab, Rp, Tg, T}
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
    M_unit::T
    MBH::T
end

"""
    load_koral(path; kwargs...)

Load a raw `.h5` snapshot or a Zarr snapshot path.

# Arguments
- `path`: `.h5` file, or `.../RUN.zarr/snapNNNN` store snapshot path.
- `kwargs...`: Forwarded to the selected loader.

# Returns
- `KoralSnapshot`.
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
        KoralSnapshot(rho, velrel, bfield, r_prof, th_grid, φ0, dφ,
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
- `KoralSnapshot`; unselected fields are `nothing`.
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
    KoralSnapshot(rho, velrel, bfield, grid.r_prof, grid.th_grid, grid.φ0, grid.dφ,
        sc(hdr["bhspin"]), sc(hdr["gam"]), sc(snap.attrs["t"]),
        sc(units["M_unit"]), ustrip(u"g", sc(units["M_bh"]) * u"Msun"))
end

"""
    r_min(snapshot)

Minimum radial grid coordinate.

# Arguments
- `snapshot`: Loaded `KoralSnapshot`.
"""
r_min(s::KoralSnapshot) = @inbounds s.r_prof[1]

"""
    r_max(snapshot)

Maximum radial grid coordinate.

# Arguments
- `snapshot`: Loaded `KoralSnapshot`.
"""
r_max(s::KoralSnapshot) = @inbounds s.r_prof[end]

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
@inline function _sample_stencil(s::KoralSnapshot, n3, x, y, z)
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
- `snapshot`: Loaded `KoralSnapshot`.
- `field_array`: One `(φ, θ, r)` field array, e.g. `snap.rho`, `snap.velrel`, or `snap.bfield`.
- `x`, `y`, `z`: Cartesian coordinates in gravitational radii.

# Returns
- Interpolated value; `zero(eltype(field_array))` outside the grid.
"""
@inline function sample_field(s::KoralSnapshot, A, x, y, z)
    st = _sample_stencil(s, size(A, :φ), x, y, z)
    # `* one(weight)` promotes the out-of-grid zero to the interpolated type (Float32 field → Float64).
    st === nothing ? zero(eltype(A)) * one(eltype(s.r_prof)) : _apply_stencil(st, A)
end

"""
    sample(snapshot, x, y, z)

Interpolate `(rho, velrel, bfield)` at Cartesian position `(x, y, z)`.

# Arguments
- `snapshot`: Loaded `KoralSnapshot` with all fluid fields present.
- `x`, `y`, `z`: Cartesian coordinates in gravitational radii.

# Returns
- Tuple `(rho, velrel, bfield)`.
"""
@inline function sample(s::KoralSnapshot, x, y, z)
    st = _sample_stencil(s, size(s.rho, :φ), x, y, z)
    if st === nothing
        # Match the interpolation result type.
        T = promote_type(eltype(s.rho), eltype(s.r_prof)); z3 = zero(SVector{3,T})
        return (zero(T), z3, z3)
    end
    return (_apply_stencil(st, s.rho), _apply_stencil(st, s.velrel), _apply_stencil(st, s.bfield))
end

include("physics.jl")

Adapt.adapt_structure(to, x::NamedDimsArray) = NamedDimsArray(Adapt.adapt(to, parent(x)), dimnames(x))

Adapt.adapt_structure(to, s::KoralSnapshot) = KoralSnapshot(
    Adapt.adapt(to, s.rho), Adapt.adapt(to, s.velrel), Adapt.adapt(to, s.bfield),
    Adapt.adapt(to, s.r_prof), Adapt.adapt(to, s.th_grid),
    s.φ0, s.dφ, s.spin, s.gam, s.t, s.M_unit, s.MBH)

end
