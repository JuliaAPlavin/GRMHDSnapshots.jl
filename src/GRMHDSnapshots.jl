module GRMHDSnapshots

# Loader + spatial sampler for ipole-format GRMHD HDF5 snapshots (KORAL, Narayan et al. 2022).
#
# The grid is a tensor product in code-coordinate index space (x1,x2,x3) → (r,θ,φ):
#   r = r(i)        monotonic in the radial index i  (independent of j,k)
#   θ = θ(i,j)      monotonic in the polar index j    (only a weak i-dependence near the poles)
#   φ = φ(k)        uniform in the azimuthal index k
# A Cartesian query point is mapped to a grid location by inverting these three 1-D maps. The fluid
# primitives (rho, the relative 4-velocity ũ^i, the lab 3-field B^i — all in Kerr-Schild components)
# are interpolated at the point.
#
# The big fields are kept in their NATIVE on-disk axis order (φ,θ,r) — `read`/`readmmap` return that
# directly, so there is no permutedims copy — and wrapped in a `NamedDimsArray` so `sample` indexes them
# by axis name (`A[φ=…, θ=…, r=…]`), independent of the physical layout. The struct is parametric and the
# sampler is GPU-compilable: a hand-written `bsearchlast` (plain comparisons) replaces `searchsortedlast`,
# all literals are type-generic, and index values share one integer type (NamedDims' keyword getindex only
# lowers to a Metal kernel for a homogeneous index tuple).

using HDF5
using StaticArrays
using Unitful, UnitfulAstro
using AxisKeys: NamedDimsArray, dimnames
using DataManipulation: mapview
import StructArrays
import Adapt
using Zarr: zopen, ZGroup
using ZarrZfp   # ZfpCompressor: registers the "zfpy" Zarr codec the library uses

export KoralSnapshot, load_koral, load_grid, r_min, r_max

struct KoralSnapshot{Ar, Av, Ab, Rp, Tg, T}
    rho::Ar                      # electron-proxy mass density [code], NamedDimsArray (:φ,:θ,:r)
    velrel::Av                   # relative 4-velocity ũ^i (KS components), SVector{3} per cell, (:φ,:θ,:r)
    bfield::Ab                   # lab-frame 3-field B^i (KS components), SVector{3} per cell, (:φ,:θ,:r)
    # (velrel and bfield have independent types so the Zarr loader can leave either `nothing` when a
    #  view needs only one fluid field — see load_koral's `fields` selection)
    r_prof::Rp                   # r[i],   strictly increasing, length n1 (plain)
    th_grid::Tg                  # θ[i,j], increasing in j for each i, size (n1, n2) = [r,θ] (plain)
    φ0::T                        # φ of the first azimuthal cell
    dφ::T                        # uniform azimuthal spacing
    spin::T                      # black-hole spin a
    gam::T                       # adiabatic index (metadata)
    t::T                         # snapshot time (metadata)
    M_unit::T                    # code mass-density unit scale (construction only)
    MBH::T                       # black-hole mass [g] (construction only; Float32 narrows to Inf, unused on GPU)
end

# Extension-sniffing entry point. `path` ending in `.h5` ⇒ a raw KORAL dump (`load_koral_h5`, supports
# `mmap`); otherwise `path` is a store-snapshot path `…/RUN.zarr/snapNNNN` — open the store at
# `dirname(path)` and load the `basename(path)` snapshot. Kwargs forward to the matched branch.
function load_koral(path::AbstractString; kwargs...)
    p = rstrip(path, '/')                          # tolerate a trailing slash on a store-snapshot path
    endswith(p, ".h5") ?
        load_koral_h5(p; kwargs...) :
        load_koral(zopen(dirname(p), "r")::ZGroup, basename(p); kwargs...)
end

# `mmap=true` ⇒ the big fields are lazy byte-swap views over the (contiguous, uncompressed) datasets, so
# only touched cells are paged in; the mmap stays valid after the file is closed. `mmap=false` reads them
# into memory. Either way the fields keep their native (φ,θ,r) order and axis names.
function load_koral_h5(path::AbstractString; mmap=false)
    h5open(path, "r") do h5
        nd(A) = NamedDimsArray(A, (:φ, :θ, :r))
        if mmap
            # raw mmap bytes are in the file's endianness; ntoh (big-endian file) / ltoh (little-endian)
            # converts to host order on access — correct for any file/host combination.
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
        # ipole discards the KORAL dump's true startx3 (=−π) and resets the azimuthal origin to 0
        # (model/iharm/model.c:1180 `startx[3]=0.`, whose own comment admits ipole and KORAL disagree
        # about where x3=0), placing every fluid cell at (file φ)+π. We add π to align Synchray's
        # reconstruction to ipole's origin, so the koral camera images at n̂=+x like kerrdisk
        # (CAMERA_CONVENTION_VERIFIED §B; pure +π rotation, no reflection). This is a CONVENTION
        # ALIGNMENT to ipole's (admittedly-offset) origin, NOT a physics fix — do not "correct" it back.
        # sample() wraps φ periodically, so no renormalization is needed.
        φ0 = φ0_file + π

        rdscalar(name) = Float64(read(h5[name]))
        KoralSnapshot(rho, velrel, bfield, r_prof, th_grid, φ0, dφ,
            rdscalar("header/bhspin"), rdscalar("header/gam"), rdscalar("t"),
            rdscalar("header/units/M_unit"), ustrip(u"g", rdscalar("header/units/M_bh") * u"Msun"))
    end
end

# Derive the shared grid (identical across a run's snapshots) from a run store's `r`/`th`/`ph` arrays,
# exactly as `load_koral_h5` does from `grid_out`. Returns a NamedTuple the caller can cache and pass to
# `load_koral` so the (relatively expensive) grid decompression happens once per store, not per step.
function load_grid(store::ZGroup)
    rd(name) = store[name][:, :, :]                # whole (φ,θ,r) Float32 grid array, zfp-decompressed
    r = rd("r"); th = rd("th"); ph = rd("ph")
    n1 = size(r, 3); n2 = size(r, 2); n3 = size(r, 1)
    φ0_file = Float64(ph[1, 1, 1])
    (; r_prof  = Float64[r[1, 1, i] for i in 1:n1],
       th_grid = Float64[th[1, j, i] for i in 1:n1, j in 1:n2],
       φ0 = φ0_file + π,                            # ipole azimuthal-origin alignment, identical to load_koral_h5
       dφ = (Float64(ph[n3, 1, 1]) - φ0_file)/(n3 - 1))
end

# Loader for the zfp/Zarr snapshot library, parallel to `load_koral_h5`.
# The shared grid arrays and run header live in the enclosing store. Returns a `KoralSnapshot` like
# `load_koral_h5` — fields Float32, grid profiles and scalars Float64. No `mmap`: zfp arrays are
# decompressed into memory on read.
#
# `fields` selects which fluid fields to decompress; the rest are left `nothing` (the struct is
# parametric). Loading only what a view needs makes stepping fast — a density scrub reads one array, not
# seven. `sample()` touches all three fluid fields, so load with the default (all) before sampling
# (streamlines / ray tracer); sampling a partial snapshot errors loudly. This `(store, snapname)` method
# reuses an already-open store and a cached `grid` (see `load_grid`): the viewer's scrubber `zopen`s once
# and derives the grid once, then steps snapshots through this.
function load_koral(store::ZGroup, snapname::AbstractString;
                    fields = (:rho, :velrel, :bfield), grid = load_grid(store))
    fields ⊆ (:rho, :velrel, :bfield) || throw(ArgumentError("unknown fluid field(s) in $fields"))
    snap = store[snapname]::ZGroup                 # one snapshot group: fluid fields + per-snapshot t
    nd(A) = NamedDimsArray(A, (:φ, :θ, :r))
    rd(name) = snap[name][:, :, :]                 # whole (φ,θ,r) Float32 array, zfp-decompressed
    svf(a, b, c) = nd(StructArrays.StructArray{SVector{3,Float32}}((rd(a), rd(b), rd(c))))
    rho    = :rho    in fields ? nd(rd("rho")) : nothing
    velrel = :velrel in fields ? svf("U1", "U2", "U3") : nothing
    bfield = :bfield in fields ? svf("B1", "B2", "B3") : nothing

    sc(x) = Float64(x)                             # header scalars from JSON .zattrs → one scalar type (Float64)
    hdr = store.attrs; units = hdr["units"]
    KoralSnapshot(rho, velrel, bfield, grid.r_prof, grid.th_grid, grid.φ0, grid.dφ,
        sc(hdr["bhspin"]), sc(hdr["gam"]), sc(snap.attrs["t"]),
        sc(units["M_unit"]), ustrip(u"g", sc(units["M_bh"]) * u"Msun"))
end

r_min(s::KoralSnapshot) = @inbounds s.r_prof[1]
r_max(s::KoralSnapshot) = @inbounds s.r_prof[end]

# `searchsortedlast` over a 1-D array, written with plain comparisons so it compiles on the GPU.
@inline function bsearchlast(a, x)
    lo = 1; hi = length(a)
    @inbounds while lo < hi
        mid = (lo + hi + 1) >> 1
        a[mid] ≤ x ? (lo = mid) : (hi = mid - 1)
    end
    return lo
end

# Same, but over one row `ir` of a 2-D array (avoids a device `view`).
@inline function bsearchlast_row(a, ir, x, n)
    lo = 1; hi = n
    @inbounds while lo < hi
        mid = (lo + hi + 1) >> 1
        a[ir, mid] ≤ x ? (lo = mid) : (hi = mid - 1)
    end
    return lo
end

# Sample (rho, ũ^i, B^i) at lab-Cartesian (KS) position (x,y,z) in gravitational radii. Trilinear on the
# curvilinear grid: the θ bracket is resolved per radial face, the azimuthal index wraps periodically. The
# big fields are indexed by name (`A[φ=…, θ=…, r=…]`). Returns zeros outside the radial domain.
@inline function sample(s::KoralSnapshot, x, y, z)
    # sentinel type = the interpolation result type: Float32 fields × Float64 grid weights promote to
    # Float64 on the CPU zarr path, keeping this branch's type equal to the in-domain result (stable);
    # on the GPU everything is Float32 → Float32.
    T = promote_type(eltype(s.rho), eltype(s.r_prof)); z3 = zero(SVector{3,T})
    r = sqrt(x^2 + y^2 + z^2)
    (r < r_min(s) || r > r_max(s)) && return (zero(T), z3, z3)
    θ = acos(clamp(z/r, -one(r), one(r)))
    φ = atan(y, x)
    n2 = size(s.th_grid, 2); n3 = size(s.rho, :φ)

    i = clamp(bsearchlast(s.r_prof, r), 1, length(s.r_prof) - 1)
    ti = (r - s.r_prof[i])/(s.r_prof[i+1] - s.r_prof[i])
    # one integer type throughout (NamedDims keyword getindex only lowers to Metal for a homogeneous tuple);
    # `unsafe_trunc(Int, floor(·))` instead of `floor(Int, ·)` — the latter boxes on the GPU.
    fk = (φ - s.φ0)/s.dφ
    k0 = unsafe_trunc(Int, floor(fk)); tk = fk - k0
    k = mod(k0, n3) + 1; kp = mod(k0 + 1, n3) + 1

    # θ bracket of each radial face, resolved once and reused for every field.
    θbracket(ir) = @inbounds begin
        j = clamp(bsearchlast_row(s.th_grid, ir, θ, n2), 1, n2 - 1)
        a = s.th_grid[ir, j]
        (j, clamp((θ - a)/(s.th_grid[ir, j+1] - a), zero(θ), one(θ)))
    end
    ji, tji = θbracket(i); jp, tjp = θbracket(i + 1)

    face(A, ir, jj, tj) = @inbounds (A[φ=k, θ=jj, r=ir]*(1-tj) + A[φ=k, θ=jj+1, r=ir]*tj)*(1-tk) +
                                    (A[φ=kp, θ=jj, r=ir]*(1-tj) + A[φ=kp, θ=jj+1, r=ir]*tj)*tk
    trilerp(A) = face(A, i, ji, tji)*(1 - ti) + face(A, i + 1, jp, tjp)*ti
    return (trilerp(s.rho), trilerp(s.velrel), trilerp(s.bfield))
end

# Preserve axis names when moving a NamedDimsArray to another array type (default `adapt` drops them).
Adapt.adapt_structure(to, x::NamedDimsArray) = NamedDimsArray(Adapt.adapt(to, parent(x)), dimnames(x))

# Move the grid arrays to another array type (e.g. an `MtlArray`); scalars pass through unchanged.
Adapt.adapt_structure(to, s::KoralSnapshot) = KoralSnapshot(
    Adapt.adapt(to, s.rho), Adapt.adapt(to, s.velrel), Adapt.adapt(to, s.bfield),
    Adapt.adapt(to, s.r_prof), Adapt.adapt(to, s.th_grid),
    s.φ0, s.dφ, s.spin, s.gam, s.t, s.M_unit, s.MBH)

end # module
