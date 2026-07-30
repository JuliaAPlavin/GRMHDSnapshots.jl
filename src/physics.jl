# Frame-invariant plasma-physics quantities in Kerr-Schild coordinates (G=c=M=1). The dump stores, per
# cell, the HARM/Gammie relative 4-velocity ũ^i and the lab-frame 3-field B^i as KS components; from
# these we reconstruct the plasma 4-velocity u^μ and comoving field b^μ (matching ipole's
# load_koral_data / get_model_fourv). Everything here is emission-agnostic and coordinate-frame-invariant.

# Kerr-Schild metric g_{μν}, coordinates (t,r,θ,φ). Type-generic (zero(r)/one(r)) so the SMatrix stays in
# the working precision — no Float64 literals leak in.
@inline function ks_gcov(r, θ, a)
    s2 = sin(θ)^2; ρ2 = r^2 + a^2*cos(θ)^2; tr = 2r/ρ2; z = zero(r); o = one(r)
    @SMatrix [ -o+tr      tr           z   -a*s2*tr ;
                tr        o+tr         z   -a*s2*(o+tr) ;
                z         z            ρ2  z ;
               -a*s2*tr  -a*s2*(o+tr)  z   s2*(ρ2 + a^2*s2*(o+tr)) ]
end

# Spatial-metric norm √(γᵢⱼVⁱVʲ) of a contravariant 3-vector in KS (γ = spatial part of ks_gcov): the
# proper coordinate-invariant magnitude of V^i — a raw Euclidean component norm instead mixes the
# radial (per-length) and angular (per-radian) parts and is not a real magnitude.
@inline function spatial_norm(V::SVector{3}, r, θ, a)
    γ = ks_gcov(r, θ, a)[SVector(2, 3, 4), SVector(2, 3, 4)]
    sqrt(V'*γ*V)
end

# Gammie relative-velocity → contravariant 4-velocity u^μ (KS). Uses the closed-form KS inverse-metric
# row g^{tμ} = (-(1+2r/ρ²), 2r/ρ², 0, 0): only u^t and u^r differ from the relative velocity. `tr = 2r/ρ²`
# is the metric's (t,r) component `g[1,2]`, reused rather than recomputed. `promote_type` keeps `q` in the
# metric's precision (a Float32 primitive against a Float64 metric would otherwise box).
@inline function gammie_ucon(ũ::SVector{3}, g::SMatrix{4,4})
    tr = g[1, 2]
    q = zero(promote_type(eltype(ũ), eltype(g)))
    @inbounds for i in 1:3, j in 1:3
        q += g[i+1, j+1]*ũ[i]*ũ[j]
    end
    ufac = sqrt((1 + q)/(1 + tr))
    SVector(ufac*(1 + tr), ũ[1] - ufac*tr, ũ[2], ũ[3])
end

"""
    fluid_ucon(ũ, r, θ, a)

Contravariant Kerr-Schild 4-velocity `u^μ` of the fluid, reconstructed from the relative-velocity
primitive `ũ^i` at KS `(r, θ)` with spin `a`. Needs no magnetic field (unlike `plasma_state`).

The spatial part `u^{r,θ,φ}` is the fluid's flow direction; `ũ` itself is motion *relative to the
infalling KS normal observer* and points elsewhere near the horizon (its radial component stays
outward where the fluid is accreting inward).
"""
@inline fluid_ucon(ũ::SVector{3}, r, θ, a) = gammie_ucon(ũ, ks_gcov(r, θ, a))

"""
    fluid_velocity(ũ, r, θ, a)
    fluid_velocity(snap)

Coordinate 3-velocity `v^i = u^i/u^t = dx^i/dt` (KS components) of the fluid: same direction as the
flow, magnitude the coordinate speed (bounded, unlike the 4-velocity's proper-time components). The
grid method maps it over a snapshot, `(:φ,:θ,:r)`.
"""
@inline function fluid_velocity(ũ::SVector{3}, r, θ, a)
    u = fluid_ucon(ũ, r, θ, a)
    SVector(u[2], u[3], u[4]) / u[1]
end

"""
    flow_speed(ũ, r, θ, a)
    flow_speed(snap)

Physical 3-speed of the fluid relative to the local normal (Eulerian) observer, as a fraction of c:
`v = √(1 − 1/γ²)`, `γ = α·uᵗ` the Lorentz factor (`α` the lapse). Bounded [0,1) and metric-correct —
unlike a raw coordinate-component norm. The grid method maps it over a snapshot, `(:φ,:θ,:r)`.
"""
@inline function flow_speed(ũ::SVector{3}, r, θ, a)
    u = fluid_ucon(ũ, r, θ, a)
    γ = lapse(r, θ, a)*u[1]
    sqrt(max(zero(γ), 1 - 1/γ^2))
end

"""
    proper_velocity(ũ, r, θ, a)
    proper_velocity(snap)

Proper velocity (celerity) `βγ = √(γ²−1)` of the fluid relative to the local normal (Eulerian)
observer, `γ = α·uᵗ` the Lorentz factor — the magnitude of the spatial part of the 4-velocity in
that observer's frame. Same frame as [`flow_speed`](@ref) but unbounded [0,∞): distinguishes highly
relativistic flows that `v/c` compresses near 1. Computed as `√(γ²−1)` directly (cleaner than `βγ`
for large γ). The grid method maps it over a snapshot, `(:φ,:θ,:r)`.
"""
@inline function proper_velocity(ũ::SVector{3}, r, θ, a)
    γ = lapse(r, θ, a)*fluid_ucon(ũ, r, θ, a)[1]
    sqrt(max(zero(γ), γ^2 - 1))
end

# Lab 3-field B^i + 4-velocity → comoving field 4-vector b^μ (HARM convention b^t = B^i u_i).
@inline function harm_bcon(B::SVector{3}, u::SVector{4}, ucov::SVector{4})
    b0 = ucov[2]*B[1] + ucov[3]*B[2] + ucov[4]*B[3]
    SVector(b0, (B[1] + b0*u[2])/u[1], (B[2] + b0*u[3])/u[1], (B[3] + b0*u[4])/u[1])
end

"""
    plasma_state(ũ, B, r, θ, a)

Reconstruct the frame-invariant plasma state at a KS point from interpolated primitives (relative
4-velocity `ũ^i` and lab 3-field `B^i`, both in KS components) with spin `a`.

# Returns
Named tuple `(u, b, bsq)`:
- `u::SVector{4}`   — contravariant KS 4-velocity u^μ (normalized, u·u = -1).
- `b::SVector{4}`   — comoving field 4-vector b^μ (orthogonal to u, b·u = 0).
- `bsq`             — invariant b·b in code units (B²[Gauss] = bsq·bunit²).
"""
# Core reconstruction given the precomputed metric `g` — lets callers that already hold `g` (e.g. the
# per-(r,θ) slab in `map_plasma`) skip rebuilding it.
@inline function _plasma_state(ũ::SVector{3}, B::SVector{3}, g::SMatrix{4,4})
    u = gammie_ucon(ũ, g)
    ucov = g*u
    b = harm_bcon(B, u, ucov)
    (; u, b, bsq = b'*g*b)
end
@inline plasma_state(ũ::SVector{3}, B::SVector{3}, r, θ, a) = _plasma_state(ũ, B, ks_gcov(r, θ, a))

"""
    lapse(r, θ, a)

Gravitational-redshift factor (lapse) α = 1/√(1 + 2r/(r²+a²cos²θ)) at KS `(r, θ)` with spin `a`.
Type-generic; α → 1 as r → ∞ and α ∈ (0,1] at finite r.
"""
@inline lapse(r, θ, a) = 1/√(1 + 2r/(r^2 + a^2*cos(θ)^2))

# ── Physical unit scalars (Unitful internally, stripped to plain CGS Float64 at the boundary — matching
# the downstream synchrotron code, which is plain CGS). ──

"""
    lunit(snap)

Gravitational radius r_g = G·MBH/c² in cm.
"""
lunit(snap::KoralSnapshot) = ustrip(u"cm", Unitful.G*(snap.MBH*u"g")/Unitful.c0^2)

"""
    rhounit(snap)

Code mass-density unit M_unit/r_g³ in g/cm³.
"""
rhounit(snap::KoralSnapshot) = ustrip(u"g/cm^3", (snap.M_unit*u"g")/(lunit(snap)*u"cm")^3)

"""
    bunit(snap)

Code field unit c·√(4π·ρ_unit) in Gauss.
"""
bunit(snap::KoralSnapshot) = ustrip(u"cm/s", Unitful.c0)*sqrt(4π*rhounit(snap))

# ── Full-grid derived arrays, (:φ,:θ,:r) matching snap.rho. For cell (φ=k,θ=j,r=i): coords r_prof[i],
# th_grid[i,j]; primitives velrel/bfield[φ=k,θ=j,r=i]. Metric depends only on (r,θ). ──

"""
    map_plasma(f, snap, extras...; threaded=true)

Map a kernel `f(ps, α, extras_cell...)` over the whole grid, returning a `(:φ,:θ,:r)` array. `ps` is
the [`plasma_state`](@ref) named tuple `(u, b, bsq)` and `α` the lapse — both reconstructed from
`velrel`/`bfield` and the metric, so `rho` need not be loaded. `extras` are full `(:φ,:θ,:r)`
`NamedDimsArray`s (e.g. `snap.rho`) whose matching cell is passed after `α`. The Kerr-Schild metric
(and hence `ps` and `α`) depends only on `(r,θ)`, so it is built once per `(r,θ)` slab and reused
across all φ. The outer radial loop runs over `OhMyThreads.tmap` (`threaded`) or `map`.
"""
function map_plasma(f, snap::KoralSnapshot, extras::Vararg{Any,N}; threaded::Bool = true) where {N}
    rr = 1:size(snap.velrel, :r)
    slab(i) = _plasma_slab(f, snap, extras, i)      # function barrier → concrete slab element type
    slabs = threaded ? tmap(slab, rr) : map(slab, rr)
    NamedDimsArray(stack(slabs), (:φ, :θ, :r))
end

# One (φ,θ) slab at radial index `i`. The (r,θ)-only metric g and lapse α are built once per θ and reused
# across φ. NamedDims keyword indexing is zero-cost. `f` is a proper argument → this specializes on it.
function _plasma_slab(f, snap::KoralSnapshot, extras, i)
    (; velrel, bfield, r_prof, th_grid, spin) = snap
    nφ, nθ = size(velrel, :φ), size(velrel, :θ)
    r = r_prof[i]
    cell(k, j, g, α) = f(_plasma_state(velrel[φ=k, θ=j, r=i], bfield[φ=k, θ=j, r=i], g), α,
                         map(e -> e[φ=k, θ=j, r=i], extras)...)
    g1 = ks_gcov(r, th_grid[i, 1], spin)
    out = Matrix{typeof(cell(1, 1, g1, 1/√(1 + g1[1, 2])))}(undef, nφ, nθ)
    @inbounds for j in 1:nθ
        g = ks_gcov(r, th_grid[i, j], spin); α = 1/√(1 + g[1, 2])
        for k in 1:nφ
            out[k, j] = cell(k, j, g, α)
        end
    end
    out
end

"""
    comoving_bsq(snap)

Per-cell invariant b·b [code units], as a `(:φ,:θ,:r)` array.
"""
comoving_bsq(snap::KoralSnapshot) = map_plasma((ps, α) -> ps.bsq, snap)

# Map a per-cell kernel of ONE primitive field + coords over the grid. Unlike `map_plasma` it touches
# only `field` (velrel or bfield), so it works on a partially-loaded snapshot.
function _grid_map1(f, field, snap::KoralSnapshot)
    (; r_prof, th_grid, spin) = snap
    nφ, nθ, nr = size(field)
    NamedDimsArray(
        [f(field[φ=k, θ=j, r=i], r_prof[i], th_grid[i, j], spin) for k in 1:nφ, j in 1:nθ, i in 1:nr],
        (:φ, :θ, :r))
end

"""
    fluid_velocity(snap)

Per-cell coordinate 3-velocity `v^i` [code units], as a `(:φ,:θ,:r)` array.
"""
fluid_velocity(snap::KoralSnapshot) = _grid_map1(fluid_velocity, snap.velrel, snap)

"""
    flow_speed(snap)

Per-cell physical speed `v/c` relative to the normal observer, as a `(:φ,:θ,:r)` array.
"""
flow_speed(snap::KoralSnapshot) = _grid_map1(flow_speed, snap.velrel, snap)

"""
    proper_velocity(snap)

Per-cell proper velocity `βγ = √(γ²−1)` relative to the normal observer, as a `(:φ,:θ,:r)` array.
"""
proper_velocity(snap::KoralSnapshot) = _grid_map1(proper_velocity, snap.velrel, snap)

"""
    bfield_magnitude(snap)

Normal-frame magnitude of the lab 3-field `B^i`, `√(γᵢⱼBⁱBʲ)` [code units], as a `(:φ,:θ,:r)` array —
the proper spatial-metric norm, not a raw component norm.
"""
bfield_magnitude(snap::KoralSnapshot) = _grid_map1(spatial_norm, snap.bfield, snap)

"""
    comoving_B_gauss(snap)

Per-cell comoving field magnitude √bsq·bunit [Gauss], as a `(:φ,:θ,:r)` array.
"""
comoving_B_gauss(snap::KoralSnapshot) = sqrt.(comoving_bsq(snap)) .* bunit(snap)

"""
    magnetization(snap)

Per-cell magnetization σ = b²/ρ [code units], as a `(:φ,:θ,:r)` array. ρ=0 cells are the natural `Inf`.
"""
magnetization(snap::KoralSnapshot) = comoving_bsq(snap) ./ snap.rho

"""
    lapse(snap)

Per-cell lapse α, as a `(:φ,:θ,:r)` array (φ-invariant, replicated for downstream texture consistency).
"""
function lapse(snap::KoralSnapshot)
    (; r_prof, th_grid, spin) = snap
    nφ = size(snap.rho, :φ)
    α = lapse.(r_prof, th_grid, spin)                     # (r, θ) — φ-invariant, computed once
    NamedDimsArray([α[i, j] for k in 1:nφ, j in axes(α, 2), i in axes(α, 1)], (:φ, :θ, :r))
end
