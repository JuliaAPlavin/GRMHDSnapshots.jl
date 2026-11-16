# Frame-invariant plasma-physics quantities in Kerr-Schild coordinates (G=c=M=1). The dump stores, per cell, the
# HARM/Gammie relative 4-velocity ũ^i and the lab-frame 3-field B^i as KS components of the grid's spherical coordinates;
# the grid maps work in the spin frame's, where the metric is Kerr-Schild. From these we reconstruct the plasma 4-velocity
# u^μ and comoving field b^μ (matching ipole's load_koral_data / get_model_fourv). Everything here is emission-agnostic
# and coordinate-frame-invariant.

# Kerr-Schild metric g_{μν}, coordinates (t,r,θ,φ). Type-generic (zero(r)/one(r)) so the SMatrix stays in
# the working precision — no Float64 literals leak in.
@inline ks_gcov(r, θ, a) = _ks_gcov_sq(r, cos(θ)^2, sin(θ)^2, a)

# The same from c2 = cos²θ and s2 = sin²θ: a turned cell has them from its spin-frame unit vector, without θ.
@inline function _ks_gcov_sq(r, c2, s2, a)
    ρ2 = r^2 + a^2*c2; tr = 2r/ρ2; z = zero(r); o = one(r)
    @SMatrix [ -o+tr      tr           z   -a*s2*tr ;
                tr        o+tr         z   -a*s2*(o+tr) ;
                z         z            ρ2  z ;
               -a*s2*tr  -a*s2*(o+tr)  z   s2*(ρ2 + a^2*s2*(o+tr)) ]
end

# Spatial-metric norm √(γᵢⱼVⁱVʲ) of a contravariant 3-vector in KS (γ = spatial part of ks_gcov): the
# proper coordinate-invariant magnitude of V^i — a raw Euclidean component norm instead mixes the
# radial (per-length) and angular (per-radian) parts and is not a real magnitude.
@inline spatial_norm(V::SVector{3}, r, θ, a) = _spatial_norm(V, ks_gcov(r, θ, a))
@inline function _spatial_norm(V::SVector{3}, g::SMatrix{4,4})
    γ = g[SVector(2, 3, 4), SVector(2, 3, 4)]
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

# Inverse of `gammie_ucon`: KS 4-velocity u^μ → relative velocity ũ^i = u^i − u^t g^{ti}/g^{tt}, with the same
# closed-form g^{tμ} (only ũ^r differs from u^i). Linear in u^μ.
@inline function gammie_uprim(u::SVector{4}, g::SMatrix{4,4})
    tr = g[1, 2]
    SVector(u[2] + u[1]*tr/(1 + tr), u[3], u[4])
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
@inline fluid_velocity(ũ::SVector{3}, r, θ, a) = _fluid_velocity(ũ, ks_gcov(r, θ, a))
@inline function _fluid_velocity(ũ::SVector{3}, g::SMatrix{4,4})
    u = gammie_ucon(ũ, g)
    SVector(u[2], u[3], u[4]) / u[1]
end

"""
    flow_speed(ũ, r, θ, a)
    flow_speed(snap)

Physical 3-speed of the fluid relative to the local normal (Eulerian) observer, as a fraction of c:
`v = βγ/√(1 + βγ²)`, `βγ` the [`proper_velocity`](@ref). Bounded [0,1) and metric-correct — unlike a
raw coordinate-component norm. No subtraction of nearly equal numbers, unlike `√(1 − 1/γ²)` for slow
flows. The grid method maps it over a snapshot, `(:φ,:θ,:r)`.
"""
@inline flow_speed(ũ::SVector{3}, r, θ, a) = _flow_speed(ũ, ks_gcov(r, θ, a))
@inline _flow_speed(ũ::SVector{3}, g::SMatrix{4,4}) = (βγ = _spatial_norm(ũ, g); βγ/√(1 + βγ^2))

"""
    proper_velocity(ũ, r, θ, a)
    proper_velocity(snap)

Proper velocity (celerity) `βγ` of the fluid relative to the local normal (Eulerian) observer —
the magnitude of the spatial part of the 4-velocity in that observer's frame. Same frame as
[`flow_speed`](@ref) but unbounded [0,∞): distinguishes highly relativistic flows that `v/c`
compresses near 1. Computed as `√(γᵢⱼũⁱũʲ)`: `ũ` is exactly that spatial part, so `γ² − 1 = γᵢⱼũⁱũʲ`
with no subtraction of nearly equal numbers. The grid method maps it over a snapshot, `(:φ,:θ,:r)`.
"""
@inline proper_velocity(ũ::SVector{3}, r, θ, a) = spatial_norm(ũ, r, θ, a)

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

Normal-(Eulerian-)observer lapse α = 1/√(−gᵗᵗ) = 1/√(1 + 2r/(r²+a²cos²θ)) at KS `(r, θ)` with spin
`a`: the normal observer's proper-time-per-coordinate-time. NOT the photon redshift — that is
[`grav_redshift`](@ref) (they differ, even in Schwarzschild). Type-generic; α → 1 as r → ∞, α ∈ (0,1].
"""
@inline lapse(r, θ, a) = _lapse(r, cos(θ)^2, a)
@inline _lapse(r, c2, a) = 1/√(1 + 2r/(r^2 + a^2*c2))      # c2 = cos²θ

"""
    grav_redshift(r, θ, a)

Gravitational redshift factor g = ν_obs/ν_emit for a static emitter reaching infinity: g = √(−g_tt)
= √(1 − 2r/(r²+a²cos²θ)) at KS `(r, θ)` with spin `a`. This is *the* gravitational redshift (the
`k`-direction cancels; Doppler/frame-dragging excluded), coordinate-invariant, → 0 at the horizon.
Clamped to 0 inside the ergosphere (−g_tt < 0), where no static emitter exists. g → 1 as r → ∞.
"""
@inline _grav_redshift(g::SMatrix{4,4}) = (v = -g[1, 1]; √(max(zero(v), v)))   # √(−g_tt), 0 in ergosphere
@inline grav_redshift(r, θ, a) = _grav_redshift(ks_gcov(r, θ, a))

"""
    volume_ratio(r, θ, a)

Ratio of proper 3-volume to Euclidean coordinate volume at KS `(r, θ)` with spin `a`:
`√γ/(r²sinθ) = (r²+a²cos²θ)/(r²·α)` (γ the spatial-metric determinant, α the [`lapse`](@ref)).
Multiplying a per-proper-volume density by this gives a per-displayed-(Cartesian-)volume density, so
its integral over the drawn `dx dy dz` box is metric-correct. → 1 as r → ∞; > 1 near the BH.
"""
@inline volume_ratio(r, θ, a) = _volume_ratio(r, cos(θ)^2, a)
@inline _volume_ratio(r, c2, a) = (ρ2 = r^2 + a^2*c2; √(ρ2*(ρ2 + 2r))/r^2)   # c2 = cos²θ; = ρ2/(r²α) with one sqrt

"""
    ks_to_bl(v, r, θ, a)

Boyer-Lindquist components of a contravariant 4-vector `v^μ` given in Kerr-Schild components at
`(r, θ)` with spin `a`. The charts share `r` and `θ`; `t` and `φ` differ by functions of `r` alone
(`dt_BL = dt_KS − (2r/Δ)dr`, `dφ_BL = dφ_KS − (a/Δ)dr`, `Δ = r² − 2r + a²`), so only `vᵗ` and `vᵠ`
change, by multiples of `vʳ`. Singular at the horizons (Δ = 0), where BL is.
"""
@inline function ks_to_bl(v::SVector{4}, r, θ, a)
    Δ = r^2 - 2r + a^2
    SVector(v[1] - 2r/Δ*v[2], v[2], v[3], v[4] - a/Δ*v[2])
end

# ── Physical unit scalars (Unitful internally, stripped to plain CGS Float64 at the boundary — matching
# the downstream synchrotron code, which is plain CGS). ──

# Scale-free dumps (e.g. iharm3d/KHARMA, H-AMR) may lack M_unit and/or MBH; fail loud naming the missing ones
# rather than compute with `nothing`.
function _require_units(snap::GRMHDSnapshot, f::Symbol, names::Symbol...)
    miss = filter(n -> getproperty(snap, n) === nothing, names)
    units = (M_unit = "M☉", MBH = "M☉")
    isempty(miss) || throw(ArgumentError("$f needs $(join(map(n -> "$n ($(units[n]))", miss), " and ")), missing from this snapshot; pass to the loader"))
end

# Gravitational radius G·MBH/c² in cm of a black-hole mass MBH in M☉.
_rg_cm(MBH) = ustrip(u"cm", Unitful.G*(MBH*u"Msun")/Unitful.c0^2)

"""
    lunit(snap)

Gravitational radius r_g = G·MBH/c² in cm. Needs `MBH` only.
"""
lunit(snap::GRMHDSnapshot) = (_require_units(snap, :lunit, :MBH); _rg_cm(snap.MBH))

"""
    rhounit(snap)

Code mass-density unit M_unit/r_g³ in g/cm³ (`M_unit` in M☉). Needs both `M_unit` and `MBH`.
"""
rhounit(snap::GRMHDSnapshot) = (_require_units(snap, :rhounit, :M_unit, :MBH); ustrip(u"g/cm^3", (Float64(snap.M_unit)*u"Msun")/(lunit(snap)*u"cm")^3))

"""
    bunit(snap)

Code field unit c·√(4π·ρ_unit) in Gauss. Needs both `M_unit` and `MBH`.
"""
bunit(snap::GRMHDSnapshot) = (_require_units(snap, :bunit, :M_unit, :MBH); ustrip(u"cm/s", Unitful.c0)*sqrt(4π*rhounit(snap)))

# ── Full-grid derived arrays, (:φ,:θ,:r) matching snap.fluid. For cell (φ=k,θ=j,r=i): grid coords r_prof[i], th_grid[i,j],
# φ0 + (k−1)·dφ; fluid record fluid[φ=k,θ=j,r=i]. Results are in the spin frame: vectors in its components, the metric
# at the cell's spin-frame colatitude θ′. ──

# f(cell, fr) at every cell, as a `(:φ,:θ,:r)` array: the cell's fluid record (grid components) and its spin-frame geometry
# fr = (; r, n, eθ, eφs) for `_tospin` and `_ks_gcov`. `f` converts only the fields it reads, so unread ones cost nothing
# and a partially loaded snapshot works. `foreach_r` runs over the radial slabs (`foreach` or `tforeach`).
function _map_cells(f, snap::GRMHDSnapshot, foreach_r)
    (; fluid, r_prof, th_grid, φ0, dφ) = snap
    A = parent(fluid)
    R = _rotation(snap.R); z = R[:, 3]
    # per φ column (center as in `_sample_stencil`): the spin-frame directions of the grid's x̂ cosφ + ŷ sinφ and φ̂
    cols = map(axes(A, 1)) do k
        sφ, cφ = sincos(φ0 + (k - 1)*dφ)
        (u = R*SVector(cφ, sφ, zero(cφ)), v = R*SVector(-sφ, cφ, zero(cφ)))
    end
    # inlined into the loop: a call per cell costs more than the cell's own work
    @inline cell(k, j, i, sθ, cθ) = @inbounds let (; u, v) = cols[k]
        @inline f(A[k, j, i], (r = r_prof[i], n = sθ*u + cθ*z, eθ = cθ*u - sθ*z, eφs = sθ*v))
    end
    out = Array{typeof(cell(1, 1, 1, sincos(th_grid[1, 1])...))}(undef, size(A))
    foreach_r(i -> _fill_slab!(out, cell, i, view(th_grid, i, :)), axes(A, 3))
    NamedDimsArray(out, (:φ, :θ, :r))
end

# Radial slab `i` of `out`: sincos of each grid θ once, then every cell.
function _fill_slab!(out, cell, i, ths)
    @inbounds for j in axes(out, 2)
        sθ, cθ = sincos(ths[j])
        for k in axes(out, 1)
            out[k, j, i] = cell(k, j, i, sθ, cθ)
        end
    end
end

# Kerr-Schild metric at the spin-frame point of geometry `fr`.
@inline _ks_gcov(fr, a) = _ks_gcov_sq(fr.r, fr.n[3]^2, fr.n[1]^2 + fr.n[2]^2, a)

"""
    map_plasma(f, snap; threaded=true)

Map a kernel `f(ps, g, cell)` over the whole grid, returning a `(:φ,:θ,:r)` array. `ps` is the
[`plasma_state`](@ref) named tuple `(u, b, bsq)` and `g` the per-cell [`grav_redshift`](@ref) √(−g_tt)
— both reconstructed from `cell.ũ`/`cell.B` and the metric — and `cell` is the cell's fluid record with its vectors in
spin-frame components, so `ρ` need not be loaded unless `f` reads `cell.ρ`. The outer radial loop runs over
`OhMyThreads.tforeach` (`threaded`) or `foreach`.
"""
map_plasma(f, snap::GRMHDSnapshot; threaded::Bool = true) =
    _map_cells(snap, threaded ? tforeach : foreach) do c, fr
        ũ, B = _tospin(c.ũ, fr), _tospin(c.B, fr)
        g = _ks_gcov(fr, snap.spin)
        f(_plasma_state(ũ, B, g), _grav_redshift(g), merge(c, (; ũ, B)))
    end

"""
    comoving_bsq(snap)

Per-cell invariant b·b [code units], as a `(:φ,:θ,:r)` array.
"""
comoving_bsq(snap::GRMHDSnapshot) = map_plasma((ps, _, _) -> ps.bsq, snap)

"""
    fluid_velocity(snap)

Per-cell coordinate 3-velocity `v^i` [code units], as a `(:φ,:θ,:r)` array.
"""
fluid_velocity(snap::GRMHDSnapshot) = _map_cells((c, fr) -> _fluid_velocity(_tospin(c.ũ, fr), _ks_gcov(fr, snap.spin)), snap, foreach)

"""
    flow_speed(snap)

Per-cell physical speed `v/c` relative to the normal observer, as a `(:φ,:θ,:r)` array.
"""
flow_speed(snap::GRMHDSnapshot) = _map_cells((c, fr) -> _flow_speed(_tospin(c.ũ, fr), _ks_gcov(fr, snap.spin)), snap, foreach)

"""
    proper_velocity(snap)

Per-cell proper velocity `βγ = √(γ²−1)` relative to the normal observer, as a `(:φ,:θ,:r)` array.
"""
proper_velocity(snap::GRMHDSnapshot) = _map_cells((c, fr) -> _spatial_norm(_tospin(c.ũ, fr), _ks_gcov(fr, snap.spin)), snap, foreach)

"""
    bfield_magnitude(snap)

Normal-frame magnitude of the lab 3-field `B^i`, `√(γᵢⱼBⁱBʲ)` [code units], as a `(:φ,:θ,:r)` array —
the proper spatial-metric norm, not a raw component norm.
"""
bfield_magnitude(snap::GRMHDSnapshot) = _map_cells((c, fr) -> _spatial_norm(_tospin(c.B, fr), _ks_gcov(fr, snap.spin)), snap, foreach)

"""
    comoving_B_gauss(snap)

Per-cell comoving field magnitude √bsq·bunit [Gauss], as a `(:φ,:θ,:r)` array.
"""
comoving_B_gauss(snap::GRMHDSnapshot) = sqrt.(comoving_bsq(snap)) .* bunit(snap)

"""
    magnetization(snap)

Per-cell magnetization σ = b²/ρ [code units], as a `(:φ,:θ,:r)` array. ρ=0 cells are the natural `Inf`.
"""
magnetization(snap::GRMHDSnapshot) = map_plasma((ps, _, c) -> ps.bsq/c.ρ, snap)

"""
    lapse(snap)

Per-cell lapse α, as a `(:φ,:θ,:r)` array.
"""
lapse(snap::GRMHDSnapshot) = _map_cells((_, fr) -> _lapse(fr.r, fr.n[3]^2, snap.spin), snap, foreach)

"""
    grav_redshift(snap)

Per-cell gravitational redshift g = √(−g_tt), as a `(:φ,:θ,:r)` array.
"""
grav_redshift(snap::GRMHDSnapshot) = _map_cells((_, fr) -> _grav_redshift(_ks_gcov(fr, snap.spin)), snap, foreach)

"""
    volume_ratio(snap)

Per-cell proper-to-coordinate volume ratio √γ/(r²sinθ), as a `(:φ,:θ,:r)` array.
"""
volume_ratio(snap::GRMHDSnapshot) = _map_cells((_, fr) -> _volume_ratio(fr.r, fr.n[3]^2, snap.spin), snap, foreach)
