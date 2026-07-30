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

# Gammie relative-velocity → contravariant 4-velocity u^μ (KS). Uses the closed-form KS inverse-metric
# row g^{tμ} = (-(1+2r/ρ²), 2r/ρ², 0, 0): only u^t and u^r differ from the relative velocity.
@inline function gammie_ucon(ũ::SVector{3}, g::SMatrix{4,4}, r, θ, a)
    tr = 2r/(r^2 + a^2*cos(θ)^2)
    q = zero(eltype(ũ))
    @inbounds for i in 1:3, j in 1:3
        q += g[i+1, j+1]*ũ[i]*ũ[j]
    end
    ufac = sqrt((1 + q)/(1 + tr))
    SVector(ufac*(1 + tr), ũ[1] - ufac*tr, ũ[2], ũ[3])
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
@inline function plasma_state(ũ::SVector{3}, B::SVector{3}, r, θ, a)
    g = ks_gcov(r, θ, a)
    u = gammie_ucon(ũ, g, r, θ, a)
    ucov = g*u
    b = harm_bcon(B, u, ucov)
    (; u, b, bsq = b'*g*b)
end

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

# Map a per-cell kernel of the primitives+coords over the whole grid.
function _grid_map(f, snap::KoralSnapshot)
    (; velrel, bfield, r_prof, th_grid, spin) = snap
    nφ, nθ, nr = size(velrel)
    NamedDimsArray(
        [f(velrel[φ=k, θ=j, r=i], bfield[φ=k, θ=j, r=i], r_prof[i], th_grid[i, j], spin)
         for k in 1:nφ, j in 1:nθ, i in 1:nr],
        (:φ, :θ, :r))
end

"""
    comoving_bsq(snap)

Per-cell invariant b·b [code units], as a `(:φ,:θ,:r)` array.
"""
comoving_bsq(snap::KoralSnapshot) = _grid_map((ũ, B, r, θ, a) -> plasma_state(ũ, B, r, θ, a).bsq, snap)

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
    nφ, nθ, nr = size(snap.rho)
    NamedDimsArray(
        [lapse(r_prof[i], th_grid[i, j], spin) for k in 1:nφ, j in 1:nθ, i in 1:nr],
        (:φ, :θ, :r))
end
