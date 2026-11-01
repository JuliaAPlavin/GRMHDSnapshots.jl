# Funky Modified Kerr-Schild (FMKS) native coordinates X = (X1, X2, X3) → Kerr-Schild (r, θ, φ), as
# used by iharm3d/KHARMA dumps. Follows ipole's coordinates.c (`bl_coord` and `set_dxdX`, FMKS branch):
# r = exp(X1), φ = X3 (uniform), θ = θ(X1, X2). The (r, θ) map depends only on (X1, X2) — the grid is
# axisymmetric — so a snapshot's `r_prof`/`th_grid` and the per-cell vector Jacobian are all φ-independent.

"""
    FMKS(; startx1, hslope, mks_smooth, poly_xt, poly_alpha)

FMKS coordinate parameters. `poly_norm` is derived from `(poly_xt, poly_alpha)`, matching ipole.
"""
struct FMKS{T}
    startx1::T
    hslope::T
    mks_smooth::T
    poly_xt::T
    poly_alpha::T
    poly_norm::T
end

FMKS(; startx1, hslope, mks_smooth, poly_xt, poly_alpha) =
    FMKS(promote(startx1, hslope, mks_smooth, poly_xt, poly_alpha,
                 0.5π/(1 + 1/((poly_alpha + 1)*poly_xt^poly_alpha)))...)

# KS radius and colatitude at native (X1, X2). θ = θG (an MKS-style de-refinement) blended by
# exp(mks_smooth·(startx1 − X1)) into θJ (a polynomial "funky" de-refinement of the poles).
@inline function bl_coord(c::FMKS, X1, X2)
    r = exp(X1)
    thG = π*X2 + ((1 - c.hslope)/2)*sin(2π*X2)
    y = 2X2 - 1
    thJ = c.poly_norm*y*(1 + (y/c.poly_xt)^c.poly_alpha/(c.poly_alpha + 1)) + π/2
    th = thG + exp(c.mks_smooth*(c.startx1 - X1))*(thJ - thG)
    (r, th)
end

# Spatial Jacobian J₃ = ∂(r, θ, φ)_KS / ∂(X1, X2, X3): contravariant 3-vectors (the relative velocity
# ũ^i and lab field B^i) transform as v_KS = J₃·v_nat. Since φ = X3 the last row/column is trivial; the
# (θ, X1) cross-term is the r-dependence of the FMKS pole de-refinement (the `mks_smooth` blend).
@inline function jacobian_spatial(c::FMKS, X1, X2)
    e = exp(c.mks_smooth*(c.startx1 - X1))
    dr_dX1  = exp(X1)
    dth_dX1 = -e*c.mks_smooth*(π/2 - π*X2
        + c.poly_norm*(2X2 - 1)*(1 + ((2X2 - 1)/c.poly_xt)^c.poly_alpha/(1 + c.poly_alpha))
        - (1 - c.hslope)/2*sin(2π*X2))
    dth_dX2 = (π + (1 - c.hslope)*π*cos(2π*X2)
        + e*(-π + 2c.poly_norm*(1 + ((2X2 - 1)/c.poly_xt)^c.poly_alpha/(c.poly_alpha + 1))
             + (2c.poly_alpha*c.poly_norm*(2X2 - 1)*((2X2 - 1)/c.poly_xt)^(c.poly_alpha - 1))
                 /((1 + c.poly_alpha)*c.poly_xt)
             - (1 - c.hslope)*π*cos(2π*X2)))
    o = one(dr_dX1); z = zero(dr_dX1)
    @SMatrix [ dr_dX1   z        z ;
               dth_dX1  dth_dX2  z ;
               z        z        o ]
end

# H-AMR native coordinates (HARMPI coord.c `vofx_cylindrified`, Gammie hslope = 1): r = exp(X1), φ = X3,
# θ = π/2·(1 + X2) except the near-pole cells at small r, which are made cylindrical.

"""
    HAMRCyl(; startx1, x10, x20)

H-AMR cylindrified native-coordinate parameters: grid start `startx1` and the cylindrification corner
`(x10, x20)` (`x10 = log(rcyl)`). The transition radius `Xtr1` is derived, matching HARMPI.
"""
struct HAMRCyl{T}
    startx1::T
    x10::T
    x20::T
    Xtr1::T
end

HAMRCyl(; startx1, x10, x20) = HAMRCyl(promote(startx1, x10, x20, log((exp(x10) + exp(startx1))/2))...)

# HARMPI SMALL: regularizes the smooth-max widths.
const HARMPI_SMALL = 1e-20

# Uncylindrified Gammie colatitude, hslope = 1.
_hamr_th0(X2) = π/2*(1 + X2)

# HARMPI smooth-ramp helpers (coord.c Fangle/limlin/mins/maxs/minmaxs), same operation order as the C.
_fangle(x) = x <= -1 ? zero(x) : x >= 1 ? x :
    (1 + x + (-140*sin((π*(1 + x))/2) + (10*sin((3π*(1 + x))/2))/3 + (2*sin((5π*(1 + x))/2))/5)/(64π))/2
_mins(f1, f2, df) = f2 - df*_fangle(-(f1 - f2)/df)
_maxs(f1, f2, df) = -_mins(-f1, -f2, df)
_minmaxs(f1, f2, df, dir) = dir >= 0 ? _maxs(f1, f2, df) : _mins(f1, f2, df)

# coord.c sinth0 / sinth1in / th2in / func1 / func2, specialized to r = exp(X1), θ = θ₀(X2).
_sinth0(c::HAMRCyl, X1) = exp(c.x10)*sin(_hamr_th0(c.x20))/exp(X1)
_sinth1in(c::HAMRCyl, X1, X2) = exp(c.x10)*sin(_hamr_th0(X2))/exp(X1)
function _th2in(c::HAMRCyl, X1, X2)
    tc0 = _hamr_th0(c.x20); th0 = asin(_sinth0(c, X1))
    (_hamr_th0(X2) - tc0)/(π/2 - tc0)*(π/2 - th0) + th0
end
_func1(X2) = sin(_hamr_th0(X2))
_func2(c::HAMRCyl, X1, X2) = _minmaxs(_sinth1in(c, X1, X2), sin(_th2in(c, X1, X2)),
    abs(sin(_th2in(c, X1, -one(X2))) - _sinth1in(c, X1, -one(X2))) + HARMPI_SMALL, X1 - c.x10)

# coord.c vofx_cylindrified: computed in the X2 ≤ 0 half and mirrored, so θ(X1, -X2) = π - θ(X1, X2).
function _hamr_theta(c::HAMRCyl, X1, X2in)
    -1 <= X2in <= 1 || throw(DomainError(X2in, "X2 must lie in [-1, 1]"))
    mirrored = X2in > 0
    X2 = mirrored ? -X2in : X2in
    r = exp(X1)
    dftr = _func2(c, c.Xtr1, X2) - _func1(X2)
    th = asin(_maxs(r*_func1(X2), r*_func2(c, X1, X2), exp(c.Xtr1)*abs(dftr) + HARMPI_SMALL)/r)
    dth = th - _hamr_th0(X2)
    _hamr_th0(X2in) + (mirrored ? -dth : dth)
end

@inline bl_coord(c::HAMRCyl, X1, X2) = (exp(X1), _hamr_theta(c, X1, X2))

# Spatial Jacobian J₃ = ∂(r, θ, φ)_KS / ∂(X1, X2, X3), as for FMKS. The θ row is the forward-mode AD derivative
# of `_hamr_theta` — exact for the same expression, no truncation error; r = exp(X1) and φ = X3 are trivial.
@inline function jacobian_spatial(c::HAMRCyl, X1, X2)
    dth = ForwardDiff.gradient(X -> _hamr_theta(c, X[1], X[2]), SVector(X1, X2))
    o = one(X1); z = zero(X1)
    @SMatrix [ exp(X1)  z       z ;
               dth[1]   dth[2]  z ;
               z        z       o ]
end

"""
    HAMRSph()

H-AMR native coordinates without cylindrification: r = exp(X1), θ = π/2·(1 + X2), φ = X3.
"""
struct HAMRSph end

@inline bl_coord(::HAMRSph, X1, X2) = (exp(X1), _hamr_th0(X2))
@inline jacobian_spatial(::HAMRSph, X1, X2) = Diagonal(SVector(exp(X1), π/2, 1))
