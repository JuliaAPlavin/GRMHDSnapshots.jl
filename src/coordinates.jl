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
