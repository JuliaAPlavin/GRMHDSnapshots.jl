using TestItems
using TestItemRunner
@run_package_tests


@testitem "store loader + sample" begin
    using StaticArrays, Zarr
    using GRMHDSnapshots: sample

    store = zopen(joinpath(@__DIR__, "data", "koral_sample.zarr"), "r")
    grid = load_grid(store)
    @test length(grid.r_prof) == 48
    @test size(grid.th_grid) == (48, 32)

    snap = load_koral(store, "snap002000")
    @test snap isa GRMHDSnapshot
    @test size(snap.rho) == (24, 32, 48)
    @test size(snap.velrel) == (24, 32, 48)
    @test size(snap.bfield) == (24, 32, 48)
    @test r_min(snap) < r_max(snap)
    @test all(>(0), diff(snap.r_prof))

    ρ, ũ, B = @inferred sample(snap, 5.0, 0.0, 0.0)
    @test ρ isa Float64 && isfinite(ρ)
    @test ũ isa SVector{3,Float64} && all(isfinite, ũ)
    @test B isa SVector{3,Float64} && all(isfinite, B)

    i, j, k = 20, 16, 12
    rn = snap.r_prof[i]; θn = snap.th_grid[i, j]; φn = snap.φ0 + (k - 1) * snap.dφ
    xn = rn * sin(θn) * cos(φn); yn = rn * sin(θn) * sin(φn); zn = rn * cos(θn)
    @test first(sample(snap, xn, yn, zn)) ≈ snap.rho[φ=k, θ=j, r=i]

    ρ0, ũ0, B0 = @inferred sample(snap, 0.0, 0.0, 10 * r_max(snap))
    @test ρ0 == 0 && iszero(ũ0) && iszero(B0)

    snap_rho = load_koral(store, "snap002000"; fields = (:rho,))
    @test snap_rho.velrel === nothing && snap_rho.bfield === nothing
    @test_throws MethodError sample(snap_rho, 5.0, 0.0, 0.0)

    # single-field interpolation: matches the bundled result, needs only its own field
    using GRMHDSnapshots: sample_field
    @test @inferred(sample_field(snap, snap.velrel, 5.0, 0.0, 0.0)) == ũ
    @test @inferred(sample_field(snap, snap.rho, xn, yn, zn)) ≈ snap.rho[φ=k, θ=j, r=i]
    B0f = @inferred sample_field(snap, snap.bfield, 0.0, 0.0, 10 * r_max(snap))
    @test B0f isa SVector{3,Float64} && iszero(B0f)

    snap_vel = load_koral(store, "snap002000"; fields = (:velrel,))
    @test snap_vel.rho === nothing && snap_vel.bfield === nothing
    @test sample_field(snap_vel, snap_vel.velrel, 5.0, 0.0, 0.0) == ũ   # works with rho/bfield absent
end

@testitem "extension-sniffing (one-arg zarr snap path)" begin
    using Zarr

    store_path = joinpath(@__DIR__, "data", "koral_sample.zarr")
    snap1 = load_koral(joinpath(store_path, "snap002000"))
    snap2 = load_koral(zopen(store_path, "r"), "snap002000")
    @test snap1.rho == snap2.rho
end

@testitem "Adapt identity move preserves axis names & values" begin
    using Adapt, NamedDims, Zarr

    snap = load_koral(zopen(joinpath(@__DIR__, "data", "koral_sample.zarr"), "r"), "snap002000")
    snap2 = Adapt.adapt_structure(Array, snap)
    @test dimnames(snap2.rho) == dimnames(snap.rho) == (:φ, :θ, :r)
    @test dimnames(snap2.velrel) == (:φ, :θ, :r)
    @test snap2.rho == snap.rho
    @test snap2.velrel == snap.velrel
    @test snap2.bfield == snap.bfield
end

@testitem "raw .h5 cross-check (optional, isfile-guarded)" begin
    using HDF5, StaticArrays, Zarr
    using GRMHDSnapshots: sample

    h5path = joinpath(@__DIR__, "data", "ipole_a0.9_2000.h5")
    if isfile(h5path)
        s_mem = load_koral(h5path; mmap = false)
        s_mm = load_koral(h5path; mmap = true)
        @test s_mem.rho == s_mm.rho
        @test s_mem.velrel == s_mm.velrel
        @test s_mem.bfield == s_mm.bfield

        @test r_min(s_mem) < r_max(s_mem)
        @test all(>(0), diff(s_mem.r_prof))
        ρ, ũ, B = sample(s_mem, 5.0, 0.0, 0.0)
        @test isfinite(ρ) && all(isfinite, ũ) && all(isfinite, B)

        s_store = load_koral(zopen(joinpath(@__DIR__, "data", "koral_sample.zarr"), "r"), "snap002000")
        sl = (1:6:144, 1:6:192, 1:6:288)
        scale_relerr(loaded, hpath) = h5open(h5path, "r") do h5
            b = read(h5[hpath])[sl...]
            maximum(abs.(Float64.(collect(loaded)) .- b)) / maximum(abs.(b))
        end
        re_rho = scale_relerr(s_store.rho, "quants/rho")
        re_B1 = scale_relerr(map(v -> v[1], s_store.bfield), "quants/B1")
        @test 0.010 < re_rho < 0.020
        @test 0.004 < re_B1 < 0.012
    else
        @test_skip "raw .h5 fixture absent"
    end
end

@testitem "plasma physics" begin
    using StaticArrays, Zarr, NamedDims
    using LinearAlgebra: det
    using GRMHDSnapshots: plasma_state, ks_gcov

    store = zopen(joinpath(@__DIR__, "data", "koral_sample.zarr"), "r")
    snap = load_koral(store, "snap002000")
    a = snap.spin

    # Physical invariants of the reconstructed plasma state at sampled cells:
    # u·u = -1 (timelike, normalized) and b·u = 0 (comoving field ⊥ 4-velocity), bsq ≥ 0.
    for (i, j, k) in ((20, 16, 12), (10, 8, 3), (40, 24, 20), (5, 30, 1))
        r = snap.r_prof[i]; θ = snap.th_grid[i, j]
        ũ = snap.velrel[φ=k, θ=j, r=i]; B = snap.bfield[φ=k, θ=j, r=i]
        g = ks_gcov(r, θ, a)
        (; u, b, bsq) = plasma_state(ũ, B, r, θ, a)
        @test u' * g * u ≈ -1 rtol = 1e-6
        @test b' * g * u ≈ 0 atol = 1e-6 * sqrt(abs(u' * g * u) * max(bsq, eps()))
        @test bsq ≥ 0
    end

    # Lapse → 1 far away; finite and in (0,1] at moderate r.
    @test lapse(1e6, 1.0, a) ≈ 1 rtol = 1e-5
    for r in (2.0, 5.0, 20.0), θ in (0.3, 1.2, 2.7)
        α = lapse(r, θ, a)
        @test isfinite(α) && 0 < α ≤ 1
    end

    # grav_redshift g = √(−g_tt) = √(1 − 2r/(r²+a²cos²θ)): independently hardcoded, → 1 far away,
    # → 0 at the horizon, and clamped to 0 inside the ergosphere. NOT equal to the lapse.
    @test grav_redshift(1e6, 1.0, a) ≈ 1 rtol = 1e-5
    @test grav_redshift(5.0, π/2, 0.0) ≈ sqrt(1 - 2/5)              # Schwarzschild equator, √0.6
    @test grav_redshift(2.0, π/2, 0.0) == 0.0                        # Schwarzschild horizon → infinite redshift
    @test grav_redshift(1.5, π/2, 0.9) == 0.0                        # inside ergosphere (equator, r<2) → clamped
    @test grav_redshift(5.0, π/2, 0.0) ≠ lapse(5.0, π/2, 0.0)        # distinct from the lapse

    # volume_ratio √γ/(r²sinθ): closed form (r²+a²cos²θ)/(r²α) must equal the numeric spatial-metric
    # determinant √det(γ)/(r²sinθ); → 1 far away; = 1/α on the equator (ρ²=r² there); ≥ 1.
    for (r, θ, a) in [(1.3, 0.6, 0.9), (2.0, 1.2, 0.9), (5.0, 2.5, 0.5), (3.0, 0.3, 0.99)]
        γdet = det(ks_gcov(r, θ, a)[SVector(2, 3, 4), SVector(2, 3, 4)])
        @test volume_ratio(r, θ, a) ≈ sqrt(γdet)/(r^2*sin(θ)) rtol = 1e-12
        @test volume_ratio(r, θ, a) ≥ 1
    end
    @test volume_ratio(1e6, 1.0, 0.9) ≈ 1 rtol = 1e-5
    @test volume_ratio(5.0, π/2, 0.9) ≈ 1/lapse(5.0, π/2, 0.9)       # equator: √γ/(r²sinθ) = 1/α

    # Grid arrays: correct shape/names, finite where ρ>0, kernel-consistent.
    bsqg = comoving_bsq(snap)
    @test dimnames(bsqg) == (:φ, :θ, :r)
    @test size(bsqg) == size(snap.rho)
    @test all(≥(0), bsqg)
    @test all(isfinite, comoving_B_gauss(snap))
    lap = lapse(snap)
    @test size(lap) == size(snap.rho) && all(α -> 0 < α ≤ 1, lap)
    σ = magnetization(snap)
    @test all(isfinite, σ[snap.rho .> 0])

    # comoving_bsq grid value equals the kernel on that cell's primitives.
    i, j, k = 20, 16, 12
    @test bsqg[φ=k, θ=j, r=i] ≈ plasma_state(snap.velrel[φ=k, θ=j, r=i], snap.bfield[φ=k, θ=j, r=i],
                                             snap.r_prof[i], snap.th_grid[i, j], a).bsq

    # map_plasma hoists the (r,θ)-only metric across φ and threads over r: over the whole grid it must
    # match a naive per-cell reconstruction (independent reference) up to FMA reassociation, and its
    # result must be bit-identical regardless of thread count.
    nφ, nθ, nr = size(snap.rho)
    bsq_ref = [plasma_state(snap.velrel[φ=k, θ=j, r=i], snap.bfield[φ=k, θ=j, r=i],
                            snap.r_prof[i], snap.th_grid[i, j], a).bsq for k in 1:nφ, j in 1:nθ, i in 1:nr]
    @test bsqg ≈ bsq_ref rtol = 1e-13
    @test map_plasma((ps, _, _) -> ps.bsq, snap; threaded = true) ==
          map_plasma((ps, _, _) -> ps.bsq, snap; threaded = false)
    @inferred comoving_bsq(snap)

    # comoving_bsq/comoving_B_gauss reconstruct b from ũ and B only — no rho needed (regression:
    # map_plasma used to index snap.rho for every kernel, crashing on a rho-less snapshot).
    snap_bv = load_koral(store, "snap002000"; fields = (:velrel, :bfield))
    @test snap_bv.rho === nothing
    @test comoving_bsq(snap_bv) == bsqg
    @test comoving_B_gauss(snap_bv) == comoving_B_gauss(snap)

    # lapse(snap) is φ-invariant and equals the scalar lapse at each (r,θ), exactly.
    @test lap == [lapse(snap.r_prof[i], snap.th_grid[i, j], a) for k in 1:nφ, j in 1:nθ, i in 1:nr]

    # grav_redshift(snap) is φ-invariant and equals the scalar grav_redshift at each (r,θ), exactly.
    zg = grav_redshift(snap)
    @test size(zg) == size(snap.rho) && all(x -> 0 ≤ x ≤ 1, zg)
    @test zg == [grav_redshift(snap.r_prof[i], snap.th_grid[i, j], a) for k in 1:nφ, j in 1:nθ, i in 1:nr]

    # volume_ratio(snap) is φ-invariant, equals the scalar at each (r,θ), and is ≥ 1 everywhere.
    vr = volume_ratio(snap)
    @test size(vr) == size(snap.rho) && all(≥(1), vr)
    @test vr == [volume_ratio(snap.r_prof[i], snap.th_grid[i, j], a) for k in 1:nφ, j in 1:nθ, i in 1:nr]

    # Unit scalars are positive plain Float64.
    @test lunit(snap) isa Float64 && lunit(snap) > 0
    @test rhounit(snap) isa Float64 && rhounit(snap) > 0
    @test bunit(snap) isa Float64 && bunit(snap) > 0

    # Kernel type stability on Float64 inputs.
    @inferred plasma_state(SVector(0.1, 0.2, 0.3), SVector(0.4, 0.5, 0.6), 5.0, 1.0, a)
end

@testitem "velocity direction: u^μ reconstruction & flow" begin
    using StaticArrays
    using GRMHDSnapshots: fluid_ucon, fluid_velocity, ks_gcov

    # ũ=0 (fluid at rest wrt the KS normal observer) ⇒ pure radial infall. Closed form (hardcoded,
    # derived independently): u^t=√(1+tr), u^r=-tr/√(1+tr), tr=2r/(r²+a²cos²θ), u^θ=u^φ=0.
    for (r, θ, expect) in ((3.0,  π/2, SVector(1.2909944487358056, -0.5163977794943222, 0.0, 0.0)),
                           (10.0, 0.7, SVector(1.0950145185619904, -0.18178461790895298, 0.0, 0.0)),
                           (1.3,  1.0, SVector(1.5328487747517299, -0.8804686988620086, 0.0, 0.0)))
        @test (@inferred fluid_ucon(SVector(0.0, 0.0, 0.0), r, θ, 0.9)) ≈ expect
    end

    # A valid, self-consistent 4-velocity: future-pointing, u·u=-1, and the ipole resynthesis
    # ũ^i = g^{ti}α²u^t + u^i inverts it exactly (independent inverse ⇒ genuine cross-check).
    invprim(u, r, θ, a) = let gc = inv(ks_gcov(r, θ, a)), α2 = -1/gc[1, 1]
        SVector(gc[1, 2]*α2*u[1] + u[2], gc[1, 3]*α2*u[1] + u[3], gc[1, 4]*α2*u[1] + u[4])
    end
    for a in (0.0, 0.5, 0.9, 0.998), r in (1.3, 2.0, 5.0, 50.0, 500.0), θ in (0.2, π/2, 2.9),
        ũ in (SVector(0.3, 0.1, -0.2), SVector(2.0, -0.05, 1.0), SVector(-1.0, 0.2, 3.0))
        u = fluid_ucon(ũ, r, θ, a)
        @test u[1] > 0
        @test u' * ks_gcov(r, θ, a) * u ≈ -1 rtol = 1e-8
        @test invprim(u, r, θ, a) ≈ ũ
        @test fluid_velocity(ũ, r, θ, a) ≈ SVector(u[2], u[3], u[4]) / u[1]
    end

    # Causality: between the two horizons every future-timelike worldline has dr/dτ<0, so the
    # reconstructed u^r is inward for ANY ũ — while the raw primitive ũ^r is not sign-constrained.
    for a in (0.0, 0.5, 0.9, 0.998)
        r_out = 1 + sqrt(1 - a^2); r_in = 1 - sqrt(1 - a^2)
        rs = range(r_in + 0.05*(r_out - r_in), r_out - 0.01*(r_out - r_in), length = 6)
        θs = range(0.1, π - 0.1, length = 5)
        ũs = [SVector(vr, vt, vp) for vr in (-3.0, 0.0, 3.0, 20.0) for vt in (-0.5, 0.5) for vp in (-2.0, 2.0)]
        @test all(fluid_ucon(ũ, r, θ, a)[2] < 0 for r in rs, θ in θs, ũ in ũs)
        @test any(ũ -> ũ[1] > 0, ũs)   # ũ^r is genuinely outward for some inputs ⇒ the test discriminates
    end
end

@testitem "Kerr-Schild → Boyer-Lindquist vectors" begin
    using StaticArrays, ForwardDiff
    using GRMHDSnapshots: ks_gcov, fluid_ucon

    # BL(KS) coordinate map: t_BL = t − T(r), φ_BL = φ − Φ(r), with T' = 2r/Δ, Φ' = a/Δ integrated in closed
    # form by partial fractions over the horizons r± (abs: also valid between them).
    function bl_coords(x, a)
        t, r, θ, φ = x
        rp, rm = 1 + √(1 - a^2), 1 - √(1 - a^2)
        T = (2rp*log(abs(r - rp)) - 2rm*log(abs(r - rm)))/(rp - rm)
        Φ = a/(rp - rm)*log(abs((r - rp)/(r - rm)))
        SVector(t - T, r, θ, φ - Φ)
    end
    # standard Boyer-Lindquist covariant metric
    function bl_gcov(r, θ, a)
        Σ = r^2 + a^2*cos(θ)^2; Δ = r^2 - 2r + a^2; s2 = sin(θ)^2
        @SMatrix [ -(1 - 2r/Σ)   0    0   -2a*r*s2/Σ ;
                    0            Σ/Δ  0    0 ;
                    0            0    Σ    0 ;
                   -2a*r*s2/Σ    0    0    (r^2 + a^2 + 2a^2*r*s2/Σ)*s2 ]
    end

    vs = (SVector(1.0, 0.3, -0.02, 0.1), SVector(-0.5, -2.0, 0.4, 1.5), SVector(0.0, 1.0, 0.0, 0.0))
    ũs = (SVector(0.3, 0.1, -0.2), SVector(-1.0, 0.2, 3.0), SVector(2.0, -0.05, 1.0))
    # points outside the outer horizon and between the horizons (Δ < 0)
    for a in (0.0, 0.5, 0.9375, 0.998), r in (1.2, 1.5, 2.3, 6.0, 40.0), θ in (0.2, π/2, 2.6)
        J = ForwardDiff.jacobian(x -> bl_coords(x, a), SVector(0.7, r, θ, -1.1))
        gks = ks_gcov(r, θ, a); gbl = bl_gcov(r, θ, a)
        # coordinate vectors and timelike fluid velocities
        for v in (vs..., map(ũ -> fluid_ucon(ũ, r, θ, a), ũs)...)
            vbl = @inferred ks_to_bl(v, r, θ, a)
            @test vbl ≈ J*v rtol = 1e-12
            @test vbl'*gbl*vbl ≈ v'*gks*v rtol = 1e-10 atol = 1e-10*sum(abs2, v)
        end
        # metric pullback: the KS metric is J' g_BL J
        @test J'*gbl*J ≈ gks rtol = 1e-10 atol = 1e-10
        # vʳ = 0: unchanged
        @test ks_to_bl(SVector(1.3, 0.0, -0.2, 0.05), r, θ, a) == SVector(1.3, 0.0, -0.2, 0.05)
    end
    # precision is carried through
    @test ks_to_bl(SVector(1f0, 0.5f0, 0f0, 0.1f0), 6f0, 1f0, 0.9f0) isa SVector{4,Float32}
end

@testitem "field magnitudes: metric norms, not raw component norms" begin
    using StaticArrays
    using GRMHDSnapshots: flow_speed, proper_velocity, spatial_norm

    # flow_speed = physical v/c relative to the normal observer: 0 at rest, in [0,1), and equal to
    # √(1-1/γ²) with γ derived the INDEPENDENT way from the relative velocity: γ=√(1+|ũ|²_spatial)
    # (HARM identity), not via the reconstruction's u^t path that flow_speed uses internally.
    # proper_velocity = βγ = √(γ²-1), same γ/frame, unbounded; the HARM identity makes it equal to
    # |ũ|_spatial exactly, so it's checked the same independent way.
    @test flow_speed(SVector(0.0, 0.0, 0.0), 5.0, 1.0, 0.9) == 0
    @test proper_velocity(SVector(0.0, 0.0, 0.0), 5.0, 1.0, 0.9) == 0
    for a in (0.0, 0.9), r in (1.3, 3.0, 30.0), θ in (0.3, π/2),
        ũ in (SVector(2.0, -0.05, 1.0), SVector(-1.0, 0.2, 3.0), SVector(0.3, 0.1, -0.2))
        γ = sqrt(1 + spatial_norm(ũ, r, θ, a)^2)
        @test flow_speed(ũ, r, θ, a) ≈ sqrt(1 - 1/γ^2)
        @test 0 ≤ flow_speed(ũ, r, θ, a) < 1
        @test proper_velocity(ũ, r, θ, a) ≈ sqrt(γ^2 - 1) ≈ spatial_norm(ũ, r, θ, a)
        @test proper_velocity(ũ, r, θ, a) ≈ flow_speed(ũ, r, θ, a)*γ   # βγ = β·γ
    end

    # spatial_norm = √(γᵢⱼVⁱVʲ). Hardcoded against the KS 3-metric written out by hand at one cell:
    # r=2, θ=π/2, a=0.9 ⇒ tr=2·2/4=1, ρ²=4; g_rr=1+tr=2, g_θθ=ρ²=4, g_φφ=ρ²+a²(1+tr)=4+1.62=5.62,
    # g_rφ=-a(1+tr)=-1.8. For V=(1,1,1): |V|²=2+4+5.62+2·(-1.8)=8.02.
    @test spatial_norm(SVector(1.0, 1.0, 1.0), 2.0, π/2, 0.9) ≈ sqrt(8.02)
    # a raw Euclidean norm would give √3 ≈ 1.732 — materially different, i.e. the fix matters.
    @test !isapprox(spatial_norm(SVector(1.0, 1.0, 1.0), 2.0, π/2, 0.9), sqrt(3); rtol = 0.05)
end

@testitem "FMKS coordinates: analytic Jacobian matches numeric" begin
    using GRMHDSnapshots: FMKS, bl_coord, jacobian_spatial

    # No data file needed — pure coordinate math (runs everywhere). The analytic spatial Jacobian must
    # equal the finite-difference derivative of bl_coord; r = exp(X1); φ-row/column are trivial.
    c = FMKS(startx1 = 0.57, hslope = 0.3, mks_smooth = 0.5, poly_xt = 0.82, poly_alpha = 14.0)
    h = 1e-6
    for X1 in (0.6, 1.5, 3.0, 6.0), X2 in (0.05, 0.3, 0.496, 0.6, 0.95)   # incl. X2<0.5 (2X2−1<0)
        dr_dX1  = (first(bl_coord(c, X1+h, X2)) - first(bl_coord(c, X1-h, X2)))/2h
        dth_dX1 = (last(bl_coord(c, X1+h, X2))  - last(bl_coord(c, X1-h, X2)))/2h
        dth_dX2 = (last(bl_coord(c, X1, X2+h))  - last(bl_coord(c, X1, X2-h)))/2h
        J = jacobian_spatial(c, X1, X2)
        @test J[1, 1] ≈ dr_dX1  rtol = 1e-5
        @test J[2, 1] ≈ dth_dX1 rtol = 1e-5 atol = 1e-7
        @test J[2, 2] ≈ dth_dX2 rtol = 1e-5
        @test J[3, 3] == 1
        @test J[1, 2] == 0 && J[1, 3] == 0 && J[2, 3] == 0 && J[3, 1] == 0 && J[3, 2] == 0
        @test first(bl_coord(c, X1, X2)) ≈ exp(X1)
    end
end

@testitem "iharm FMKS loader: synthetic round-trip" begin
    using HDF5, StaticArrays, NamedDims, Accessors
    using GRMHDSnapshots: FMKS, bl_coord, jacobian_spatial, plasma_state, ks_gcov

    # Write a tiny valid FMKS dump and load it — exercises header parsing, prims unpacking, the grid
    # build and the native→KS Jacobian transform end-to-end, with no external fixture (CI-safe).
    n1, n2, n3, nprim = 4, 3, 2, 8
    sx1, sx2, sx3 = 0.5, 0.0, 0.0
    dx1, dx2, dx3 = 0.4, 1/n2, 2π/n3
    par = (a = 0.7, hslope = 0.3, mks_smooth = 0.5, poly_xt = 0.82, poly_alpha = 14.0)
    prims = reshape(collect(1:nprim*n3*n2*n1) .* 0.01, nprim, n3, n2, n1)   # deterministic, small ⇒ valid ũ

    path = joinpath(mktempdir(), "mini_fmks.h5")
    h5open(path, "w") do h5
        write(h5, "header/metric", "FMKS")
        write(h5, "header/n1", n1); write(h5, "header/n2", n2); write(h5, "header/n3", n3)
        write(h5, "header/gam", 1.444444); write(h5, "t", 123.0)
        write(h5, "header/geom/startx1", sx1); write(h5, "header/geom/startx2", sx2); write(h5, "header/geom/startx3", sx3)
        write(h5, "header/geom/dx1", dx1); write(h5, "header/geom/dx2", dx2); write(h5, "header/geom/dx3", dx3)
        write(h5, "header/geom/mmks/a", par.a); write(h5, "header/geom/mmks/hslope", par.hslope)
        write(h5, "header/geom/mmks/mks_smooth", par.mks_smooth)
        write(h5, "header/geom/mmks/poly_xt", par.poly_xt); write(h5, "header/geom/mmks/poly_alpha", par.poly_alpha)
        write(h5, "prims", prims)
    end

    snap = load_iharm(path)
    @test size(snap.rho) == (n3, n2, n1) && size(snap.velrel) == (n3, n2, n1)
    @test snap.spin == par.a && snap.gam ≈ 1.444444 && snap.t == 123.0
    @test snap.M_unit === missing && snap.MBH === missing         # scale-free by default
    @test snap.φ0 ≈ sx3 + 0.5dx3 && snap.dφ ≈ dx3                 # φ = X3, no +π offset

    # units are independently optional: lunit needs MBH only, rhounit/bunit both; errors name the missing ones.
    @test_throws "lunit needs MBH (grams)" lunit(snap)
    foreach(f -> @test_throws("$f needs M_unit and MBH", f(snap)), (rhounit, bunit))
    snapb = load_iharm(path; MBH = 1.3e43)
    @test snapb.M_unit === missing
    @test lunit(snapb) ≈ 6.6743e-8*1.3e43/2.99792458e10^2 rtol = 1e-6
    foreach(f -> @test_throws("$f needs M_unit (grams)", f(snapb)), (rhounit, bunit))
    snapm = load_iharm(path; M_unit = 1e26)
    @test snapm.MBH === missing
    foreach(f -> @test_throws("$f needs MBH (grams)", f(snapm)), (lunit, rhounit, bunit))
    # setting the other unit later (as downstream viewers do for M_unit) rebuilds the snapshot with its new type
    snapbm = @set snapb.M_unit = 1e26
    @test snapbm.M_unit === 1e26 && snapbm.rho == snap.rho
    @test rhounit(snapbm) ≈ 1e26/lunit(snapb)^3 && bunit(snapbm) ≈ 2.99792458e10*sqrt(4π*rhounit(snapbm))
    @test rhounit(@set snapm.MBH = 1.3e43) == rhounit(snapbm)
    @test (@set snapbm.M_unit = missing).M_unit === missing

    c = FMKS(startx1 = sx1, hslope = par.hslope, mks_smooth = par.mks_smooth, poly_xt = par.poly_xt, poly_alpha = par.poly_alpha)
    for i in 1:n1, j in 1:n2, k in 1:n3
        X1 = sx1 + (i-0.5)*dx1; X2 = sx2 + (j-0.5)*dx2
        r, th = bl_coord(c, X1, X2); J = jacobian_spatial(c, X1, X2)
        @test snap.r_prof[i] ≈ r
        @test snap.th_grid[i, j] ≈ th
        @test snap.rho[φ=k, θ=j, r=i] ≈ prims[1, k, j, i]
        @test snap.velrel[φ=k, θ=j, r=i] ≈ J*SVector(prims[3, k, j, i], prims[4, k, j, i], prims[5, k, j, i])
        @test snap.bfield[φ=k, θ=j, r=i] ≈ J*SVector(prims[6, k, j, i], prims[7, k, j, i], prims[8, k, j, i])
        u = plasma_state(snap.velrel[φ=k, θ=j, r=i], snap.bfield[φ=k, θ=j, r=i], r, th, par.a).u
        @test u' * ks_gcov(r, th, par.a) * u ≈ -1 rtol = 1e-10   # valid 4-velocity after transform
    end
end

@testitem "iharm FMKS loader: real data + pyharm golden (optional, isfile-guarded)" begin
    using StaticArrays, NamedDims
    using GRMHDSnapshots: plasma_state, ks_gcov

    path = joinpath(@__DIR__, "data", "iharm_fmks_a0.h5")
    if isfile(path)
        snap = load_iharm(path)
        @test size(snap.rho) == (128, 128, 288)
        @test r_min(snap) < r_max(snap) && all(>(0), diff(snap.r_prof))
        @test snap.spin == 0.0

        # lazy units: scale-free by default ⇒ throw; supplied ⇒ positive scalars.
        @test_throws ArgumentError lunit(snap)
        snapu = load_iharm(path; M_unit = 1e26, MBH = 1.3e43)
        @test lunit(snapu) > 0 && rhounit(snapu) > 0 && bunit(snapu) > 0
        # auto-detect routes an FMKS .h5 to load_iharm.
        @test load_snapshot(path).rho == snap.rho

        # Golden values from pyharm (see gen_fmks_golden.py), an independent reference reader.
        # 0-based native (i=radial, j=θ, k=φ); bsq/sigma/Gamma are frame-invariant ⇒ compared directly.
        gold = [(59, 63, 9,    6.56053574719, 1.56595075338, 0.68453335762,  0.00538246981389,  0.0078629766599,  1.1482527967),
                (119, 39, 69,  24.5568056735, 1.29190538653, 0.162381529808, 0.000236250350958, 0.00145490901112, 1.02877584453),
                (29, 99, 4,    3.39096277761, 2.02063373719, 0.0109128253534, 0.382428746929,   35.0439720735,    1.02475536928),
                (199, 109, 119, 142.719055604, 2.37849777729, 0.0111680263653, 9.7388717037e-6, 0.000872031582404, 1.00266551412)]
        for (i, j, k, rg, thg, rhog, bsqg, sigg, Gamg) in gold
            I, J, K = i+1, j+1, k+1
            r = snap.r_prof[I]; th = snap.th_grid[I, J]; a = snap.spin
            rho = snap.rho[φ=K, θ=J, r=I]
            ps = plasma_state(snap.velrel[φ=K, θ=J, r=I], snap.bfield[φ=K, θ=J, r=I], r, th, a)
            Γ = lapse(r, th, a) * ps.u[1]
            @test r ≈ rg rtol = 1e-8
            @test th ≈ thg rtol = 1e-8
            @test rho ≈ rhog rtol = 1e-6
            @test ps.bsq ≈ bsqg rtol = 1e-6
            @test ps.bsq/rho ≈ sigg rtol = 1e-6
            @test Γ ≈ Gamg rtol = 1e-6
            @test ps.u' * ks_gcov(r, th, a) * ps.u ≈ -1 rtol = 1e-8
            @test ps.b' * ks_gcov(r, th, a) * ps.u ≈ 0 atol = 1e-6*sqrt(max(ps.bsq, eps()))
        end
    else
        @test_skip "iharm FMKS fixture absent"
    end
end

@testitem "H-AMR cylindrified coordinates" begin
    using GRMHDSnapshots: HAMRCyl, bl_coord, jacobian_spatial

    # EHT code-comparison run geometry (startx1 from its `parameters`, rcyl = 24.32, 1056 θ cells).
    c = HAMRCyl(startx1 = 0.19325057145871735, x10 = log(24.32), x20 = -1 + 1/1056)

    # Golden θ and finite-differenced dxdxp written by H-AMR itself into the run's gdump files (blocks 109,
    # 17825, 547): an independent reference, incl. a strongly cylindrified cell near the pole at small r.
    for (X1, X2, θ, dθ_dX1, dθ_dX2) in (
            (0.19512288412257414, -0.9555731318870523, 0.09683481083775666, -0.028480421330300647, 1.5424894932385946),
            (2.704021853690688,    0.9971608844123048, 3.136200683926511,    0.0024136447729203397, 1.5698613743913405),
            (0.4572466570625263,  -0.6455553403351699, 0.5705978640536499,  -0.014801254022408136, 1.5493613022014587))
        r, th = bl_coord(c, X1, X2); J = jacobian_spatial(c, X1, X2)
        @test r == exp(X1)
        @test th ≈ θ rtol = 1e-14
        @test J[2, 1] ≈ dθ_dX1 atol = 1e-7
        @test J[2, 2] ≈ dθ_dX2 atol = 1e-7
    end
    @test !isapprox(last(bl_coord(c, 0.19512288412257414, -0.9555731318870523)), π/2*(1 - 0.9555731318870523); rtol = 0.1)

    # Equatorial symmetry θ(X1, -X2) = π - θ(X1, X2), and plain Gammie θ = π/2·(1+X2) far outside rcyl.
    for X1 in (0.2, 1.0, 2.5, 3.19, 3.5, 6.0), X2 in (-0.999, -0.99, -0.9, -0.5, -0.1, 0.0)
        @test last(bl_coord(c, X1, -X2)) ≈ π - last(bl_coord(c, X1, X2)) atol = 1e-14
    end
    for X1 in (log(24.32) + 2, 6.2), X2 in (-0.999, -0.5, 0.3, 0.99)
        @test last(bl_coord(c, X1, X2)) ≈ π/2*(1 + X2) rtol = 1e-14
    end
    @test_throws DomainError bl_coord(c, 1.0, 1.2)

    # The AD Jacobian equals the finite-difference derivative of bl_coord; r = exp(X1); φ row/column trivial.
    h = 1e-6
    for X1 in (0.2, 1.0, 2.5, 3.19, 4.0), X2 in (-0.995, -0.9, -0.5, 0.2, 0.97)
        dth_dX1 = (last(bl_coord(c, X1+h, X2)) - last(bl_coord(c, X1-h, X2)))/2h
        dth_dX2 = (last(bl_coord(c, X1, X2+h)) - last(bl_coord(c, X1, X2-h)))/2h
        J = @inferred jacobian_spatial(c, X1, X2)
        @test J[1, 1] ≈ exp(X1)
        @test J[2, 1] ≈ dth_dX1 atol = 1e-8
        @test J[2, 2] ≈ dth_dX2 rtol = 1e-8
        @test J[3, 3] == 1
        @test J[1, 2] == 0 && J[1, 3] == 0 && J[2, 3] == 0 && J[3, 1] == 0 && J[3, 2] == 0
    end
end

@testitem "H-AMR loader: synthetic round-trip" begin
    using StaticArrays, NamedDims
    using GRMHDSnapshots: HAMRCyl, bl_coord, jacobian_spatial, ks_gcov, fluid_ucon

    # A tiny H-AMR dump dir: 2 radial rings × 3 θ rows at φ levels (0, 2, 1) ⇒ 4/16/8 φ cells per row on a
    # 16-cell finest grid; block ids scrambled and non-contiguous; 4 blocks per new_dump file (last one short).
    bs = (4, 2, 4); nb = (2, 3, 1); rowlevel = (0, 2, 1)
    n1, n2, n3 = 8, 6, 16
    sx = (log(2.5), -1 + 1/6, 0.0); dx = (0.3, (2 - 2/6)/n2, 2π/(nb[3]*bs[3]))
    a, gam, t, rcyl = 0.8, 13/9, 1234.5, 3.0
    c = HAMRCyl(startx1 = sx[1], x10 = log(rcyl), x20 = -1 + 1/n2)
    X1(i) = sx[1] + (i + 1/2)*dx[1]; X2(j) = sx[2] + (j + 1/2)*dx[2]     # 0-based global cells
    rθ(i, j) = bl_coord(c, X1(i), X2(j))

    blks = [(; level = rowlevel[c2+1], c1, c2, c3) for c1 in 0:nb[1]-1 for c2 in 0:nb[2]-1 for c3 in 0:nb[3]*2^rowlevel[c2+1]-1]
    nact = length(blks)
    ids = 3 .* mod.(5 .* (0:nact-1), nact) .+ 1
    nmax = 3nact + 2

    # Known KS fields at global cell (i, j, coarse φ cell m): ρ, the 4-velocity's spatial part (u^t from
    # the KS normalization, future root) and the lab field; the dump holds them in native components.
    ρ(i, j, m) = 1 + i + 10j + 100m
    uks(i, j, m) = let ui = SVector(-0.1 + 0.01i, 0.002j - 0.005, 0.05 + 0.01m), (r, th) = rθ(i, j), g = ks_gcov(r, th, a)
        A = g[1, 1]; B = 2*(g[1, 2]*ui[1] + g[1, 3]*ui[2] + g[1, 4]*ui[3]); C = ui'*g[2:4, 2:4]*ui + 1
        SVector((-B - sqrt(B^2 - 4A*C))/(2A), ui...)
    end
    Bks(i, j, m) = SVector(0.5 + 0.1m, -0.1 + 0.01j, 0.2 + 0.001i)
    # relative velocity via the full inverse metric (independent of the loader's closed form)
    ũks(i, j, m) = let u = uks(i, j, m), gc = inv(ks_gcov(rθ(i, j)..., a))
        SVector(u[2] - u[1]*gc[1, 2]/gc[1, 1], u[3] - u[1]*gc[1, 3]/gc[1, 1], u[4] - u[1]*gc[1, 4]/gc[1, 1])
    end
    function cellrec(b, ii, jj, kk)
        i = b.c1*bs[1] + ii; j = b.c2*bs[2] + jj; m = b.c3*bs[3] + kk
        Ji = inv(jacobian_spatial(c, X1(i), X2(j))); u = uks(i, j, m)
        SVector{9,Float32}(ρ(i, j, m), 7, u[1], (Ji*u[SVector(2, 3, 4)])..., (Ji*Bks(i, j, m))...)
    end

    dir = mktempdir()
    open(joinpath(dir, "parameters"), "w") do io
        w(xs...) = foreach(x -> write(io, htol(x)), xs)
        w(t); w(Int32.((nact, nact, 100))...); w(0.0, 0.0, 0.0); w(Int32(1), Int32(1)); w(0.1); w(Int32(0))
        w(Int32.(bs)...); w(Int32(nmax)); w(Int32.(nb)...)
        w(sx...); w(dx...); w(2000.0, a, gam, 0.9, exp(sx[1]), exp(sx[1] + n1*dx[1]), 0.0, 1.0)
        w(zeros(Int32, 15)...)
        active = zeros(Int32, nmax); active[ids .+ 1] .= 1
        w(zeros(Int32, nmax)...); w(active...)
    end
    open(joinpath(dir, "grid"), "w") do io
        g = fill(Int32(-1), 36, nmax); g[1, :] .= 0
        foreach((id, b) -> g[1:6, id+1] = [1, b.level, 0, b.c1, b.c2, b.c3], ids, blks)
        write(io, htol(Int32(nmax))); write(io, htol.(g))
    end
    for (u, part) in enumerate(Iterators.partition(sortperm(ids), 4))
        open(joinpath(dir, "new_dump$(u-1)"), "w") do io
            foreach(part) do n
                cells = [cellrec(blks[n], ii, jj, kk) for kk in 0:bs[3]-1, jj in 0:bs[2]-1, ii in 0:bs[1]-1]
                write(io, htol.(reinterpret(Float32, vec(cells))))
            end
        end
    end
    @test length(readdir(dir)) == 2 + 4

    snap = load_hamr(dir; rcyl)
    @test size(snap.rho) == size(snap.velrel) == size(snap.bfield) == (n3, n2, n1)
    @test dimnames(snap.rho) == dimnames(snap.velrel) == (:φ, :θ, :r)
    @test eltype(snap.rho) == Float32 && eltype(snap.velrel) == eltype(snap.bfield) == SVector{3,Float32}
    @test snap.t == t && snap.spin == a && snap.gam == gam
    @test snap.φ0 ≈ π/16 && snap.dφ ≈ 2π/16
    @test snap.M_unit === missing && snap.MBH === missing
    @test snap.r_prof ≈ exp.(X1.(0:n1-1))
    @test snap.th_grid ≈ [last(rθ(i, j)) for i in 0:n1-1, j in 0:n2-1]

    # ρ: finest row (θ row 1) copied as-is; coarse rows are periodic piecewise-linear "sawtooths" through the
    # coarse centers (values 1 + 100m on top of the i, j offset), hardcoded here.
    saw4 = [113.5; 38.5; 13.5:25:288.5; 263.5; 188.5]    # level 0: 4 coarse cells, s = 4
    saw2 = [176; 26:50:676; 526]                        # level 1: 8 coarse cells, s = 2
    for i in 0:n1-1
        @test snap.rho[θ=1, r=i+1] ≈ saw4 .+ i   && snap.rho[θ=2, r=i+1] ≈ saw4 .+ (i + 10)
        @test snap.rho[θ=3, r=i+1] == ρ.(i, 2, 0:15) && snap.rho[θ=4, r=i+1] == ρ.(i, 3, 0:15)
        @test snap.rho[θ=5, r=i+1] ≈ saw2 .+ (i + 40) && snap.rho[θ=6, r=i+1] ≈ saw2 .+ (i + 50)
    end

    # vectors: B^r is linear in the coarse φ cell, so coarse rows give hardcoded sawtooths as for ρ; ũ is
    # exact in the finest rows, where the Gammie reconstruction also returns the input 4-velocity; coarse rows
    # interpolate ũ between coarse centers (hardcoded fine k → (coarse m₀, weight t) pairs).
    Brow = Dict(1 => 0.5 .+ 0.1 .* (0:15),
                2 => [0.675; 0.525:0.05:1.175; 1.025],
                4 => [0.6125; 0.5375; 0.5125:0.025:0.7875; 0.7625; 0.6875])
    for i in 0:n1-1, j in 0:n2-1
        Br = Brow[2^(2 - rowlevel[j ÷ 2 + 1])]
        @test collect(snap.bfield[θ=j+1, r=i+1]) ≈ SVector.(Br, -0.1 + 0.01j, 0.2 + 0.001i) rtol = 1e-6
    end
    for i in 0:n1-1, j in 2:3, k in 0:n3-1
        ũ = snap.velrel[φ=k+1, θ=j+1, r=i+1]
        @test ũ ≈ ũks(i, j, k) rtol = 1e-5
        @test fluid_ucon(ũ, rθ(i, j)..., a) ≈ uks(i, j, k) rtol = 1e-5
    end
    for i in 0:n1-1, (j, k, m₀, w) in ((0, 2, 0, 0.125), (1, 0, 3, 0.625), (4, 3, 1, 0.25), (5, 15, 7, 0.25))
        nc = n3 ÷ 2^(2 - rowlevel[j ÷ 2 + 1])
        @test snap.velrel[φ=k+1, θ=j+1, r=i+1] ≈ (1 - w)*ũks(i, j, m₀) + w*ũks(i, j, mod(m₀ + 1, nc)) rtol = 1e-5
    end

    # radial subset across a block boundary: global cells 3..5, only rho
    sub = load_hamr(dir; rcyl, rmin = exp(X1(3)) - 1e-9, rmax = exp(X1(5)) + 1e-9, fields = (:rho,))
    @test sub.r_prof == snap.r_prof[4:6] && sub.th_grid == snap.th_grid[4:6, :]
    @test sub.rho == snap.rho[r=4:6]
    @test sub.velrel === nothing && sub.bfield === nothing
    @test load_hamr(dir; rcyl, rmin = exp(X1(6)) - 1e-9).rho == snap.rho[r=7:8]

    # auto-detection; loud failures
    @test GRMHDSnapshots.snapshot_format(dir) === :hamr
    @test load_snapshot(dir; rcyl).rho == snap.rho
    @test_throws UndefKeywordError load_hamr(dir)
    @test_throws ArgumentError load_hamr(dir; rcyl, rmin = 1e3)
    @test_throws ArgumentError load_hamr(dir; rcyl, fields = (:rho, :ug))
    @test_throws ArgumentError load_hamr(dir; rcyl, fields = ())
    snapb = load_hamr(dir; rcyl, MBH = 1.3e43)                                     # units independently optional
    @test snapb.MBH === 1.3e43 && snapb.M_unit === missing && lunit(snapb) > 0
    @test_throws "rhounit needs M_unit" rhounit(snapb)
    @test load_hamr(dir; rcyl, M_unit = 1e20).M_unit === 1e20
    bad = mktempdir()
    foreach(f -> cp(joinpath(dir, f), joinpath(bad, f)), readdir(dir))
    open(io -> write(io, zeros(UInt8, 4)), joinpath(bad, "new_dump1"), "a")        # not a whole number of blocks
    @test_throws ArgumentError load_hamr(bad; rcyl)
    rm(joinpath(bad, "new_dump1")); cp(joinpath(dir, "new_dump1"), joinpath(bad, "new_dump1"))
    open(io -> write(io, zeros(UInt8, 12)), joinpath(bad, "parameters"), "a")      # newer `parameters` layout
    @test_throws ArgumentError load_hamr(bad; rcyl)
    cp(joinpath(dir, "parameters"), joinpath(bad, "parameters"); force = true)
    open(io -> (seek(io, 100); write(io, htol(-0.9))), joinpath(bad, "parameters"), "r+")   # startx2: asymmetric X2 grid
    @test_throws ArgumentError load_hamr(bad; rcyl)
end

@testitem "H-AMR loader: real data (optional, isfile-guarded)" begin
    using StaticArrays, NamedDims
    using GRMHDSnapshots: HAMRCyl, jacobian_spatial, fluid_ucon

    dir = joinpath(@__DIR__, "data", "hamr_dumps1000")
    if isfile(joinpath(dir, "parameters"))
        snap = load_hamr(dir; rcyl = 24.32, rmax = 1.235)      # the innermost 5 radial cells, all fields
        @test size(snap.rho) == size(snap.velrel) == (1024, 1056, 5)
        @test snap.t == 5993.937771363162 && snap.spin == 0.9375 && snap.gam == 4/3
        @test snap.dφ ≈ 2π/1024 && snap.φ0 ≈ π/1024
        @test snap.r_prof[1] ≈ 1.2154603380902425 rtol = 1e-14         # gdump-stored r, θ of cell (i=0, j=22)
        @test snap.th_grid[1, 23] ≈ 0.09683481083775666 rtol = 1e-14
        @test all(>(0), snap.rho)
        @test all(v -> all(isfinite, v), snap.velrel) && all(v -> all(isfinite, v), snap.bfield)
        @test load_snapshot(dir; rcyl = 24.32, rmax = 1.221).rho == snap.rho[r=1:2]

        # Raw cells by independent byte arithmetic: block → (new_dumpN, slot) from the verified block table;
        # 9 float32 per cell (rho, ug, u^0..3, B^1..3), k fastest, then j, then i.
        blockbytes = 9*134*22*32*4
        raw(file, slot, ii, jj, kk) = open(joinpath(dir, "new_dump$file")) do io
            seek(io, slot*blockbytes + 4*9*((ii*22 + jj)*32 + kk))
            ltoh.(read!(io, Vector{Float32}(undef, 9)))
        end
        c = HAMRCyl(startx1 = 0.19325057145871735, x10 = log(24.32), x20 = -1 + 1/1056)
        X1(i) = 0.19325057145871735 + (i + 1/2)*0.003744625327713603
        X2(j) = -0.9981060606060606 + (j + 1/2)*0.0018903523875114783

        # finest-level block 500 (LEVEL 4, COORD = (0, 8, 0)) = new_dump0 slot 86: copied as-is. The loaded ũ
        # must reproduce the stored u^t (frame-independent here) and J·u^i — a check of J and the metric.
        for ii in 0:4, jj in (0, 3, 21), kk in (0, 7, 31)
            p = raw(0, 86, ii, jj, kk)
            i, j = ii, 8*22 + jj
            @test snap.rho[φ=kk+1, θ=j+1, r=i+1] == p[1]
            u = fluid_ucon(snap.velrel[φ=kk+1, θ=j+1, r=i+1], snap.r_prof[i+1], snap.th_grid[i+1, j+1], snap.spin)
            @test u[1] ≈ p[3] rtol = 1e-5
            @test u[SVector(2, 3, 4)] ≈ jacobian_spatial(c, X1(i), X2(j))*p[SVector(4, 5, 6)] rtol = 1e-5
            @test snap.bfield[φ=kk+1, θ=j+1, r=i+1] ≈ jacobian_spatial(c, X1(i), X2(j))*p[SVector(7, 8, 9)] rtol = 1e-5
        end
        # level-0 pole blocks 0 and 31 (COORD3 = 0, 1) = new_dump0 slots 0, 1: 16 fine cells per coarse cell.
        # Fine k=41 (1-based) lies 1/32 of the way from coarse cell 2 to 3; fine k=1 wraps around φ = 0,
        # between coarse cell 63 (block 31's last) and 0.
        for ii in 0:4, jj in (0, 5, 21)
            @test snap.rho[φ=41, θ=jj+1, r=ii+1] ≈ 31/32*raw(0, 0, ii, jj, 2)[1] + 1/32*raw(0, 0, ii, jj, 3)[1] rtol = 1e-6
            @test snap.rho[φ=1, θ=jj+1, r=ii+1] ≈ 15/32*raw(0, 1, ii, jj, 31)[1] + 17/32*raw(0, 0, ii, jj, 0)[1] rtol = 1e-6
        end
    else
        @test_skip "H-AMR dump dir absent"
    end
end

@testitem "H-AMR geometry & block placement vs stored gdumps (optional, isfile-guarded)" begin
    using GRMHDSnapshots: HAMRCyl, bl_coord, jacobian_spatial, _hamr_parameters, _hamr_blocks

    pdir = joinpath(@__DIR__, "data", "hamr_dumps1000"); gdir = joinpath(@__DIR__, "data", "hamr_gdumps")
    # levels 0–4, radial rings 0–11, both hemispheres
    ids = (109, 188, 390, 547, 3480, 6046, 14742, 17825, 30243, 33426)
    if isfile(joinpath(pdir, "parameters")) && all(id -> isfile(joinpath(gdir, "gdump$id")), ids)
        par = _hamr_parameters(joinpath(pdir, "parameters"))
        (; blocks, lmax) = _hamr_blocks(pdir, par)
        @test lmax == 4 && length(blocks) == 14352
        c = HAMRCyl(startx1 = par.startx[1], x10 = log(24.32), x20 = -1 + 1/1056)
        X1(i) = par.startx[1] + (i + 1/2)*par.dx[1]; X2(j) = par.startx[2] + (j + 1/2)*par.dx[2]
        nc = 9*134*22*32
        for id in ids
            # valid prefix: 9 doubles per cell (i, j, k, X1, X2, X3, r, θ, φ), k fastest; then 49 per (i, j),
            # j fastest (gcov, gcon, gdet, dxdxp — each 4×4 row-major)
            a = ltoh.(reinterpret(Float64, read(joinpath(gdir, "gdump$id"), 8*(nc + 49*134*22))))
            cells = reshape(a[1:nc], 9, 32, 22, 134); met = reshape(a[nc+1:end], 49, 22, 134)
            b = only(filter(b -> b.id == id, blocks)); s = 2^(lmax - b.level)
            # placement: H-AMR's own global indices and φ centers match the block's grid coordinates
            @test cells[1, :, :, :] == [b.c1*134 + ii for kk in 1:32, jj in 1:22, ii in 0:133]
            @test cells[2, :, :, :] == [b.c2*22 + jj for kk in 1:32, jj in 0:21, ii in 1:134]
            @test cells[9, :, :, :] ≈ [(b.c3*32 + kk + 1/2)*s*2π/1024 for kk in 0:31, jj in 1:22, ii in 1:134] rtol = 1e-14
            rt = [bl_coord(c, X1(b.c1*134 + ii), X2(b.c2*22 + jj)) for jj in 0:21, ii in 0:133]
            J  = [jacobian_spatial(c, X1(b.c1*134 + ii), X2(b.c2*22 + jj)) for jj in 0:21, ii in 0:133]
            @test cells[7, 1, :, :] ≈ first.(rt) rtol = 1e-12
            @test maximum(abs, cells[8, 1, :, :] .- last.(rt)) < 1e-12
            # stored dxdxp is finite-differenced (noise ~1e-8): rows r, θ; columns X1, X2
            @test met[39, :, :] ≈ getindex.(J, 1, 1) rtol = 1e-7
            @test maximum(abs, met[43, :, :] .- getindex.(J, 2, 1)) < 1e-7
            @test maximum(abs, met[44, :, :] .- getindex.(J, 2, 2)) < 1e-7
        end
    else
        @test_skip "H-AMR gdump fixtures absent"
    end
end

@testitem "_" begin
    import Aqua
    Aqua.test_all(GRMHDSnapshots; ambiguities=false)
    Aqua.test_ambiguities(GRMHDSnapshots)

    import CompatHelperLocal as CHL
    CHL.@check()
end
