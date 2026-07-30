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
    @test snap isa KoralSnapshot
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
    using Adapt, AxisKeys, Zarr

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
    using StaticArrays, Zarr, AxisKeys
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
    @test map_plasma((ps, ρ, α) -> ps.bsq, snap; threaded = true) ==
          map_plasma((ps, ρ, α) -> ps.bsq, snap; threaded = false)
    @inferred comoving_bsq(snap)

    # lapse(snap) is φ-invariant and equals the scalar lapse at each (r,θ), exactly.
    @test lap == [lapse(snap.r_prof[i], snap.th_grid[i, j], a) for k in 1:nφ, j in 1:nθ, i in 1:nr]

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

@testitem "_" begin
    import Aqua
    Aqua.test_all(GRMHDSnapshots; ambiguities=false)
    Aqua.test_ambiguities(GRMHDSnapshots)

    import CompatHelperLocal as CHL
    CHL.@check()
end
