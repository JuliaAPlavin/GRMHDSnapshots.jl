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
    @test size(snap.fluid) == (24, 32, 48)
    @test parent(snap.fluid) isa Array{@NamedTuple{ρ::Float32, ũ::SVector{3,Float32}, B::SVector{3,Float32}}, 3}
    @test r_min(snap) < r_max(snap)
    @test (r_min(snap), r_max(snap)) == (snap.grid.r_prof[1], snap.grid.r_prof[end])
    @test all(>(0), diff(snap.grid.r_prof))

    (; ρ, ũ, B) = rec5 = @inferred sample(snap, 5.0, 0.0, 0.0)
    @test rec5 isa @NamedTuple{ρ::Float64, ũ::SVector{3,Float64}, B::SVector{3,Float64}}
    @test ρ isa Float64 && isfinite(ρ)
    @test ũ isa SVector{3,Float64} && all(isfinite, ũ)
    @test B isa SVector{3,Float64} && all(isfinite, B)

    i, j, k = 20, 16, 12
    rn = snap.grid.r_prof[i]; θn = snap.grid.th_grid[i, j]; φn = snap.grid.φ0 + (k - 1) * snap.grid.dφ
    xn = rn * sin(θn) * cos(φn); yn = rn * sin(θn) * sin(φn); zn = rn * cos(θn)
    @test sample(snap, xn, yn, zn).ρ ≈ snap.fluid[φ=k, θ=j, r=i].ρ
    @test sample(snap, xn, yn, zn).ũ ≈ snap.fluid[φ=k, θ=j, r=i].ũ rtol = 1e-11
    @test sample(snap, xn, yn, zn).B ≈ snap.fluid[φ=k, θ=j, r=i].B rtol = 1e-11

    rec0 = @inferred sample(snap, 0.0, 0.0, 10 * r_max(snap))
    @test rec0 === (ρ = 0.0, ũ = zero(SVector{3,Float64}), B = zero(SVector{3,Float64}))

    # partial loads: the records hold the requested fields only, always in the order (ρ, ũ, B); each is the same data
    using NamedDims
    for (fields, K) in (((:ρ,), (:ρ,)), ((:ũ,), (:ũ,)), ((:B,), (:B,)), ((:B, :ũ), (:ũ, :B)), ((:B, :ρ), (:ρ, :B)),
                        ((:ũ, :ρ), (:ρ, :ũ)), ((:B, :ũ, :ρ), (:ρ, :ũ, :B)))
        sp = load_koral(store, "snap002000"; fields)
        @test fieldnames(eltype(sp.fluid)) == K
        @test parent(sp.fluid) isa Array && dimnames(sp.fluid) == (:φ, :θ, :r)
        @test sp.fluid == map(c -> c[K], snap.fluid)
    end
    @test_throws ArgumentError load_koral(store, "snap002000"; fields = ())
    @test_throws ArgumentError load_koral(store, "snap002000"; fields = (:rho,))

    # sample on full and partial snapshots: each field interpolated independently of the others, so a partial record
    # is bitwise the matching part of the full one; out-of-grid zeros have the in-grid type
    # φ = 0 is across the φ wrap (φ0 > 0); y = ±0 straddles atan's ±π; z-axis points are at the θ edges
    pts_in = [(5.0, 0.0, 0.0), (-7.0, -1e-9, 0.3), (-7.0, 1e-9, 0.3), (0.0, 0.0, 6.0), (0.0, 0.0, -6.0), (1e-6, 0.0, 8.0), (xn, yn, zn), (3.0, -4.0, 2.0)]
    pts_out = [(0.0, 0.0, 10 * r_max(snap)), (0.1 * r_min(snap), 0.0, 0.0)]
    RT = @NamedTuple{ρ::Float64, ũ::SVector{3,Float64}, B::SVector{3,Float64}}
    @test all(p -> @inferred(sample(snap, p...)) isa RT, [pts_in; pts_out])
    @test all(p -> sample(snap, p...) === (ρ = 0.0, ũ = zero(SVector{3,Float64}), B = zero(SVector{3,Float64})), pts_out)
    snap_rho = load_koral(store, "snap002000"; fields = (:ρ,))
    snap_vel = load_koral(store, "snap002000"; fields = (:ũ,))
    snap_vb = load_koral(store, "snap002000"; fields = (:ũ, :B))
    for p in [pts_in; pts_out]
        full = sample(snap, p...)
        @test @inferred(sample(snap_rho, p...)) === (; full.ρ)
        @test @inferred(sample(snap_vel, p...)) === (; full.ũ)
        @test @inferred(sample(snap_vb, p...)) === (; full.ũ, full.B)
    end
    @test sample(snap_vel, 5.0, 0.0, 0.0).ũ == ũ
end

@testitem "extension-sniffing (one-arg zarr snap path)" begin
    using Zarr

    store_path = joinpath(@__DIR__, "data", "koral_sample.zarr")
    snap1 = load_koral(joinpath(store_path, "snap002000"))
    snap2 = load_koral(zopen(store_path, "r"), "snap002000")
    @test snap1.fluid == snap2.fluid
end

@testitem "Adapt identity move preserves axis names & values" begin
    using Adapt, NamedDims, Zarr, StaticArrays

    snap = load_koral(zopen(joinpath(@__DIR__, "data", "koral_sample.zarr"), "r"), "snap002000")
    snap2 = Adapt.adapt_structure(Array, snap)
    @test dimnames(snap2.fluid) == dimnames(snap.fluid) == (:φ, :θ, :r)
    @test snap2.fluid == snap.fluid
    @test eltype(snap2.fluid) == eltype(snap.fluid)
    @test snap2.grid.r_extent === snap.grid.r_extent
    @test snap2.R === snap.R === Val(one(SMatrix{3,3,Int}))
end

@testitem "raw .h5 cross-check (optional, isfile-guarded)" begin
    using HDF5, StaticArrays, Zarr
    using GRMHDSnapshots: sample

    h5path = joinpath(@__DIR__, "data", "ipole_a0.9_2000.h5")
    if isfile(h5path)
        s = load_koral(h5path)
        @test s.M_unit === nothing && s.MBH === nothing        # a non-radiative run: scale-free
        @test r_min(s) < r_max(s)
        @test (r_min(s), r_max(s)) == (s.grid.r_prof[1], s.grid.r_prof[end])
        @test all(>(0), diff(s.grid.r_prof))
        @test eltype(s.fluid) == @NamedTuple{ρ::Float64, ũ::SVector{3,Float64}, B::SVector{3,Float64}}
        (; ρ, ũ, B) = @inferred sample(s, 5.0, 0.0, 0.0)
        @test isfinite(ρ) && all(isfinite, ũ) && all(isfinite, B)

        # the grid from slices equals that from the full arrays; a ρ-only load is the full load's ρ on the same grid
        h5open(h5path, "r") do h5
            r = read(h5["grid_out/r"]); th = read(h5["grid_out/th"]); ph = read(h5["grid_out/ph"])
            n3, n2, n1 = size(r)
            @test s.grid.r_prof == Float64[r[1, 1, i] for i in 1:n1]
            @test s.grid.th_grid == Float64[th[1, j, i] for i in 1:n1, j in 1:n2]
            @test s.grid.φ0 == Float64(ph[1, 1, 1]) + π
            @test s.grid.dφ == (Float64(ph[n3, 1, 1]) - Float64(ph[1, 1, 1]))/(n3 - 1)
        end
        sρ = load_koral(h5path; fields = (:ρ,))
        @test sρ.fluid == map(c -> c[(:ρ,)], s.fluid)
        @test (sρ.grid.r_prof, sρ.grid.th_grid, sρ.grid.φ0, sρ.grid.dφ) == (s.grid.r_prof, s.grid.th_grid, s.grid.φ0, s.grid.dφ)

        s_store = load_koral(zopen(joinpath(@__DIR__, "data", "koral_sample.zarr"), "r"), "snap002000")
        sl = (1:6:144, 1:6:192, 1:6:288)
        scale_relerr(loaded, hpath) = h5open(h5path, "r") do h5
            b = read(h5[hpath])[sl...]
            maximum(abs.(Float64.(collect(loaded)) .- b)) / maximum(abs.(b))
        end
        re_rho = scale_relerr(map(c -> c.ρ, s_store.fluid), "quants/rho")
        re_B1 = scale_relerr(map(c -> c.B[1], s_store.fluid), "quants/B1")
        @test 0.010 < re_rho < 0.020
        @test 0.004 < re_B1 < 0.012
    else
        @test_skip "raw .h5 fixture absent"
    end
end

@testitem "koral header units: radiative runs only, M_unit a density" begin
    using HDF5, StaticArrays, Zarr

    # the header's M_unit is KORAL's rhoGU2CGS(1), a density: for a radiative run rhounit is it, and bunit the header's
    # own B_unit; keywords override each unit independently
    M_hdr, Mbh_hdr, B_hdr = 0.014610581643228402, 6.5e9, 1.2845747778405998e10
    function checkunits(ld)
        s = ld(; has_radiation = 1)
        @test s.MBH === Mbh_hdr
        @test rhounit(s) ≈ M_hdr rtol = 1e-12
        @test bunit(s) ≈ B_hdr rtol = 1e-6
        # KORAL's mass unit is M_bh; they differ by the header's rounded constants. In M☉ it fits Float32.
        @test s.M_unit ≈ 6.5e9 rtol = 1e-3
        @test isfinite(Float32(s.M_unit))
        @test ld(; has_radiation = 1, MBH = 4e6).MBH === 4e6
        @test ld(; has_radiation = 1, MBH = 4e6).M_unit === s.M_unit
        @test ld(; has_radiation = 1, M_unit = 5e-8).M_unit === 5e-8
        @test ld(; has_radiation = 1, M_unit = 5e-8).MBH === Mbh_hdr
        foreach((0, nothing)) do flag
            s = ld(; has_radiation = flag)
            @test s.M_unit === nothing && s.MBH === nothing
            sk = ld(; has_radiation = flag, MBH = 6.5e9, M_unit = 5e-8)
            @test sk.MBH === 6.5e9 && sk.M_unit === 5e-8
        end
    end

    # Zarr: copies of the fixture (has_radiation = 0) with the flag set, and removed
    src = joinpath(@__DIR__, "data", "koral_sample.zarr")
    zattrs = read(joinpath(src, ".zattrs"), String)
    @test occursin("\"has_radiation\":0,", zattrs)
    zcopies = Dict(map((1, nothing)) do flag
        dst = joinpath(mktempdir(), "run.zarr")
        cp(src, dst)
        write(joinpath(dst, ".zattrs"), replace(zattrs, "\"has_radiation\":0," => isnothing(flag) ? "" : "\"has_radiation\":$flag,"))
        flag => dst
    end)
    zpath(flag) = flag == 0 ? src : zcopies[flag]
    checkunits((; has_radiation, kw...) -> load_koral(zopen(zpath(has_radiation), "r"), "snap002000"; fields = (:ρ,), kw...))
    @test load_koral(joinpath(zpath(1), "snap002000"); MBH = 4e6).MBH === 4e6

    # synthetic koral .h5: datasets indexed (φ, θ, r), r depending on r only, θ on (θ, r), φ on φ, uniformly spaced
    n3, n2, n1 = 4, 3, 5
    rs = [2.0 + i^1.5 for i in 1:n1]
    ths = [0.4 + 0.7j + 0.01i for i in 1:n1, j in 1:n2]
    phs = [-π + (k - 1)*2π/n3 for k in 1:n3]
    grid = (r = [rs[i] for k in 1:n3, j in 1:n2, i in 1:n1], th = [ths[i, j] for k in 1:n3, j in 1:n2, i in 1:n1],
            ph = [phs[k] for k in 1:n3, j in 1:n2, i in 1:n1])
    q(c) = [c + 0.001k + 0.01j + 0.1i for k in 1:n3, j in 1:n2, i in 1:n1]
    quants = (rho = q(1.0), U1 = q(0.1), U2 = q(0.2), U3 = q(0.3), B1 = q(-1.0), B2 = q(-2.0), B3 = q(-3.0))
    dir = mktempdir()
    function h5file(flag)
        path = joinpath(dir, "koral_$flag.h5")
        h5open(path, "w") do h5
            foreach(k -> write(h5, "quants/$k", quants[k]), keys(quants))
            foreach(k -> write(h5, "grid_out/$k", grid[k]), keys(grid))
            write(h5, "header/bhspin", 0.9); write(h5, "header/gam", 13/9); write(h5, "t", 2000.0)
            write(h5, "header/units/M_unit", M_hdr); write(h5, "header/units/M_bh", Mbh_hdr)
            isnothing(flag) || write(h5, "header/has_radiation", Int32(flag))
        end
        path
    end
    h5paths = Dict(map(f -> f => h5file(f), (1, 0, nothing)))
    checkunits((; has_radiation, kw...) -> load_koral(h5paths[has_radiation]; kw...))
    checkunits((; has_radiation, kw...) -> load_snapshot(h5paths[has_radiation]; kw...))

    s = load_koral(h5paths[0])
    @test size(s.fluid) == (n3, n2, n1)
    @test s.spin == 0.9 && s.gam == 13/9 && s.t == 2000.0
    @test s.grid.r_prof == rs && s.grid.th_grid == ths
    @test s.grid.φ0 ≈ phs[1] + π && s.grid.dφ ≈ 2π/n3
    k, j, i = 3, 2, 4
    @test s.fluid[φ=k, θ=j, r=i].ρ == quants.rho[k, j, i]
    @test s.fluid[φ=k, θ=j, r=i].ũ == SVector(quants.U1[k, j, i], quants.U2[k, j, i], quants.U3[k, j, i])
    @test s.fluid[φ=k, θ=j, r=i].B == SVector(quants.B1[k, j, i], quants.B2[k, j, i], quants.B3[k, j, i])

    # partial loads: the requested fields only, in the order (ρ, ũ, B), each the same data; the grid is the same
    using NamedDims
    for (fields, K) in (((:ρ,), (:ρ,)), ((:ũ,), (:ũ,)), ((:B, :ρ), (:ρ, :B)), ((:ũ, :B), (:ũ, :B)))
        sp = load_koral(h5paths[0]; fields)
        @test fieldnames(eltype(sp.fluid)) == K
        @test parent(sp.fluid) isa Array && dimnames(sp.fluid) == (:φ, :θ, :r)
        @test sp.fluid == map(c -> c[K], s.fluid)
        @test (sp.grid.r_prof, sp.grid.th_grid, sp.grid.φ0, sp.grid.dφ) == (s.grid.r_prof, s.grid.th_grid, s.grid.φ0, s.grid.dφ)
    end
    @test_throws ArgumentError load_koral(h5paths[0]; fields = ())
    @test_throws ArgumentError load_koral(h5paths[0]; fields = (:rho,))
    sρ = load_snapshot(h5paths[0]; fields = (:ρ,))
    @test sρ.fluid == map(c -> c[(:ρ,)], s.fluid) && sρ.grid.th_grid == ths
end

@testitem "plasma physics" begin
    using StaticArrays, Zarr, NamedDims, Accessors
    using LinearAlgebra: det
    using GRMHDSnapshots: plasma_state, ks_gcov, spatial_norm

    store = zopen(joinpath(@__DIR__, "data", "koral_sample.zarr"), "r")
    # the fixture run is non-radiative, so scale-free: units given explicitly
    snap = load_koral(store, "snap002000"; MBH = 6.5e9, M_unit = 5e-8)
    a = snap.spin

    # Physical invariants of the reconstructed plasma state at sampled cells:
    # u·u = -1 (timelike, normalized) and b·u = 0 (comoving field ⊥ 4-velocity), bsq ≥ 0.
    for (i, j, k) in ((20, 16, 12), (10, 8, 3), (40, 24, 20), (5, 30, 1))
        r = snap.grid.r_prof[i]; θ = snap.grid.th_grid[i, j]
        (; ũ, B) = snap.fluid[φ=k, θ=j, r=i]
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
    @test size(bsqg) == size(snap.fluid)
    @test all(≥(0), bsqg)
    @test all(isfinite, comoving_B_gauss(snap))
    lap = lapse(snap)
    @test size(lap) == size(snap.fluid) && all(α -> 0 < α ≤ 1, lap)
    σ = magnetization(snap)
    @test all(isfinite, σ[map(c -> c.ρ > 0, snap.fluid)])
    @test σ == comoving_bsq(snap) ./ map(c -> c.ρ, snap.fluid)

    # comoving_bsq grid value equals the kernel on that cell's primitives.
    i, j, k = 20, 16, 12
    @test bsqg[φ=k, θ=j, r=i] ≈ plasma_state(snap.fluid[φ=k, θ=j, r=i].ũ, snap.fluid[φ=k, θ=j, r=i].B,
                                             snap.grid.r_prof[i], snap.grid.th_grid[i, j], a).bsq

    # map_plasma hoists the (r,θ)-only metric across φ and threads over r: over the whole grid it must
    # match a naive per-cell reconstruction (independent reference) up to FMA reassociation, and its
    # result must be bit-identical regardless of thread count.
    nφ, nθ, nr = size(snap.fluid)
    bsq_ref = [plasma_state(snap.fluid[φ=k, θ=j, r=i].ũ, snap.fluid[φ=k, θ=j, r=i].B,
                            snap.grid.r_prof[i], snap.grid.th_grid[i, j], a).bsq for k in 1:nφ, j in 1:nθ, i in 1:nr]
    @test bsqg ≈ bsq_ref rtol = 1e-13
    @test map_plasma((ps, _, _) -> ps.bsq, snap; threaded = true) ==
          map_plasma((ps, _, _) -> ps.bsq, snap; threaded = false)
    @inferred comoving_bsq(snap)

    # map_plasma's kernel gets the cell's record
    @test map_plasma((_, _, c) -> c.ρ, snap) == map(c -> c.ρ, snap.fluid)

    # comoving_bsq/comoving_B_gauss reconstruct b from ũ and B only — no ρ needed.
    snap_bv = load_koral(store, "snap002000"; fields = (:ũ, :B), MBH = 6.5e9, M_unit = 5e-8)
    @test !hasfield(eltype(snap_bv.fluid), :ρ)
    @test comoving_bsq(snap_bv) == bsqg
    @test comoving_B_gauss(snap_bv) == comoving_B_gauss(snap)
    # the single-field grid maps need only their own field
    snap_u = load_koral(store, "snap002000"; fields = (:ũ,)); snap_b = load_koral(store, "snap002000"; fields = (:B,))
    @test flow_speed(snap_u) == flow_speed(snap) && proper_velocity(snap_u) == proper_velocity(snap)
    @test fluid_velocity(snap_u) == fluid_velocity(snap)
    @test bfield_magnitude(snap_b) == bfield_magnitude(snap)
    @test lapse(snap_b) == lapse(snap)

    # lapse(snap) equals the scalar lapse at each cell's (r,θ).
    @test lap ≈ [lapse(snap.grid.r_prof[i], snap.grid.th_grid[i, j], a) for k in 1:nφ, j in 1:nθ, i in 1:nr] rtol = 1e-12

    # grav_redshift(snap) equals the scalar grav_redshift at each cell's (r,θ).
    zg = grav_redshift(snap)
    @test size(zg) == size(snap.fluid) && all(x -> 0 ≤ x ≤ 1, zg)
    @test zg ≈ [grav_redshift(snap.grid.r_prof[i], snap.grid.th_grid[i, j], a) for k in 1:nφ, j in 1:nθ, i in 1:nr] rtol = 1e-12

    # volume_ratio(snap) equals the scalar at each cell's (r,θ), and is ≥ 1 everywhere.
    vr = volume_ratio(snap)
    @test size(vr) == size(snap.fluid) && all(≥(1), vr)
    @test vr ≈ [volume_ratio(snap.grid.r_prof[i], snap.grid.th_grid[i, j], a) for k in 1:nφ, j in 1:nθ, i in 1:nr] rtol = 1e-12

    # the vector-field maps and map_plasma's record equal the per-cell scalar functions at each cell's (r, θ)
    percell(f) = [f(snap.fluid[φ=k, θ=j, r=i], snap.grid.r_prof[i], snap.grid.th_grid[i, j]) for k in 1:nφ, j in 1:nθ, i in 1:nr]
    @test flow_speed(snap) ≈ percell((c, r, θ) -> flow_speed(c.ũ, r, θ, a)) rtol = 1e-13
    @test proper_velocity(snap) ≈ percell((c, r, θ) -> proper_velocity(c.ũ, r, θ, a)) rtol = 1e-13
    @test fluid_velocity(snap) ≈ percell((c, r, θ) -> fluid_velocity(c.ũ, r, θ, a)) rtol = 1e-13
    @test bfield_magnitude(snap) ≈ percell((c, r, θ) -> spatial_norm(c.B, r, θ, a)) rtol = 1e-13
    @test map_plasma((_, _, c) -> c.ũ, snap) ≈ map(c -> c.ũ, snap.fluid) rtol = 1e-13
    @test map_plasma((_, _, c) -> c.B, snap) ≈ map(c -> c.B, snap.fluid) rtol = 1e-13

    # Unit scalars are plain CGS Float64. The fixture's header units (M_bh = 6.5e9) are placeholders of a non-radiative
    # run: without keywords it has none.
    snap0 = load_koral(store, "snap002000")
    @test snap0.M_unit === nothing && snap0.MBH === nothing
    @test_throws "lunit needs MBH (M☉)" lunit(snap0)
    unitsof(s) = map(f -> f(s), [lunit, rhounit, bunit])
    @test snap.MBH === 6.5e9 && snap.M_unit === 5e-8
    @test unitsof(snap) isa Vector{Float64}
    @test lunit(snap) ≈ 9.598062747325812e14
    Msun_g = 1.98841e33                                         # M☉ in g, independent of the package's Unitful path
    @test rhounit(snap) ≈ 5e-8*Msun_g/lunit(snap)^3 rtol = 1e-4
    @test bunit(snap) ≈ 2.99792458e10*sqrt(4π*rhounit(snap))
    # Float32 scalars (as after a whole-snapshot Float32 conversion) still give Float64 units, also where r_g³ in cm³
    # or a large M_unit in grams would overflow Float32.
    snap32 = setproperties(snap, (; grid = setproperties(snap.grid, map(Float32, (; snap.grid.φ0, snap.grid.dφ))),
                                    map(Float32, (; snap.spin, snap.gam, snap.t, snap.M_unit, snap.MBH))...))
    @test snap32.MBH isa Float32 && snap32.M_unit isa Float32
    @test unitsof(snap32) isa Vector{Float64}
    @test unitsof(snap32) ≈ unitsof(snap) rtol = 1e-6
    @test unitsof(@set snap32.M_unit = 1f10) ≈ unitsof(@set snap.M_unit = 1e10) rtol = 1e-6

    # Kernel type stability on Float64 inputs.
    @inferred plasma_state(SVector(0.1, 0.2, 0.3), SVector(0.4, 0.5, 0.6), 5.0, 1.0, a)
end

@testitem "spin frame turned relative to the grid" begin
    using StaticArrays, Zarr, NamedDims, Accessors
    using LinearAlgebra: norm, dot, normalize, det
    using GRMHDSnapshots: sample, plasma_state, spatial_norm, _apply_stencil, _sample_stencil

    # rotations R (grid Cartesian → spin frame): Rx takes grid −x to +z; a cyclic permutation; a half turn about z
    Rx = SMatrix{3,3,Int}(0, 0, -1, 0, 1, 0, 1, 0, 0)
    Ms = (Rx, SMatrix{3,3,Int}(0, 1, 0, 0, 0, 1, 1, 0, 0), SMatrix{3,3,Int}(-1, 0, 0, 0, -1, 0, 0, 0, 1))
    I3 = one(SMatrix{3,3,Int})
    @test all(M -> M'*M == I3 && det(M) == 1, Ms)

    # independent reference: spherical basis by trig, Cartesian vector of (V^r, V^θ, V^φ) at p, with φ given on the z axis
    function cart(V, p, φ = atan(p[2], p[1]))
        r = norm(p); θ = acos(clamp(p[3]/r, -1, 1))
        V[1]*SVector(sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)) + V[2]*r*SVector(cos(θ)*cos(φ), cos(θ)*sin(φ), -sin(θ)) +
            V[3]*r*sin(θ)*SVector(-sin(φ), cos(φ), 0)
    end
    # the interpolated grid record at grid point q, unconverted
    raw(snap, q) = _apply_stencil(_sample_stencil(snap.grid, size(snap.fluid, :φ), q...), snap.fluid)

    koral = load_koral(zopen(joinpath(@__DIR__, "data", "koral_sample.zarr"), "r"), "snap002000")
    # each snapshot with in-grid radii; the H-AMR one with spin along grid z, so R is the identity
    dir = joinpath(@__DIR__, "data", "hamr_dumps10000")
    hamr = if isfile(joinpath(dir, "parameters")) && isfile(joinpath(@__DIR__, "data", "gdumps", "grid"))
        ((load_hamr(dir; rcyl = nothing, spin_axis = (0, 0, 1), rmin = 4, rmax = 4.2), (4.07, 4.13, 4.19)),)
    else
        @test_skip "H-AMR new-layout dump dir or gdumps absent"
        ()
    end
    snaps = ((koral, (5.0, 12.0, 40.0)), hamr...)
    dirs = normalize.([SVector(0.3, -0.5, 0.8), SVector(-0.7, 0.2, -0.4), SVector(0.1, 0.9, 0.35), SVector(-0.6, -0.6, 0.5)])

    for (snap, rs) in snaps, M in Ms
        snapR = @set snap.R = Val(M)
        a = snap.spin
        RT = typeof(sample(snap, (rs[1]*dirs[1])...))

        # a. covariance off both axes: same ρ, and the vectors are the grid ones turned by M
        for r in rs, d in dirs
            p = r*d; q = M'*p
            s = @inferred sample(snapR, p...); s0 = sample(snap, q...)
            @test s isa RT
            @test s.ρ ≈ s0.ρ rtol = 1e-14
            @test cart(s.ũ, p) ≈ M*cart(s0.ũ, q) rtol = 1e-12
            @test cart(s.B, p) ≈ M*cart(s0.B, q) rtol = 1e-12
        end
        @test sample(snapR, 0.0, 0.0, 10*r_max(snap)) === sample(snapR, 0.1*r_min(snap), 0.0, 0.0) ===
              sample(snapR, 0.0, 0.0, 0.0) === sample(snap, 0.0, 0.0, 10*r_max(snap))

        # on the grid axis, off the spin axis: covariant against the grid basis at the lookup's φ = atan(q[2], q[1]),
        # 0 or π by the sign of zero in q[1] (both occur among these points)
        zc = M[:, 3]
        if !iszero(zc[1]^2 + zc[2]^2)
            onq = vec([SVector(ntuple(i -> iszero(zc[i]) ? sg[i]*0.0 : z*zc[i], 3))
                       for sg in Iterators.product((1, -1), (1, -1), (1, -1)), z in (rs[2], -rs[2])])
            @test any(p -> signbit((M'*p)[1]), onq) && any(p -> !signbit((M'*p)[1]), onq)
            for p in onq
                q = M'*p
                s = sample(snapR, p...); w = raw(snap, q)
                @test all(isfinite, s.ũ) && all(isfinite, s.B)
                @test cart(s.ũ, p) ≈ M*cart(w.ũ, q) rtol = 1e-12
                @test cart(s.B, p) ≈ M*cart(w.B, q) rtol = 1e-12
            end
        end

        # c. grid maps at nodes against the scalar functions of the sampled spin-frame record at its colatitude θ′; at a node
        # the interpolation weights are 0/1 up to rounding (of acos near the poles)
        # nodes at φ < π: past it the lookup's atan wraps by 2π, while the grid's period nφ·dφ is 2π only to the precision
        # of the stored φ. k = 1 is a node of the stencil across the wrap (k = nφ, 1).
        nφ, nθ, nr = size(snap.fluid)
        km = findlast(k -> snap.grid.φ0 + (k - 1)*snap.grid.dφ < π, 1:nφ)
        ir = (2, nr - 1, nr ÷ 2)
        cells = [(1, 1, ir[1]), (km, nθ, ir[2]), (km ÷ 2, 2, ir[3]), (1, nθ ÷ 2, ir[1]), (km, 1, ir[3]), (3, nθ - 1, ir[2])]
        pts = map(cells) do (k, j, i)
            r = snap.grid.r_prof[i]; θ = snap.grid.th_grid[i, j]; φ = snap.grid.φ0 + (k - 1)*snap.grid.dφ
            p = M*(r*SVector(sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)))
            (; r, θ′ = acos(p[3]/norm(p)), s = sample(snapR, p...))
        end
        at(A) = map(((k, j, i),) -> A[φ=k, θ=j, r=i], cells)
        ps(x) = plasma_state(x.s.ũ, x.s.B, x.r, x.θ′, a)
        mp = map_plasma((ps, g, c) -> (; ps.u, ps.b, g, c), snapR)
        for (grid, scalar) in ((flow_speed(snapR), x -> flow_speed(x.s.ũ, x.r, x.θ′, a)),
                               (proper_velocity(snapR), x -> proper_velocity(x.s.ũ, x.r, x.θ′, a)),
                               (fluid_velocity(snapR), x -> fluid_velocity(x.s.ũ, x.r, x.θ′, a)),
                               (bfield_magnitude(snapR), x -> spatial_norm(x.s.B, x.r, x.θ′, a)),
                               (comoving_bsq(snapR), x -> ps(x).bsq),
                               (map(m -> m.u, mp), x -> ps(x).u), (map(m -> m.b, mp), x -> ps(x).b),
                               (map(m -> m.g, mp), x -> grav_redshift(x.r, x.θ′, a)),
                               (map(m -> m.c.ρ, mp), x -> x.s.ρ), (map(m -> m.c.ũ, mp), x -> x.s.ũ), (map(m -> m.c.B, mp), x -> x.s.B),
                               (lapse(snapR), x -> lapse(x.r, x.θ′, a)), (grav_redshift(snapR), x -> grav_redshift(x.r, x.θ′, a)),
                               (volume_ratio(snapR), x -> volume_ratio(x.r, x.θ′, a)))
            @test at(grid) ≈ map(scalar, pts) rtol = 1e-11
        end
    end

    # b. on the spin axis: θ′ along φ′ = atan(p[2], p[1]), 0 or π by the sign of zero in p[1], so ê_θ′ = (±1, 0, 0); the φ′
    # component is 0
    for (snap, rs) in snaps, T in (Ms..., I3), z in (rs[2], -rs[2]), x0 in (0.0, -0.0)
        sR = @set snap.R = Val(T)
        p = SVector(x0, 0.0, z); q = T'*p
        s = sample(sR, p...); w = raw(snap, q)
        eθ = SVector(sign(z)*copysign(1, x0), 0, 0)
        for (V, W) in ((s.ũ, w.ũ), (s.B, w.B))
            @test all(isfinite, V)
            @test V[3] == 0
            @test V[1] == W[1]
            @test V[2] ≈ dot(T*cart(W, q), eθ)/abs(z) rtol = 1e-12
        end
    end

    # d. maps on partial snapshots with a turned R need only the fields they read; the same results as the full one
    store = zopen(joinpath(@__DIR__, "data", "koral_sample.zarr"), "r")
    part(fields) = @set load_koral(store, "snap002000"; fields).R = Val(Rx)
    full = @set koral.R = Val(Rx)
    sρ, sũ, sB = part((:ρ,)), part((:ũ,)), part((:B,))
    @test lapse(sρ) == lapse(sB) == lapse(full)
    @test grav_redshift(sρ) == grav_redshift(sũ) == grav_redshift(full)
    @test volume_ratio(sρ) == volume_ratio(sB) == volume_ratio(full)
    @test flow_speed(sũ) == flow_speed(full) && proper_velocity(sũ) == proper_velocity(full)
    @test fluid_velocity(sũ) == fluid_velocity(full)
    @test bfield_magnitude(sB) == bfield_magnitude(full)
    @test comoving_bsq(part((:ũ, :B))) == comoving_bsq(full)
    foreach(f -> @inferred(f(full)), (flow_speed, proper_velocity, fluid_velocity, bfield_magnitude, comoving_bsq,
                                      lapse, grav_redshift, volume_ratio))
    @inferred lapse(sρ)
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
    using GRMHDSnapshots: flow_speed, proper_velocity, spatial_norm, lapse, fluid_ucon

    @test flow_speed(SVector(0.0, 0.0, 0.0), 5.0, 1.0, 0.9) == 0
    @test proper_velocity(SVector(0.0, 0.0, 0.0), 5.0, 1.0, 0.9) == 0

    # spatial_norm = √(γᵢⱼVⁱVʲ). Hardcoded against the KS 3-metric written out by hand at one cell:
    # r=2, θ=π/2, a=0.9 ⇒ tr=2·2/4=1, ρ²=4; g_rr=1+tr=2, g_θθ=ρ²=4, g_φφ=ρ²+a²(1+tr)=4+1.62=5.62,
    # g_rφ=-a(1+tr)=-1.8. For V=(1,1,1): |V|²=2+4+5.62+2·(-1.8)=8.02.
    @test spatial_norm(SVector(1.0, 1.0, 1.0), 2.0, π/2, 0.9) ≈ sqrt(8.02)
    # a raw Euclidean norm would give √3 ≈ 1.732 — materially different, i.e. the fix matters.
    @test !isapprox(spatial_norm(SVector(1.0, 1.0, 1.0), 2.0, π/2, 0.9), sqrt(3); rtol = 0.05)
    # ũ as the velocity at that cell: βγ = |ũ| = √8.02, γ² = 9.02
    @test proper_velocity(SVector(1.0, 1.0, 1.0), 2.0, π/2, 0.9) ≈ sqrt(8.02)
    @test flow_speed(SVector(1.0, 1.0, 1.0), 2.0, π/2, 0.9) ≈ sqrt(8.02/9.02)

    # independent route: γ = α·uᵗ from the 4-velocity reconstruction
    for a in (0.0, 0.9), r in (1.3, 3.0, 30.0), θ in (0.3, π/2),
        ũ in (SVector(2.0, -0.05, 1.0), SVector(-1.0, 0.2, 3.0), SVector(0.3, 0.1, -0.2))
        γ = lapse(r, θ, a)*fluid_ucon(ũ, r, θ, a)[1]
        @test proper_velocity(ũ, r, θ, a) ≈ sqrt(γ^2 - 1)
        @test flow_speed(ũ, r, θ, a) ≈ sqrt(1 - 1/γ^2)
        @test 0 ≤ flow_speed(ũ, r, θ, a) < 1
    end

    # slow flow in Float32: γ ≈ 1, so √(γ²−1) would cancel to 0
    @test proper_velocity(SVector{3,Float32}(1f-5, 0, 0), 10f0, 1.2f0, 0.9f0) ≈ 1.0953480973390266e-5 rtol=1e-6
    @test flow_speed(SVector{3,Float32}(1f-5, 0, 0), 10f0, 1.2f0, 0.9f0) ≈ 1.0953480972733173e-5 rtol=1e-6
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
    @test size(snap.fluid) == (n3, n2, n1)
    @test eltype(snap.fluid) == @NamedTuple{ρ::Float64, ũ::SVector{3,Float64}, B::SVector{3,Float64}}
    sp = load_iharm(path; fields = (:B, :ρ))
    @test eltype(sp.fluid) == @NamedTuple{ρ::Float64, B::SVector{3,Float64}} && sp.fluid == map(c -> (; c.ρ, c.B), snap.fluid)
    @test_throws ArgumentError load_iharm(path; fields = ())
    @test_throws ArgumentError load_iharm(path; fields = (:velrel,))
    @test snap.spin == par.a && snap.gam ≈ 1.444444 && snap.t == 123.0
    @test snap.M_unit === nothing && snap.MBH === nothing         # scale-free by default
    @test snap.grid.φ0 ≈ sx3 + 0.5dx3 && snap.grid.dφ ≈ dx3                 # φ = X3, no +π offset

    # units are independently optional: lunit needs MBH only, rhounit/bunit both; errors name the missing ones.
    @test_throws "lunit needs MBH (M☉)" lunit(snap)
    foreach(f -> @test_throws("$f needs M_unit (M☉) and MBH (M☉)", f(snap)), (rhounit, bunit))
    snapb = load_iharm(path; MBH = 6.5e9)
    @test snapb.M_unit === nothing
    @test lunit(snapb) ≈ 9.598062747325812e14
    foreach(f -> @test_throws("$f needs M_unit (M☉)", f(snapb)), (rhounit, bunit))
    snapm = load_iharm(path; M_unit = 5e-8)
    @test snapm.MBH === nothing
    foreach(f -> @test_throws("$f needs MBH (M☉)", f(snapm)), (lunit, rhounit, bunit))
    # setting the other unit later (as downstream viewers do for M_unit) rebuilds the snapshot with its new type
    snapbm = @set snapb.M_unit = 5e-8
    @test snapbm.M_unit === 5e-8 && snapbm.fluid == snap.fluid
    Msun_g = 1.98841e33                                         # M☉ in g, independent of the package's Unitful path
    @test isapprox(rhounit(snapbm), 5e-8*Msun_g/lunit(snapb)^3; rtol = 1e-4) && bunit(snapbm) ≈ 2.99792458e10*sqrt(4π*rhounit(snapbm))
    @test rhounit(@set snapm.MBH = 6.5e9) == rhounit(snapbm)
    @test (@set snapbm.M_unit = nothing).M_unit === nothing

    @test (r_min(snap), r_max(snap)) == (snap.grid.r_prof[1], snap.grid.r_prof[end])
    c = FMKS(startx1 = sx1, hslope = par.hslope, mks_smooth = par.mks_smooth, poly_xt = par.poly_xt, poly_alpha = par.poly_alpha)
    for i in 1:n1, j in 1:n2, k in 1:n3
        X1 = sx1 + (i-0.5)*dx1; X2 = sx2 + (j-0.5)*dx2
        r, th = bl_coord(c, X1, X2); J = jacobian_spatial(c, X1, X2)
        @test snap.grid.r_prof[i] ≈ r
        @test snap.grid.th_grid[i, j] ≈ th
        @test snap.fluid[φ=k, θ=j, r=i].ρ ≈ prims[1, k, j, i]
        @test snap.fluid[φ=k, θ=j, r=i].ũ ≈ J*SVector(prims[3, k, j, i], prims[4, k, j, i], prims[5, k, j, i])
        @test snap.fluid[φ=k, θ=j, r=i].B ≈ J*SVector(prims[6, k, j, i], prims[7, k, j, i], prims[8, k, j, i])
        u = plasma_state(snap.fluid[φ=k, θ=j, r=i].ũ, snap.fluid[φ=k, θ=j, r=i].B, r, th, par.a).u
        @test u' * ks_gcov(r, th, par.a) * u ≈ -1 rtol = 1e-10   # valid 4-velocity after transform
    end
end

@testitem "iharm FMKS loader: real data + pyharm golden (optional, isfile-guarded)" begin
    using StaticArrays, NamedDims
    using GRMHDSnapshots: plasma_state, ks_gcov

    path = joinpath(@__DIR__, "data", "iharm_fmks_a0.h5")
    if isfile(path)
        snap = load_iharm(path)
        @test size(snap.fluid) == (128, 128, 288)
        @test r_min(snap) < r_max(snap) && all(>(0), diff(snap.grid.r_prof))
        @test snap.spin == 0.0

        # lazy units: scale-free by default ⇒ throw; supplied ⇒ positive scalars.
        @test_throws ArgumentError lunit(snap)
        snapu = load_iharm(path; M_unit = 5e-8, MBH = 6.5e9)
        @test lunit(snapu) > 0 && rhounit(snapu) > 0 && bunit(snapu) > 0
        # auto-detect routes an FMKS .h5 to load_iharm.
        @test load_snapshot(path).fluid == snap.fluid

        # Golden values from pyharm (see gen_fmks_golden.py), an independent reference reader.
        # 0-based native (i=radial, j=θ, k=φ); bsq/sigma/Gamma are frame-invariant ⇒ compared directly.
        gold = [(59, 63, 9,    6.56053574719, 1.56595075338, 0.68453335762,  0.00538246981389,  0.0078629766599,  1.1482527967),
                (119, 39, 69,  24.5568056735, 1.29190538653, 0.162381529808, 0.000236250350958, 0.00145490901112, 1.02877584453),
                (29, 99, 4,    3.39096277761, 2.02063373719, 0.0109128253534, 0.382428746929,   35.0439720735,    1.02475536928),
                (199, 109, 119, 142.719055604, 2.37849777729, 0.0111680263653, 9.7388717037e-6, 0.000872031582404, 1.00266551412)]
        for (i, j, k, rg, thg, rhog, bsqg, sigg, Gamg) in gold
            I, J, K = i+1, j+1, k+1
            r = snap.grid.r_prof[I]; th = snap.grid.th_grid[I, J]; a = snap.spin
            rho = snap.fluid[φ=K, θ=J, r=I].ρ
            ps = plasma_state(snap.fluid[φ=K, θ=J, r=I].ũ, snap.fluid[φ=K, θ=J, r=I].B, r, th, a)
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

@testitem "H-AMR spin axis → rotation" begin
    using StaticArrays, LinearAlgebra
    using GRMHDSnapshots: _hamr_rotation, _rotation

    ẑ = SVector(0, 0, 1)
    for a in (SVector(1, 0, 0), SVector(-1, 0, 0), SVector(0, 1, 0), SVector(0, -1, 0), ẑ)
        R = _rotation(_hamr_rotation(Tuple(a)))
        @test _hamr_rotation(a) === _hamr_rotation(Tuple(a))
        @test R isa SMatrix{3,3,Int}
        @test R*a == ẑ && R'R == I && det(R) == 1
        # smallest rotation: about a × ẑ, which it fixes
        a == ẑ ? (@test R == I) : (@test R*cross(a, ẑ) == cross(a, ẑ))
    end
    @test _hamr_rotation((-1, 0, 0)) === Val(SMatrix{3,3,Int}(0, 0, -1, 0, 1, 0, 1, 0, 0))
    @test_throws "no unique smallest rotation" _hamr_rotation((0, 0, -1))
    @test_throws ArgumentError _hamr_rotation((1, 1, 0))
    @test_throws ArgumentError _hamr_rotation((0, 0, 2))
    @test_throws UndefKeywordError load_hamr(joinpath(@__DIR__, "data", "hamr_dumps10000"); rcyl = nothing)
end

@testitem "H-AMR loader: real data (optional, isfile-guarded)" begin
    using StaticArrays, NamedDims
    using GRMHDSnapshots: HAMRCyl, jacobian_spatial, fluid_ucon

    dir = joinpath(@__DIR__, "data", "hamr_dumps1000")
    if isfile(joinpath(dir, "parameters"))
        snap = load_hamr(dir; rcyl = 24.32, spin_axis = (0, 0, 1), rmax = 1.235)      # the innermost 5 radial cells, all fields
        @test size(snap.fluid) == (1024, 1056, 5)
        @test snap.t == 5993.937771363162 && snap.spin == 0.9375 && snap.gam == 4/3
        @test snap.grid.dφ ≈ 2π/1024 && snap.grid.φ0 ≈ π/1024
        @test snap.grid.r_prof[1] ≈ 1.2154603380902425 rtol = 1e-14         # gdump-stored r, θ of cell (i=0, j=22)
        @test snap.grid.th_grid[1, 23] ≈ 0.09683481083775666 rtol = 1e-14
        @test all(c -> c.ρ > 0, snap.fluid)
        @test all(c -> all(isfinite, c.ũ) && all(isfinite, c.B), snap.fluid)
        @test load_snapshot(dir; rcyl = 24.32, spin_axis = (0, 0, 1), rmax = 1.221).fluid == snap.fluid[r=1:2]
        # partial: the requested fields in the order (ρ, ũ, B), the same data
        sp = load_hamr(dir; rcyl = 24.32, spin_axis = (0, 0, 1), rmax = 1.235, fields = (:B, :ρ))
        @test eltype(sp.fluid) == @NamedTuple{ρ::Float32, B::SVector{3,Float32}}
        @test sp.fluid == map(c -> (; c.ρ, c.B), snap.fluid)

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
            @test snap.fluid[φ=kk+1, θ=j+1, r=i+1].ρ == p[1]
            u = fluid_ucon(snap.fluid[φ=kk+1, θ=j+1, r=i+1].ũ, snap.grid.r_prof[i+1], snap.grid.th_grid[i+1, j+1], snap.spin)
            @test u[1] ≈ p[3] rtol = 1e-5
            @test u[SVector(2, 3, 4)] ≈ jacobian_spatial(c, X1(i), X2(j))*p[SVector(4, 5, 6)] rtol = 1e-5
            @test snap.fluid[φ=kk+1, θ=j+1, r=i+1].B ≈ jacobian_spatial(c, X1(i), X2(j))*p[SVector(7, 8, 9)] rtol = 1e-5
        end
        # level-0 pole blocks 0 and 31 (COORD3 = 0, 1) = new_dump0 slots 0, 1: each coarse cell m is copied into the
        # 16 fine cells it covers, 16m+1 … 16m+16 (1-based).
        for ii in 0:4, jj in (0, 5, 21), (slot, kk) in ((0, 0), (0, 2), (1, 31))
            p = raw(0, slot, ii, jj, kk); m = 32*slot + kk
            @test map(c -> c.ρ, snap.fluid[φ=16m+1:16m+16, θ=jj+1, r=ii+1]) == fill(p[1], 16)
            @test map(c -> c.B, snap.fluid[φ=16m+1:16m+16, θ=jj+1, r=ii+1]) == fill(snap.fluid[φ=16m+1, θ=jj+1, r=ii+1].B, 16)
            @test snap.fluid[φ=16m+1, θ=jj+1, r=ii+1].B ≈ jacobian_spatial(c, X1(ii), X2(jj))*p[SVector(7, 8, 9)] rtol = 1e-5
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
        blocks = _hamr_blocks(pdir, par)
        lmax = maximum(b -> b.level[3], blocks)
        @test lmax == 4 && length(blocks) == 14352 && all(b -> b.level[1] == b.level[2] == 0, blocks)
        c = HAMRCyl(startx1 = par.startx[1], x10 = log(24.32), x20 = -1 + 1/1056)
        X1(i) = par.startx[1] + (i + 1/2)*par.dx[1]; X2(j) = par.startx[2] + (j + 1/2)*par.dx[2]
        nc = 9*134*22*32
        for id in ids
            # valid prefix: 9 doubles per cell (i, j, k, X1, X2, X3, r, θ, φ), k fastest; then 49 per (i, j),
            # j fastest (gcov, gcon, gdet, dxdxp — each 4×4 row-major)
            a = ltoh.(reinterpret(Float64, read(joinpath(gdir, "gdump$id"), 8*(nc + 49*134*22))))
            cells = reshape(a[1:nc], 9, 32, 22, 134); met = reshape(a[nc+1:end], 49, 22, 134)
            b = only(filter(b -> b.id == id, blocks)); s = 2^(lmax - b.level[3])
            # placement: H-AMR's own global indices and φ centers match the block's grid coordinates
            @test cells[1, :, :, :] == [b.coord[1]*134 + ii for kk in 1:32, jj in 1:22, ii in 0:133]
            @test cells[2, :, :, :] == [b.coord[2]*22 + jj for kk in 1:32, jj in 0:21, ii in 1:134]
            @test cells[9, :, :, :] ≈ [(b.coord[3]*32 + kk + 1/2)*s*2π/1024 for kk in 0:31, jj in 1:22, ii in 1:134] rtol = 1e-14
            rt = [bl_coord(c, X1(b.coord[1]*134 + ii), X2(b.coord[2]*22 + jj)) for jj in 0:21, ii in 0:133]
            J  = [jacobian_spatial(c, X1(b.coord[1]*134 + ii), X2(b.coord[2]*22 + jj)) for jj in 0:21, ii in 0:133]
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

@testitem "H-AMR loader: new layout, real data (optional, isfile-guarded)" begin
    using StaticArrays, NamedDims, LinearAlgebra, ForwardDiff
    using GRMHDSnapshots: ks_gcov, snapshot_format

    # dumps10000 with the run's grid in the sibling `gdumps/` (toolkit layout)
    dir = joinpath(@__DIR__, "data", "hamr_dumps10000")
    if isfile(joinpath(dir, "parameters")) && isfile(joinpath(@__DIR__, "data", "gdumps", "grid"))
        # the run's spin lies along grid −x
        ld(; kw...) = load_hamr(dir; rcyl = nothing, spin_axis = (-1, 0, 0), kw...)
        # r ∈ [4, 4.2]: blocks at levels (r, θ, φ) = (1, 1, 1), (1, 1, 0), (0, 0, 1) ⇒ output at level 1 in each direction
        snap = ld(rmin = 4, rmax = 4.2)
        @test snap.t == 250000.0048901291 && snap.spin == 0.9375
        @test snap.gam ≈ 13/9 rtol = 1e-15
        @test size(snap.fluid) == (192, 192, 3)
        @test dimnames(snap.fluid) == (:φ, :θ, :r)
        @test parent(snap.fluid) isa Array{@NamedTuple{ρ::Float32, ũ::SVector{3,Float32}, B::SVector{3,Float32}}, 3}
        @test snap.M_unit === nothing && snap.MBH === nothing
        R = SMatrix{3,3,Int}(0, 0, -1, 0, 1, 0, 1, 0, 0)
        @test snap.R === Val(R)

        # level-1 spacings: half the level-0 ones log(Rout/Rin)/420, 2·0.98958…/96, 2π/96
        h1 = log(1e6/1.2131867454091887)/840; h2 = 2*0.9895833333333334/192
        X1(i) = 0.19325057145871735 + (i + 1/2)*h1; X2(j) = -0.9895833333333334 + (j + 1/2)*h2; X3(k) = (k + 1/2)*2π/192
        @test snap.grid.r_prof ≈ [4.060943340300723, 4.127336454415088, 4.194815041837602] rtol = 1e-14   # level-1 cells 74–76
        @test (r_min(snap), r_max(snap)) == (snap.grid.r_prof[1], snap.grid.r_prof[end])
        @test snap.grid.th_grid ≈ [π/2*(1 + X2(j)) for i in 74:76, j in 0:191] rtol = 1e-14
        @test snap.grid.th_grid[1, 1] ≈ 0.024458471451287742 rtol = 1e-14
        @test snap.grid.φ0 ≈ π/192 && snap.grid.dφ ≈ 2π/192

        # Raw cells by independent byte arithmetic, slot 0 of a new_dump file (block → file from the `parameters`
        # triples, levels/coords from the grid records): 9 float32 per cell, φ fastest, then θ, then r.
        raw(file, ii, jj, kk) = open(joinpath(dir, "new_dump$file")) do io
            seek(io, 4*9*((ii*16 + jj)*24 + kk))
            ltoh.(read!(io, Vector{Float32}(undef, 9)))
        end
        J(i, j) = Diagonal(SVector(exp(X1(i)), π/2, 1))
        # (t, r, θ, φ) → the same point's (t, r, θ, φ) in the frame turned by M
        turn(M, x) = let c = M*(x[2]*SVector(sin(x[3])*cos(x[4]), sin(x[3])*sin(x[4]), cos(x[3])))
            SVector(x[1], x[2], acos(c[3]/x[2]), atan(c[2], c[1]))
        end
        # relative velocity in grid components via the full inverse metric: the KS one at the spin-frame point, turned to
        # grid components; ũ^i = u^i − u^t g^{ti}/g^{tt}. (i, j) the stored cell's level-1 indices, φ its center.
        ũ(i, j, φ, p) = let u = SVector(p[3], (J(i, j)*p[SVector(4, 5, 6)])...),
                y = turn(R, SVector(0, exp(X1(i)), π/2*(1 + X2(j)), φ)),
                Λ = ForwardDiff.jacobian(y -> turn(R', y), y),          # ∂(grid)/∂(spin)
                gc = Λ*inv(ks_gcov(y[2], y[3], 0.9375))*Λ'
            u[SVector(2, 3, 4)] - u[1]*gc[1, SVector(2, 3, 4)]/gc[1, 1]
        end

        # block 25969 = new_dump6 slot 0, levels (1, 1, 1), COORD (2, 1, 1): its level-1 cells are copied as-is
        for ii in 4:6, jj in (0, 7, 15), kk in (0, 11, 23)
            p = raw(6, ii, jj, kk); i, j, k = 70 + ii, 16 + jj, 24 + kk
            @test snap.fluid[φ=k+1, θ=j+1, r=i-73].ρ == p[1]
            @test snap.fluid[φ=k+1, θ=j+1, r=i-73].ũ ≈ ũ(i, j, X3(k), p) rtol = 1e-6
            @test snap.fluid[φ=k+1, θ=j+1, r=i-73].B ≈ J(i, j)*p[SVector(7, 8, 9)] rtol = 1e-6
        end
        # block 24989 = new_dump8 slot 0, levels (1, 1, 0), COORD (2, 0, 0): each φ cell copied into 2 output cells
        for ii in 4:6, jj in (0, 9), kk in (0, 23)
            p = raw(8, ii, jj, kk)
            @test map(c -> c.ρ, snap.fluid[φ=2kk+1:2kk+2, θ=jj+1, r=ii-3]) == [p[1], p[1]]
            @test snap.fluid[φ=2kk+2, θ=jj+1, r=ii-3].B ≈ J(70 + ii, jj)*p[SVector(7, 8, 9)] rtol = 1e-6
            @test snap.fluid[φ=2kk+2, θ=jj+1, r=ii-3].ũ ≈ ũ(70 + ii, jj, (kk + 1/2)*2π/96, p) rtol = 1e-6   # at its level-0 φ
        end

        # the spin direction: u⁰ rebuilt from the loaded ũ in the spin-frame metric reproduces the stored u⁰ at block 25969's
        # copied cells (θ rows 16–31), to ~3e-3 since H-AMR's values are not exactly at the cell centers; with the spin
        # along another axis it does not
        cells = [(ii, jj, kk) for ii in 4:6, jj in 0:15, kk in 0:23]
        u0dev(s) = let u0 = map_plasma((ps, _, _) -> ps.u[1], s)
            maximum(((ii, jj, kk),) -> abs(u0[φ=25+kk, θ=17+jj, r=ii-3]/raw(6, ii, jj, kk)[3] - 1), cells)
        end
        @test u0dev(snap) < 1e-2
        @test all(a -> u0dev(load_hamr(dir; rcyl = nothing, spin_axis = a, rmin = 4, rmax = 4.2)) > 1e-2, ((0, 0, 1), (1, 0, 0)))

        # gas angular momentum L = Σ ρ x × v dV along the spin: x, v spin-frame Cartesian, dV = r² sinθ dr dθ dφ ∝ r³ sinθ
        # (dr ∝ r, dθ and dφ uniform)
        sL = ld(rmin = 5, rmax = 15, maxlevel = (r = 0, θ = 0, φ = 0))
        v = fluid_velocity(sL)
        L = sum(CartesianIndices(parent(sL.fluid))) do I
            k, j, i = Tuple(I)
            r = sL.grid.r_prof[i]; θ = sL.grid.th_grid[i, j]; φ = sL.grid.φ0 + (k - 1)*sL.grid.dφ
            x = R*(r*SVector(sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)))
            θ′ = acos(x[3]/r); φ′ = atan(x[2], x[1])
            vr, vθ, vφ = v[φ=k, θ=j, r=i]
            vx = vr*SVector(sin(θ′)*cos(φ′), sin(θ′)*sin(φ′), cos(θ′)) + vθ*r*SVector(cos(θ′)*cos(φ′), cos(θ′)*sin(φ′), -sin(θ′)) +
                 vφ*r*sin(θ′)*SVector(-sin(φ′), cos(φ′), 0)
            sL.fluid[φ=k, θ=j, r=i].ρ*cross(x, vx)*r^3*sin(θ)
        end
        @test acosd(L[3]/norm(L)) < 2

        # caps: an output cell is the mean of the stored cells it covers; the other directions stay at level 1
        sφ = ld(rmin = 4, rmax = 4.2, maxlevel = (φ = 0,))
        @test size(sφ.fluid) == (96, 192, 3) && sφ.grid.φ0 ≈ π/96 && sφ.grid.dφ ≈ 2π/96
        @test sφ.grid.r_prof == snap.grid.r_prof && sφ.grid.th_grid == snap.grid.th_grid
        for ii in 4:6, jj in (0, 15), m in (0, 11)
            p, q = raw(6, ii, jj, 2m), raw(6, ii, jj, 2m + 1); i, j = 70 + ii, 16 + jj
            @test sφ.fluid[φ=12+m+1, θ=j+1, r=i-73].ρ ≈ (p[1] + q[1])/2 rtol = 1e-6
            @test sφ.fluid[φ=12+m+1, θ=j+1, r=i-73].ũ ≈ (ũ(i, j, X3(24 + 2m), p) + ũ(i, j, X3(25 + 2m), q))/2 rtol = 1e-6
            @test sφ.fluid[φ=12+m+1, θ=j+1, r=i-73].B ≈ J(i, j)*(p[SVector(7, 8, 9)] + q[SVector(7, 8, 9)])/2 rtol = 1e-6
        end
        for ii in 4:6, jj in (0, 9), kk in (0, 23)     # block 24989 is already at φ level 0: copied
            @test sφ.fluid[φ=kk+1, θ=jj+1, r=ii-3].ρ == raw(8, ii, jj, kk)[1]
        end
        # partial, averaged: the requested fields in the order (ρ, ũ, B), the same data
        sφp = ld(rmin = 4, rmax = 4.2, maxlevel = (φ = 0,), fields = (:B, :ũ))
        @test eltype(sφp.fluid) == @NamedTuple{ũ::SVector{3,Float32}, B::SVector{3,Float32}}
        @test sφp.fluid == map(c -> (; c.ũ, c.B), sφ.fluid)
        @test_throws ArgumentError ld(rmin = 4, rmax = 4.2, fields = ())

        # level-0 radial cell 37 = level-1 cells 74, 75 = block 25969's ii = 4, 5; each stored cell converted at its own r
        sr = ld(rmin = 4, rmax = 4.2, maxlevel = (r = 0, θ = nothing))
        @test size(sr.fluid) == (192, 192, 1) && sr.grid.dφ == snap.grid.dφ
        @test sr.grid.r_prof ≈ [4.09400531115158] rtol = 1e-14
        @test sr.grid.th_grid ≈ snap.grid.th_grid[1:1, :] rtol = 1e-14
        for jj in (0, 15), kk in (0, 23)
            p, q = raw(6, 4, jj, kk), raw(6, 5, jj, kk); j, k = 16 + jj, 24 + kk
            @test sr.fluid[φ=k+1, θ=j+1, r=1].ρ ≈ (p[1] + q[1])/2 rtol = 1e-6
            @test sr.fluid[φ=k+1, θ=j+1, r=1].ũ ≈ (ũ(74, j, X3(k), p) + ũ(75, j, X3(k), q))/2 rtol = 1e-6
            @test sr.fluid[φ=k+1, θ=j+1, r=1].B ≈ (J(74, j)*p[SVector(7, 8, 9)] + J(75, j)*q[SVector(7, 8, 9)])/2 rtol = 1e-6
        end

        # level-0 θ cell m = level-1 cells 2m, 2m+1 = block 25969's jj = 2m-16, 2m-15
        sθ = ld(rmin = 4, rmax = 4.2, maxlevel = (θ = 0,), fields = (:ρ,))
        @test size(sθ.fluid) == (192, 96, 3) && eltype(sθ.fluid) == @NamedTuple{ρ::Float32}
        @test sθ.grid.r_prof == snap.grid.r_prof && sθ.grid.dφ == snap.grid.dφ
        @test sθ.grid.th_grid ≈ [π/2*(1 - 0.9895833333333334 + (j + 1/2)*2h2) for i in 1:3, j in 0:95] rtol = 1e-14
        for ii in 4:6, m in (8, 15), kk in (0, 23)
            @test sθ.fluid[φ=24+kk+1, θ=m+1, r=ii-3].ρ ≈ (raw(6, ii, 2m-16, kk)[1] + raw(6, ii, 2m-15, kk)[1])/2 rtol = 1e-6
        end

        # the output level comes from the blocks whose radial extent overlaps the range: below the edge r = 3.7751… of the
        # ring with levels up to (1, 1, 1), the inner ring's (0, 0, 1); rmax exactly at the edge already raises it
        @test size(ld(rmin = 3.6, rmax = 3.77, fields = (:ρ,)).fluid) == (192, 96, 1)
        @test size(ld(rmin = 3.6, rmax = 3.7751454194494656, fields = (:ρ,)).fluid) == (192, 192, 3)
        @test size(ld(rmin = 3.6, rmax = 3.81, fields = (:ρ,)).fluid) == (192, 192, 4)

        # auto-detection without any grid argument; units independently optional
        @test snapshot_format(dir) === :hamr
        @test load_snapshot(dir; rcyl = nothing, spin_axis = (-1, 0, 0), rmin = 4, rmax = 4.2, fields = (:ρ,)).fluid == map(c -> (; c.ρ), snap.fluid)
        snapb = ld(rmin = 4, rmax = 4.1, fields = (:ρ,), MBH = 6.5e9)
        @test snapb.MBH === 6.5e9 && snapb.M_unit === nothing && lunit(snapb) ≈ 9.598062747325812e14
    else
        @test_skip "H-AMR new-layout dump dir or gdumps absent"
    end
end

@testitem "BlockGrid: nearest-cell lookup on a synthetic AMR grid" begin
    using StaticArrays, NamedDims, Adapt
    using GRMHDSnapshots: sample, HAMRSph, HAMRCyl, bl_coord, native_coord, _block_table, _grid_sample, _tospin,
        _sample_frame, r_extent
    using GRMHDSnapshots.IntervalSets: Interval

    # native_coord inverts bl_coord; no point lookup through cylindrified θ
    for X1 in (log(2.0), 1.0, 4.5), X2 in (-0.99, -0.3, 0.0, 0.7)
        @test all(isapprox.(native_coord(HAMRSph(), bl_coord(HAMRSph(), X1, X2)...), (X1, X2); atol = 1e-14))
    end
    @test native_coord(HAMRSph(), 2f0, 1f0) isa Tuple{Float32,Float32}
    cyl = HAMRCyl(startx1 = 0.19325057145871735, x10 = log(24.32), x20 = -1 + 1/1056)
    @test_throws "resample" native_coord(cyl, 5.0, 1.0)

    # 2×2×2 level-0 blocks of 4×3×5 cells (r, θ, φ); blocks (1, 0, 1) and (0, 1, 0) replaced by their children at levels
    # (1, 1, 0) and (1, 1, 2)
    nb = (2, 2, 2); bs = (4, 3, 5)
    lv(t...) = SVector{3,Int32}(t)
    base = [lv(a, b, c) for a in 0:1 for b in 0:1 for c in 0:1 if (a, b, c) ∉ ((1, 0, 1), (0, 1, 0))]
    kids1 = [lv(2 + a, b, 1) for a in 0:1 for b in 0:1]
    kids2 = [lv(a, 2 + b, c) for a in 0:1 for b in 0:1 for c in 0:3]
    coord = [base; kids1; kids2]
    level = [fill(lv(0, 0, 0), length(base)); fill(lv(1, 1, 0), 4); fill(lv(1, 1, 2), 16)]
    nblk = length(coord)
    startx = SVector(log(2.0), -0.9, 0.3); ext0 = SVector((log(100.0) - log(2.0))/2, 0.9, π)
    g = BlockGrid(HAMRSph(), level, coord, nb, startx, ext0, Interval(2.0, 100.0))
    # records encoding (block, cell), cells 0-based
    rec(b, i, j, k) = (ρ = 1000.0b + 100i + 10j + k, ũ = SVector(0.01b, 0.02i - 0.03, 0.01j + 0.001k),
                       B = SVector(0.1k, -0.05b, 0.2i*j + 0.25))
    fluid = NamedDimsArray([rec(b, i, j, k) for k in 0:bs[3]-1, j in 0:bs[2]-1, i in 0:bs[1]-1, b in 1:nblk], (:φ, :θ, :r, :block))
    I3 = Val(one(SMatrix{3,3,Int}))
    snap = GRMHDSnapshot(fluid, g, I3, 0.9, 4/3, 0.0, nothing, nothing)
    @test r_extent(g) == Interval(2.0, 100.0) && (r_min(snap), r_max(snap)) == (2.0, 100.0)

    # the table: every slot owned by exactly the block whose box holds it
    @test g.lmax === lv(1, 1, 2)
    @test size(g.table) == (4, 4, 8) && eltype(g.table) == Int32 && _block_table(level, coord, nb) == g.table
    owners(s) = findall(b -> all(d -> coord[b][d] == s[d] >> (g.lmax[d] - level[b][d]), 1:3), eachindex(coord))
    @test all(I -> owners(Tuple(I) .- 1) == [g.table[I]], CartesianIndices(g.table))

    # independent reference geometry: block box, cell size, native → grid Cartesian
    h(b) = ext0 ./ (SVector(bs) .* 2 .^ level[b])
    lo(b) = startx .+ coord[b] .* ext0 ./ 2 .^ level[b]
    cart(X) = (r = exp(X[1]); θ = π/2*(1 + X[2]); r*SVector(sin(θ)*cos(X[3]), sin(θ)*sin(X[3]), cos(θ)))
    # the cell holding native X (X3 wrapped), by brute force over the blocks
    function ref(X)
        Xw = SVector(X[1], X[2], startx[3] + mod(X[3] - startx[3], 2π))
        b = only(filter(b -> all(lo(b) .≤ Xw .< lo(b) .+ ext0 ./ 2 .^ level[b]), 1:nblk))
        (b, Int.(floor.((Xw .- lo(b)) ./ h(b)))...)
    end
    # with R = I the vectors only pass through `_tospin`
    expect(q, b, i, j, k) = map(c -> _tospin(c, _sample_frame(one(SMatrix{3,3,Int}), q, q)), fluid[k+1, j+1, i+1, b])

    # every cell centre, and points jittered inside the cell, give that cell; φ > π among them (atan's negative branch)
    cells = [(b, i, j, k) for b in 1:nblk for i in 0:bs[1]-1 for j in 0:bs[2]-1 for k in 0:bs[3]-1]
    offs = [zero(SVector{3}); vec([SVector(a, c, d) for a in (-0.45, 0.45), c in (-0.45, 0.45), d in (-0.45, 0.45)])]
    centre(b, i, j, k) = lo(b) .+ (SVector(i, j, k) .+ 0.5) .* h(b)
    at(b, i, j, k, o) = cart(centre(b, i, j, k) .+ o .* h(b))
    for (b, i, j, k) in cells
        @test all(o -> (q = at(b, i, j, k, o); sample(snap, q...) == expect(q, b, i, j, k)), offs)
    end
    (b, i, j, k) = cells[100]
    @test _grid_sample(g, fluid, at(b, i, j, k, zero(SVector{3}))) === fluid[k+1, j+1, i+1, b]
    @test any(c -> centre(c...)[3] > π, cells)
    @test all(c -> ref(centre(c...)) == c, cells)

    # φ periodicity: φ ± 2π is the same cell
    for (b, i, j, k) in cells[1:7:end]
        @test all(s -> sample(snap, cart(centre(b, i, j, k) + SVector(0, 0, s))...).ρ == fluid[k+1, j+1, i+1, b].ρ, (-2π, 2π))
    end

    # beyond the θ range (near both poles, and on the axis): the first/last θ cell
    for r in (3.0, 10.0, 60.0), φ in (1.0, 4.0, -2.5), (θ, X2) in ((0.05, -0.9 + 1e-9), (π - 0.05, 0.9 - 1e-9))
        q = r*SVector(sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ))
        @test sample(snap, q...) == expect(q, ref(SVector(log(r), X2, φ))...)
    end
    for r in (3.0, 60.0), z in (r, -r)
        q = SVector(0.0, 0.0, z)
        @test sample(snap, q...) == expect(q, ref(SVector(log(r), z > 0 ? -0.9 + 1e-9 : 0.9 - 1e-9, 0.0))...)
    end

    # outside r_extent: zeros
    Z = (ρ = 0.0, ũ = zero(SVector{3,Float64}), B = zero(SVector{3,Float64}))
    @test all(p -> sample(snap, p...) === Z, [(1.5, 0.0, 0.0), (0.0, 0.0, -1.9), (150.0, 1.0, 2.0), (0.0, 0.0, 0.0)])

    # type-stable and allocation-free
    q = at(cells[500]..., zero(SVector{3}))
    @inferred sample(snap, q...)
    @inferred sample(snap, 0.0, 0.0, 1000.0)
    alloc(s, q) = @allocated sample(s, q...)
    alloc(snap, q)
    @test alloc(snap, q) == 0

    # all-Float32: computed in Float32, no promotion
    names = (:φ, :θ, :r, :block)
    fluid32 = NamedDimsArray(map(c -> map(x -> Float32.(x), c), parent(fluid)), names)
    g32 = BlockGrid(HAMRSph(), level, coord, nb, Float32.(startx), Float32.(ext0), Interval(2f0, 100f0))
    snap32 = GRMHDSnapshot(fluid32, g32, I3, 0.9f0, Float32(4/3), 0f0, nothing, nothing)
    RT32 = @NamedTuple{ρ::Float32, ũ::SVector{3,Float32}, B::SVector{3,Float32}}
    @test @inferred(sample(snap32, Float32.(q)...)) isa RT32
    @test @inferred(sample(snap32, 0f0, 0f0, 1f3)) isa RT32
    @test !occursin("Float64", string(code_typed(_grid_sample, Tuple{typeof(g32), typeof(fluid32), SVector{3,Float32}})))
    alloc(snap32, Float32.(q))
    @test alloc(snap32, Float32.(q)) == 0
    for (b, i, j, k) in cells
        @test sample(snap32, Float32.(at(b, i, j, k, zero(SVector{3})))...).ρ == fluid[k+1, j+1, i+1, b].ρ
    end

    # Adapt round trip
    sa = Adapt.adapt(Array, snap)
    @test sa.grid.table == g.table && sa.grid.lmax === g.lmax && sa.grid.level == level && sa.grid.coord == coord
    @test sa.grid.startx === startx && sa.grid.ext0 === ext0 && sa.grid.r_extent === g.r_extent && sa.grid.coords === g.coords
    @test parent(sa.fluid) == parent(fluid) && dimnames(sa.fluid) == names
    @test sample(sa, q...) === sample(snap, q...)

    # a cylindrified BlockGrid has no point lookup
    @test_throws "resample" sample(GRMHDSnapshot(fluid, BlockGrid(cyl, level, coord, nb, startx, ext0, Interval(2.0, 100.0)),
                                                 I3, 0.9, 4/3, 0.0, nothing, nothing), q...)
end

@testitem "BlockGrid: grid maps equal the TensorGrid ones at one level; sample zeros typed as inside" begin
    using StaticArrays, NamedDims
    using GRMHDSnapshots: sample, HAMRSph, bl_coord
    using GRMHDSnapshots.IntervalSets: Interval

    # a level-0 grid of 2×2×2 blocks of 4×3×5 cells as one tensor grid, centres by the same expression as the blocks'
    nb = (2, 2, 2); bs = (4, 3, 5); n = nb .* bs
    startx = SVector(log(2.0), -0.9, 0.3); ext0 = SVector((log(100.0) - log(2.0))/2, 0.9, π)
    cen(d, i) = startx[d] + (2i + 1)*(ext0[d]/(2bs[d]))
    r_prof = map(i -> first(bl_coord(HAMRSph(), cen(1, i), cen(2, 0))), 0:n[1]-1)
    th_grid = [last(bl_coord(HAMRSph(), cen(1, i), cen(2, j))) for i in 0:n[1]-1, j in 0:n[2]-1]
    tg = TensorGrid(r_prof, th_grid, cen(3, 0), ext0[3]/bs[3])
    T = [(ρ = 1.0 + i + 0.1j + 0.01k, ũ = SVector(0.1sin(i + j), 0.01cos(k), 0.003j),
          B = SVector(1.0 + 0.1k, 0.02sin(i + k), 0.01j)) for k in 0:n[3]-1, j in 0:n[2]-1, i in 0:n[1]-1]
    coord = vec([SVector{3,Int32}(a, b, c) for a in 0:1, b in 0:1, c in 0:1])
    level = fill(zero(SVector{3,Int32}), length(coord))
    blk(A, c) = A[c[3]*bs[3] .+ (1:bs[3]), c[2]*bs[2] .+ (1:bs[2]), c[1]*bs[1] .+ (1:bs[1])]
    toblocks(A) = stack(c -> blk(A, c), coord)
    bg = BlockGrid(HAMRSph(), level, coord, nb, startx, ext0, tg.r_extent)
    tnames = (:φ, :θ, :r); bnames = (:φ, :θ, :r, :block)
    mk(A, grid, R = Val(one(SMatrix{3,3,Int}))) =
        GRMHDSnapshot(NamedDimsArray(A, ndims(A) == 3 ? tnames : bnames), grid, R, 0.9, 4/3, 0.0, nothing, nothing)

    # r and θ of every cell are bit-identical; φ = φ0 + (k−1)·dφ from each block's own φ0 differs from the tensor one by
    # round-off, so maps depending on φ (any of them with a turned R, the vector ones always) agree to round-off only
    Rx = Val(SMatrix{3,3,Int}(0, 0, -1, 0, 1, 0, 1, 0, 0))
    @test parent(volume_ratio(mk(toblocks(T), bg))) == toblocks(parent(volume_ratio(mk(T, tg))))
    for R in (Val(one(SMatrix{3,3,Int})), Rx)
        st, sb = mk(T, tg, R), mk(toblocks(T), bg, R)
        for f in (s -> map_plasma((ps, _, _) -> ps.bsq, s), s -> map_plasma((ps, _, _) -> ps.u, s),
                  s -> map_plasma((_, g, c) -> (g, c.B), s), flow_speed, volume_ratio)
            ot, ob = f(st), f(sb)
            @test dimnames(ob) == bnames && size(ob) == (5, 3, 4, 8)
            @test all(map((x, y) -> all(isapprox.(x, y; rtol = 1e-13)), parent(ob), toblocks(parent(ot))))
        end
        @test map_plasma((ps, _, _) -> ps.bsq, sb; threaded = true) == map_plasma((ps, _, _) -> ps.bsq, sb; threaded = false)
        @inferred flow_speed(sb)
    end

    # out-of-grid zeros have the in-grid type: Float32 fluid with Float64 grid and point, and all Float32; both grid kinds
    f32(A) = map(c -> map(x -> Float32.(x), c), A)
    tg32 = TensorGrid(Float32.(r_prof), Float32.(th_grid), Float32(tg.φ0), Float32(tg.dφ))
    bg32 = BlockGrid(HAMRSph(), level, coord, nb, Float32.(startx), Float32.(ext0),
                     Interval(Float32(first(r_prof)), Float32(last(r_prof))))
    for (grid, grid32, A) in ((tg, tg32, T), (bg, bg32, toblocks(T))),
        (s, P) in ((mk(f32(A), grid), Float64), (mk(f32(A), grid32), Float32))
        vi = @inferred sample(s, P.((5.0, 1.0, 2.0))...)
        vo = @inferred sample(s, P.((0.0, 0.0, 1000.0))...)
        @test typeof(vi) == typeof(vo) && eltype(vi.ũ) == eltype(vi.B) == P
        @test vo == map(zero, vi)
    end
end

@testitem "_" begin
    import Aqua
    Aqua.test_all(GRMHDSnapshots; ambiguities=false)
    Aqua.test_ambiguities(GRMHDSnapshots)

    import CompatHelperLocal as CHL
    CHL.@check()
end
