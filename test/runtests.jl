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

@testitem "_" begin
    import Aqua
    Aqua.test_all(GRMHDSnapshots; ambiguities=false)
    Aqua.test_ambiguities(GRMHDSnapshots)

    import CompatHelperLocal as CHL
    CHL.@check()
end
