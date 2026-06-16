using Random

@testset "alanine_dipeptide" begin
    config = AlanineDipeptideConfig(n_steps = 20)

    @testset "build_system and compute_pcoord" begin
        sys = build_system(config)
        @test length(sys.coords) == 22

        pcoord = compute_pcoord(config, sys)
        @test length(pcoord) == 2
        @test all(-π <= p <= π for p in pcoord)
    end

    @testset "restart roundtrip" begin
        mktempdir() do dir
            sys = build_system(config)
            path = joinpath(dir, "state.jld2")
            save_restart(path, sys)

            coords, velocities = load_restart(path)
            @test coords == sys.coords
            @test velocities == sys.velocities

            sys2 = build_system(config; coords = coords, velocities = velocities)
            @test compute_pcoord(config, sys2) == compute_pcoord(config, sys)
        end
    end

    @testset "init_basis_state! and run_segment!" begin
        mktempdir() do dir
            rng = Random.MersenneTwister(1)

            basis_path = joinpath(dir, "basis_state.jld2")
            pcoord0 = init_basis_state!(config, basis_path; rng = rng, n_equil_steps = 20)

            @test length(pcoord0) == 2
            @test isfile(basis_path)

            sim = SimMetadata(;
                weight = 1.0,
                simulation_id = 1,
                iteration_id = 1,
                parent_restart_file = basis_path,
                parent_pcoord = pcoord0,
            )

            restart_path = joinpath(dir, "seg1.jld2")
            run_segment!(config, sim, restart_path; rng = rng)

            @test isfile(restart_path)
            @test sim.restart_file == restart_path
            @test num_frames(sim) == 2
            @test sim.pcoord[1] == pcoord0
            @test length(sim.pcoord[2]) == 2
            @test walltime(sim) >= 0.0
        end
    end

    @testset "BasisStates loading" begin
        mktempdir() do dir
            rng = Random.MersenneTwister(2)

            sysdir = joinpath(dir, "system1")
            mkpath(sysdir)
            basis_path = joinpath(sysdir, "basis_state.jld2")
            pcoord0 = init_basis_state!(config, basis_path; rng = rng, n_equil_steps = 20)

            bs = BasisStates(;
                basis_state_dir = dir,
                basis_state_ext = ".jld2",
                initial_ensemble_members = 2,
            )

            load_basis_states!(bs, basis_state_initializer(config))

            @test bs.num_basis_files == 1
            @test length(bs) == 2
            @test all(s.parent_pcoord ≈ pcoord0 for s in bs)
            @test all(s.weight ≈ 0.5 for s in bs)
        end
    end
end
