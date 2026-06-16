using JSON3

@testset "api" begin
    @testset "SimMetadata helpers" begin
        sim = SimMetadata(;
            weight = 0.5,
            simulation_id = 1,
            iteration_id = 1,
            parent_restart_file = "basis/state1.jld2",
            parent_pcoord = [1.0],
        )

        @test sim.endpoint_type == 1
        @test sim.wtg_parent_ids == Int[]
        @test sim.pcoord == Vector{Float64}[]
        @test num_frames(sim) == 0
        @test simulation_name(sim) == "000001/000001"

        # append_pcoord! requires matching frame counts
        push!(sim.pcoord, Float64[])
        push!(sim.pcoord, Float64[])
        append_pcoord!(sim, [1.0, 2.0])
        @test sim.pcoord == [[1.0], [2.0]]
        @test_throws ArgumentError append_pcoord!(sim, [1.0, 2.0, 3.0])

        mark_simulation_start!(sim)
        mark_simulation_end!(sim)
        @test walltime(sim) >= 0.0
    end

    @testset "BasisStates loading" begin
        mktempdir() do dir
            for (i, pcoord) in enumerate([0.1, 0.9])
                sysdir = joinpath(dir, "system$i")
                mkpath(sysdir)
                write(joinpath(sysdir, "state.jld2"), "dummy")
            end

            bs = BasisStates(;
                basis_state_dir = dir,
                basis_state_ext = ".jld2",
                initial_ensemble_members = 4,
            )

            initializer(file) = [parse(Float64, split(basename(dirname(file)), "system")[2]) / 10]

            load_basis_states!(bs, initializer)

            @test bs.num_basis_files == 2
            @test length(bs) == 4
            @test length(unique_basis_states(bs)) == 2

            # Uniform weights summing to 1
            @test all(s.weight ≈ 0.25 for s in bs)
            @test sum(s.weight for s in bs) ≈ 1.0

            # Simulation ids 1..4, cycling through the 2 basis files
            @test [s.simulation_id for s in bs] == [1, 2, 3, 4]
            @test [s.parent_simulation_id for s in bs] == [-1, -2, -1, -2]
            @test [s.wtg_parent_ids for s in bs] == [[-1], [-2], [-1], [-2]]
            @test all(s.iteration_id == 1 for s in bs)
        end
    end

    @testset "WeightedEnsemble JSON round trip" begin
        bs = BasisStates(;
            basis_state_dir = "basis",
            initial_ensemble_members = 2,
            num_basis_files = 1,
            basis_states = [
                SimMetadata(;
                    weight = 0.5,
                    simulation_id = 1,
                    iteration_id = 1,
                    parent_restart_file = "basis/state1.jld2",
                    parent_pcoord = [0.1],
                    parent_simulation_id = -1,
                    wtg_parent_ids = [-1],
                ),
                SimMetadata(;
                    weight = 0.5,
                    simulation_id = 2,
                    iteration_id = 1,
                    parent_restart_file = "basis/state1.jld2",
                    parent_pcoord = [0.1],
                    parent_simulation_id = -1,
                    wtg_parent_ids = [-1],
                ),
            ],
        )

        we = WeightedEnsemble(;
            basis_states = bs,
            target_states = [TargetState(; label = "folded", pcoord = [1.0])],
        )
        we.next_sims = deepcopy(bs.basis_states)

        json_str = JSON3.write(we)
        we2 = JSON3.read(json_str, WeightedEnsemble)

        @test we2.metadata.iteration_id == we.metadata.iteration_id
        @test length(we2.next_sims) == 2
        @test we2.next_sims[1].weight == 0.5
        @test we2.target_states[1].label == "folded"
        @test we2.basis_states.initial_ensemble_members == 2
        @test iteration(we2) == 1
    end
end
