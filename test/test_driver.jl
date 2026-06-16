using Random
using JLD2

# A minimal synthetic 1D-diffusion backend implementing the simulation
# backend interface (see src/simulation/backend.jl), used to test `run_we!`
# without depending on Molly.
Base.@kwdef struct ToyConfig
    n_steps::Int = 5
    step_size::Float64 = 0.1
end

DeepDriveWE.simulation_steps(config::ToyConfig) = config.n_steps

function DeepDriveWE.init_basis_state!(
    config::ToyConfig,
    path::AbstractString;
    rng::Random.AbstractRNG = Random.default_rng(),
    x0::Float64 = 0.0,
)
    JLD2.jldsave(path; x = x0)
    return [x0]
end

function DeepDriveWE.run_segment!(
    config::ToyConfig,
    sim::SimMetadata,
    restart_path::AbstractString;
    rng::Random.AbstractRNG = Random.default_rng(),
)
    x = JLD2.load(sim.parent_restart_file, "x")

    mark_simulation_start!(sim)
    for _ in 1:config.n_steps
        x += config.step_size * randn(rng)
    end
    mark_simulation_end!(sim)

    sim.pcoord = [copy(sim.parent_pcoord), [x]]
    JLD2.jldsave(restart_path; x = x)
    sim.restart_file = restart_path
    return sim
end

toy_basis_state_initializer(config::ToyConfig) =
    (basis_file::String) -> [JLD2.load(basis_file, "x")]

function setup_toy_we(dir::AbstractString, config::ToyConfig; sims_per_bin::Int = 4)
    basis_dir = mkpath(joinpath(dir, "basis", "system1"))
    basis_path = joinpath(basis_dir, "basis_state.jld2")
    init_basis_state!(config, basis_path; x0 = 0.0)

    bs = BasisStates(;
        basis_state_dir = joinpath(dir, "basis"),
        basis_state_ext = ".jld2",
        initial_ensemble_members = sims_per_bin,
    )

    we = WeightedEnsemble(; basis_states = bs, target_states = TargetState[])
    initialize_basis_states!(we, toy_basis_state_initializer(config))

    binner = RectilinearBinner(
        collect(range(-2.0, 2.0; length = 9)), sims_per_bin;
        target_state_inds = Int[], pcoord_idx = 1,
    )
    recycler = LowRecycler(bs, -100.0)
    resampler = HuberKimResampler(; sims_per_bin = sims_per_bin)

    return we, binner, recycler, resampler
end

@testset "driver" begin
    config = ToyConfig(n_steps = 5, step_size = 0.1)

    @testset "run_we! basic loop" begin
        mktempdir() do dir
            rng = Random.MersenneTwister(1)
            we, binner, recycler, resampler = setup_toy_we(dir, config)

            run_config = WERunConfig(;
                output_dir = joinpath(dir, "run1"), n_iterations = 3,
                checkpoint_interval = 2, rng = rng,
            )
            run_we!(we, config, binner, recycler, resampler, run_config)

            @test we.metadata.iteration_id == 3
            @test we.next_sims[1].iteration_id == 4
            @test sum(s.weight for s in we.next_sims) ≈ 1.0

            @test isfile(joinpath(dir, "run1", "checkpoints", "checkpoint-000002.json"))
            @test isfile(joinpath(dir, "run1", "checkpoints", "checkpoint-000003.json"))

            for sim in we.cur_sims
                @test isfile(joinpath(dir, "run1", "simulations", simulation_name(sim), "restart.jld2"))
                @test num_frames(sim) == 2
            end
        end
    end

    @testset "resume from checkpoint" begin
        mktempdir() do dir
            rng = Random.MersenneTwister(2)
            we, binner, recycler, resampler = setup_toy_we(dir, config)

            run_config = WERunConfig(;
                output_dir = joinpath(dir, "run1"), n_iterations = 2,
                checkpoint_interval = 1, rng = rng,
            )
            run_we!(we, config, binner, recycler, resampler, run_config)
            @test we.metadata.iteration_id == 2

            checkpointer = EnsembleCheckpointer(joinpath(dir, "run1"))
            we_resumed = load_checkpoint(checkpointer)
            @test we_resumed.metadata.iteration_id == 2
            @test we_resumed.next_sims[1].iteration_id == 3

            run_config2 = WERunConfig(;
                output_dir = joinpath(dir, "run1"), n_iterations = 2,
                checkpoint_interval = 1, rng = rng,
            )
            run_we!(we_resumed, config, binner, recycler, resampler, run_config2)

            @test we_resumed.metadata.iteration_id == 4
            @test sum(s.weight for s in we_resumed.next_sims) ≈ 1.0
            @test isfile(joinpath(dir, "run1", "checkpoints", "checkpoint-000004.json"))
        end
    end
end
