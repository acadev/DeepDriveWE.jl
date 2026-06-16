export WERunConfig, run_we!

"""
    WERunConfig(; output_dir, n_iterations, checkpoint_interval=1, rng=Random.default_rng())

Configuration for [`run_we!`](@ref).

- `output_dir`: directory for per-segment restart files (under
  `output_dir/simulations/<iteration>/<simulation_id>/`) and checkpoints
  (under `output_dir/checkpoints/`, via [`EnsembleCheckpointer`](@ref)).
- `n_iterations`: number of WE iterations to run.
- `checkpoint_interval`: save a checkpoint every this many iterations (and
  always after the final iteration).
- `rng`: random number generator passed to `run_segment!` and the resampler.
"""
Base.@kwdef struct WERunConfig
    output_dir::String
    n_iterations::Int
    checkpoint_interval::Int = 1
    rng::Random.AbstractRNG = Random.default_rng()
end

"""
    run_we!(we, sim_config, binner, recycler, resampler, run_config) -> WeightedEnsemble

Run `run_config.n_iterations` weighted-ensemble iterations, advancing `we` in
place.

For each iteration, every simulation in `we.next_sims` is propagated with
`run_segment!(sim_config, sim, restart_path; rng)`, then resampled via
[`run_resampling`](@ref) and [`advance_iteration!`](@ref). Restart files are
written under `output_dir/simulations/<iteration>/<simulation_id>/restart.jld2`,
and checkpoints are written every `checkpoint_interval` iterations.

The starting iteration is taken from `we.next_sims[1].iteration_id`, so this
function can be used both to start a fresh run (after
[`initialize_basis_states!`](@ref)) and to resume from a checkpoint loaded via
[`load_checkpoint`](@ref).
"""
function run_we!(
    we::WeightedEnsemble,
    sim_config,
    binner::AbstractBinner,
    recycler::AbstractRecycler,
    resampler::AbstractResampler,
    run_config::WERunConfig,
)
    isempty(we.next_sims) && throw(ArgumentError("`we.next_sims` is empty; call `initialize_basis_states!` first."))

    segdir = mkpath(joinpath(run_config.output_dir, "simulations"))
    checkpointer = EnsembleCheckpointer(run_config.output_dir)

    start_iter = we.next_sims[1].iteration_id
    end_iter = start_iter + run_config.n_iterations - 1

    for it in start_iter:end_iter
        cur_sims = SimMetadata[]
        for sim in we.next_sims
            sim_dir = mkpath(joinpath(segdir, simulation_name(sim)))
            restart_path = joinpath(sim_dir, "restart.jld2")
            run_segment!(sim_config, sim, restart_path; rng = run_config.rng)
            push!(cur_sims, sim)
        end

        new_cur, new_sims, metadata = run_resampling(resampler, cur_sims, binner, recycler)
        advance_iteration!(we, new_cur, new_sims, metadata)

        total_weight = sum(s.weight for s in we.next_sims)
        @info "WE iteration $it" n_walkers=length(we.next_sims) total_weight bin_target_counts=metadata.bin_target_counts

        if it % run_config.checkpoint_interval == 0 || it == end_iter
            save_checkpoint(checkpointer, we)
        end
    end

    return we
end
