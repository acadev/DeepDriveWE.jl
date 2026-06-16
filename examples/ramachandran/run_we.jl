using DeepDriveWE
using Random
using Unitful
using JLD2

function main()
    rng = Random.MersenneTwister(42)
    config = AlanineDipeptideConfig(
        n_steps = 500,           # 1 ps segments
        dt = 0.002u"ps",
        temperature = 300.0u"K",
        friction = 1.0u"ps^-1",
    )

    n_iterations = 80
    sims_per_bin = 4
    nbins = 10

    outdir = mkpath(joinpath(@__DIR__, "output"))
    basis_dir = mkpath(joinpath(outdir, "basis", "system1"))
    basis_path = joinpath(basis_dir, "basis_state.jld2")
    init_basis_state!(config, basis_path; rng = rng, n_equil_steps = 5000)

    bs = BasisStates(;
        basis_state_dir = joinpath(outdir, "basis"),
        basis_state_ext = ".jld2",
        initial_ensemble_members = sims_per_bin,
    )

    we = WeightedEnsemble(; basis_states = bs, target_states = TargetState[])
    initialize_basis_states!(we, basis_state_initializer(config))

    binner = RectilinearBinner(
        collect(range(-pi, pi; length = nbins + 1)), sims_per_bin;
        target_state_inds = Int[], pcoord_idx = 1,
    )
    recycler = LowRecycler(bs, -100.0)  # never recycle (sampling-only run)
    resampler = HuberKimResampler(; sims_per_bin = sims_per_bin)

    segdir = mkpath(joinpath(outdir, "segments"))

    next_sims = we.next_sims
    records = NamedTuple{(:iteration, :phi, :psi, :weight), Tuple{Int, Float64, Float64, Float64}}[]
    total_steps = 0

    for it in 1:n_iterations
        cur_sims = SimMetadata[]
        for sim in next_sims
            restart_path = joinpath(segdir, "iter$(it)_sim$(sim.simulation_id).jld2")
            run_segment!(config, sim, restart_path; rng = rng)
            push!(cur_sims, sim)
            push!(records, (
                iteration = it,
                phi = sim.pcoord[end][1],
                psi = sim.pcoord[end][2],
                weight = sim.weight,
            ))
            total_steps += config.n_steps
        end

        new_cur, new_sims, metadata = run_resampling(resampler, cur_sims, binner, recycler)
        advance_iteration!(we, new_cur, new_sims, metadata)

        next_sims = new_sims
        for s in next_sims
            s.iteration_id = it + 1
        end

        total_weight = sum(s.weight for s in new_sims)
        println("iter $it: n_walkers=$(length(new_sims)), total_weight=$(round(total_weight; digits=10)), bins=$(metadata.bin_target_counts)")
    end

    println("Total MD steps run (WE): $total_steps")

    JLD2.jldsave(joinpath(outdir, "we_data.jld2"); records = records, total_steps = total_steps)
    return records, total_steps
end

main()
