using DeepDriveWE
using Random
using Unitful
using JLD2

function main()
    rng = Random.MersenneTwister(7)
    config = AlanineDipeptideConfig(
        n_steps = 500,           # 1 ps segments
        dt = 0.002u"ps",
        temperature = 300.0u"K",
        friction = 1.0u"ps^-1",
    )

    n_iterations = 150
    bin_target_count = 2
    nbins_per_dim = 12

    outdir = mkpath(joinpath(@__DIR__, "output_2d"))
    basis_dir = mkpath(joinpath(outdir, "basis", "system1"))
    basis_path = joinpath(basis_dir, "basis_state.jld2")
    init_basis_state!(config, basis_path; rng = rng, n_equil_steps = 5000)

    bs = BasisStates(;
        basis_state_dir = joinpath(outdir, "basis"),
        basis_state_ext = ".jld2",
        initial_ensemble_members = bin_target_count,
    )

    we = WeightedEnsemble(; basis_states = bs, target_states = TargetState[])
    initialize_basis_states!(we, basis_state_initializer(config))

    edges = collect(range(-pi, pi; length = nbins_per_dim + 1))
    binner = RectilinearBinner2D(edges, edges, bin_target_count; pcoord_idxs = (1, 2))
    recycler = LowRecycler(bs, -100.0)  # never recycle (sampling-only run)
    resampler = HuberKimResampler(; sims_per_bin = bin_target_count)

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
        n_occupied = length(unique(assign_bins(binner, _pcoords_parent(new_sims))))
        println("iter $it: n_walkers=$(length(new_sims)), total_weight=$(round(total_weight; digits=10)), occupied_bins=$n_occupied")

        if it % 10 == 0
            JLD2.jldsave(joinpath(outdir, "we_data.jld2"); records = records, total_steps = total_steps)
        end
    end

    println("Total MD steps run (WE): $total_steps")

    JLD2.jldsave(joinpath(outdir, "we_data.jld2"); records = records, total_steps = total_steps)
    return records, total_steps
end

function _pcoords_parent(sims)
    n = length(sims)
    mat = Matrix{Float64}(undef, n, 2)
    for (i, s) in enumerate(sims)
        mat[i, :] .= s.parent_pcoord
    end
    return mat
end

main()
