using DeepDriveWE
using Random
using Unitful
using JLD2

"""
Data-collection WE run for the 5AWL IDP fragment.

The progress coordinate is the full 18-element backbone dihedral vector
(`compute_pcoord(config, sys)`, see `IDPFragmentConfig`). For the WE binning
itself we only need *some* low-dimensional CV to drive exploration, so we bin
on the (phi_2, psi_1) pair (pcoord indices 1 and 10) - the first interior
residue's backbone dihedrals. The full 18-dimensional pcoord for every walker
at every iteration is recorded to `we_data.jld2` and used downstream as the
training set for the CVAE (Phase 4 step 3).
"""
function main()
    rng = Random.MersenneTwister(11)
    config = IDPFragmentConfig(
        n_steps = 500,           # 1 ps segments
        dt = 0.002u"ps",
        temperature = 300.0u"K",
        friction = 1.0u"ps^-1",
    )

    n_iterations = 100
    bin_target_count = 2
    nbins_per_dim = 8

    outdir = mkpath(joinpath(@__DIR__, "output_driver"))
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
    binner = RectilinearBinner2D(edges, edges, bin_target_count; pcoord_idxs = (1, 10))
    recycler = LowRecycler(bs, -100.0)  # never recycle (sampling-only run)
    resampler = HuberKimResampler(; sims_per_bin = bin_target_count)

    # Drive the WE campaign one iteration at a time via the Phase 3 driver,
    # so checkpoints are written after every iteration and can be resumed.
    run_config = WERunConfig(; output_dir = outdir, n_iterations = 1, checkpoint_interval = 1, rng = rng)

    records = NamedTuple{(:iteration, :pcoord, :weight), Tuple{Int, Vector{Float64}, Float64}}[]
    total_steps = 0

    for it in 1:n_iterations
        run_we!(we, config, binner, recycler, resampler, run_config)

        for sim in we.cur_sims
            push!(records, (
                iteration = it,
                pcoord = sim.pcoord[end],
                weight = sim.weight,
            ))
            total_steps += simulation_steps(config)
        end

        if it % 10 == 0
            println("iteration $it / $n_iterations, n_walkers = $(length(we.cur_sims)), total_steps = $total_steps")
        end
    end

    println("Total MD steps run (WE): $total_steps")

    JLD2.jldsave(joinpath(outdir, "we_data.jld2"); records = records, total_steps = total_steps)
    return records, total_steps
end

main()
