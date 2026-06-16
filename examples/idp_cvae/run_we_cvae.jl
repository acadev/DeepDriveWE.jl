using DeepDriveWE
using Random
using Unitful
using JLD2

include("cvae_config.jl")

"""
CVAE-driven WE run for the 5AWL IDP fragment.

Uses the trained CVAE encoder (`cvae.jld2`, from `train_cvae.jl`) as the
progress coordinate via [`CVAELatentConfig`](@ref): each segment's 18
backbone dihedral angles are mapped to 36 sin/cos features, normalized, and
encoded to a 2D latent code. WE binning/resampling then operates directly on
this learned 2D latent space via `RectilinearBinner2D`.
"""
function main()
    rng = Random.MersenneTwister(23)

    physical = IDPFragmentConfig(
        n_steps = 500,           # 1 ps segments
        dt = 0.002u"ps",
        temperature = 300.0u"K",
        friction = 1.0u"ps^-1",
    )
    config = CVAELatentConfig(physical, joinpath(@__DIR__, "cvae.jld2"))

    n_iterations = 100
    bin_target_count = 2
    nbins_per_dim = 8
    latent_lo, latent_hi = -5.0, 5.0

    outdir = mkpath(joinpath(@__DIR__, "output_cvae"))
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

    edges = collect(range(latent_lo, latent_hi; length = nbins_per_dim + 1))
    binner = RectilinearBinner2D(edges, edges, bin_target_count; pcoord_idxs = (1, 2))
    recycler = LowRecycler(bs, -100.0)  # never recycle (sampling-only run)
    resampler = HuberKimResampler(; sims_per_bin = bin_target_count)

    run_config = WERunConfig(; output_dir = outdir, n_iterations = 1, checkpoint_interval = 1, rng = rng)

    records = NamedTuple{(:iteration, :pcoord, :latent, :weight), Tuple{Int, Vector{Float64}, Vector{Float64}, Float64}}[]
    total_steps = 0

    for it in 1:n_iterations
        run_we!(we, config, binner, recycler, resampler, run_config)

        for sim in we.cur_sims
            coords, velocities = load_restart(sim.restart_file)
            sys = build_system(physical; coords = coords, velocities = velocities)
            push!(records, (
                iteration = it,
                pcoord = compute_pcoord(physical, sys),  # full 18-dim dihedrals
                latent = sim.pcoord[end],                # 2D CVAE latent
                weight = sim.weight,
            ))
            total_steps += simulation_steps(config)
        end

        if it % 10 == 0
            println("iteration $it / $n_iterations, n_walkers = $(length(we.cur_sims)), total_steps = $total_steps")
        end
    end

    println("Total MD steps run (CVAE-driven WE): $total_steps")

    JLD2.jldsave(joinpath(outdir, "we_data.jld2"); records = records, total_steps = total_steps)
    return records, total_steps
end

main()
