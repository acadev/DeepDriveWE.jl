using DeepDriveWE
using Molly
using Enzyme
using Optimisers
using Functors
using Random
using Unitful
using JLD2
using LinearAlgebra

include("model.jl")

"""
Track A Stage 1 - force-matching pretrain (alanine dipeptide).

Runs a short plain-MD trajectory with the full a99SB-disp force field
(`build_system`, with units), and for a set of sampled configurations:

- computes the 4-dim dihedral feature vector `[sin phi, cos phi, sin psi,
  cos psi]` (the [`TrunkHeads`](@ref) trunk input),
- computes the *classical* a99SB-disp net-force component on each backbone
  pair atom along the pair's separation vector (the force-matching target for
  the correction head),
- computes the raw `[phi, psi]` angles (the CV head's regression target).

[`TrunkHeads`](@ref) is then pretrained (via Enzyme + Optimisers) so its
correction head approximately reproduces a *scaled-down* version of these
classical force projections - a stable, non-random starting point before
Stage 2's differentiable fine-tuning - and its CV head learns to recover
`[phi, psi]` from the trunk embedding.
"""

function collect_force_matching_data(config; n_samples = 200, sample_every = 20, rng)
    sys = build_system(config)
    random_velocities!(sys, config.temperature; rng = rng)
    simulator = Langevin(dt = config.dt, temperature = config.temperature, friction = config.friction)

    simulate!(sys, simulator, 5000; rng = rng)  # equilibration

    pairs = backbone_pairs(config)
    n_pairs = length(pairs)

    features = Matrix{Float64}(undef, 4, n_samples)
    targets = Matrix{Float64}(undef, n_pairs, n_samples)
    cv_targets = Matrix{Float64}(undef, 2, n_samples)

    for s in 1:n_samples
        simulate!(sys, simulator, sample_every; rng = rng)

        pcoord = compute_pcoord(config, sys)
        features[:, s] = dihedral_features(pcoord)
        cv_targets[:, s] = pcoord

        fs = forces(sys)
        for (idx, (i, j)) in enumerate(pairs)
            vec_ij = vector(sys.coords[i], sys.coords[j], sys.boundary)
            f_i = ustrip.(fs[i])
            targets[idx, s] = dot(f_i, normalize(vec_ij))
        end

        if s % 50 == 0
            println("force-matching data: sample $s / $n_samples")
        end
    end

    return features, targets, cv_targets
end

function main()
    rng = Random.MersenneTwister(42)
    config = AlanineDipeptideConfig()

    features, targets, cv_targets = collect_force_matching_data(config; n_samples = 200, sample_every = 20, rng = rng)

    scale = maximum(abs.(targets))
    targets_scaled = targets ./ scale
    println("force-matching target scale = $scale")

    pairs = backbone_pairs(config)
    model = TrunkHeads(Float64, 4, 16, length(pairs))

    function loss(model, features, targets_scaled, cv_targets)
        h = trunk_embed(model, features)
        corr = correction_magnitudes(model, h)
        cv = cv_latent(model, h)
        corr_loss = sum(abs2, corr .- targets_scaled) / length(targets_scaled)
        cv_loss = sum(abs2, cv .- cv_targets) / length(cv_targets)
        return corr_loss + cv_loss
    end

    opt_state = Optimisers.setup(Optimisers.Adam(1e-2), model)

    n_iters = 500
    for it in 1:n_iters
        d_model = Functors.fmap(zero, model)

        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse), loss, Enzyme.Active,
            Enzyme.Duplicated(model, d_model),
            Enzyme.Const(features), Enzyme.Const(targets_scaled), Enzyme.Const(cv_targets),
        )

        opt_state, model = Optimisers.update!(opt_state, model, d_model)

        if it % 50 == 0 || it == 1
            loss_val = loss(model, features, targets_scaled, cv_targets)
            println("iter $it / $n_iters, loss = $loss_val")
        end
    end

    outdir = mkpath(joinpath(@__DIR__, "output_stage1"))
    JLD2.jldsave(
        joinpath(outdir, "trunkheads_pretrained.jld2");
        Wt = model.Wt, bt = model.bt, Wcv = model.Wcv, bcv = model.bcv,
        Wc = model.Wc, bc = model.bc, scale = scale,
    )
    println("Saved ", joinpath(outdir, "trunkheads_pretrained.jld2"))

    return model, scale
end

main()
