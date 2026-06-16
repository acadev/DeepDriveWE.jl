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
Track A Stage 2 - differentiable fine-tuning on alanine dipeptide.

Loads the [`TrunkHeads`](@ref) model pretrained in `stage1_pretrain.jl`,
builds a NoUnits `System` with [`NNCorrection2`](@ref) as a `general_inter`
(same validated pattern as `spike_alanine.jl`), and fine-tunes the trunk + CV
head + correction head end-to-end via `Enzyme.autodiff` through a
`Langevin` rollout.

Loss combines:
- `cv_loss`: the CV head should recover `[phi, psi]` from the trunk embedding
  of the *end-of-rollout* configuration (representation-learning term, keeps
  the trunk's embedding meaningful).
- `obs_loss`: `(phi_end - phi_target)^2`, an observable-matching term that
  nudges the corrected dynamics towards a target backbone dihedral basin
  (here the C7eq/alpha region, `phi_target = -1.0`, matching `spike_alanine.jl`).

Checkpoint: confirm the combined loss decreases over training iterations and
that gradients remain finite throughout.
"""

function build_nounits_system(config, inter)
    ff = MolecularForceField(config.ff_files...; units=false)
    sys_ref = System(config.pdb_file, ff; units=false, nonbonded_method=:none, strictness=:nowarn)

    return System(
        atoms = sys_ref.atoms,
        coords = sys_ref.coords,
        boundary = sys_ref.boundary,
        velocities = zero(sys_ref.coords),
        specific_inter_lists = sys_ref.specific_inter_lists,
        general_inters = (inter,),
        force_units = NoUnits,
        energy_units = NoUnits,
    )
end

function loss(model, coords, velocities, sys_ref, config, pairs, scale, simulator, n_steps, phi_target, cv_weight, obs_weight)
    inter = NNCorrection2(model, config, pairs, scale)
    sys = System(
        atoms = sys_ref.atoms,
        coords = coords,
        boundary = sys_ref.boundary,
        velocities = velocities,
        specific_inter_lists = sys_ref.specific_inter_lists,
        general_inters = (inter,),
        force_units = NoUnits,
        energy_units = NoUnits,
    )

    simulate!(sys, simulator, n_steps; rng = Random.default_rng())

    pcoord = compute_pcoord(config, sys)
    features = dihedral_features(pcoord)
    h = trunk_embed(model, features)
    cv = cv_latent(model, h)

    cv_loss = sum(abs2, cv .- pcoord)
    obs_loss = (pcoord[1] - phi_target)^2

    return cv_weight * cv_loss + obs_weight * obs_loss
end

function main()
    config = AlanineDipeptideConfig(n_steps = 100)
    pairs = backbone_pairs(config)

    pretrained = JLD2.load(joinpath(@__DIR__, "output_stage1", "trunkheads_pretrained.jld2"))
    model = TrunkHeads(pretrained["Wt"], pretrained["bt"], pretrained["Wcv"], pretrained["bcv"], pretrained["Wc"], pretrained["bc"])
    scale = pretrained["scale"]

    inter0 = NNCorrection2(model, config, pairs, scale)
    sys_ref = build_nounits_system(config, inter0)

    simulator = Langevin(
        dt = ustrip(u"ps", config.dt),
        temperature = ustrip(u"K", config.temperature),
        friction = ustrip(u"ps^-1", config.friction),
    )

    phi_target = -1.0
    cv_weight = 1.0
    obs_weight = 5.0
    n_steps = config.n_steps

    opt_state = Optimisers.setup(Optimisers.Adam(1e-3), model)

    n_iters = 30
    for it in 1:n_iters
        d_model = Functors.fmap(zero, model)

        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse), loss, Enzyme.Active,
            Enzyme.Duplicated(model, d_model),
            Enzyme.Duplicated(copy(sys_ref.coords), zero(sys_ref.coords)),
            Enzyme.Duplicated(copy(sys_ref.velocities), zero(sys_ref.velocities)),
            Enzyme.Const(sys_ref), Enzyme.Const(config), Enzyme.Const(pairs), Enzyme.Const(scale),
            Enzyme.Const(simulator), Enzyme.Const(n_steps), Enzyme.Const(phi_target),
            Enzyme.Const(cv_weight), Enzyme.Const(obs_weight),
        )

        opt_state, model = Optimisers.update!(opt_state, model, d_model)

        loss_val = loss(model, copy(sys_ref.coords), copy(sys_ref.velocities), sys_ref, config, pairs, scale,
                         simulator, n_steps, phi_target, cv_weight, obs_weight)
        println("iter $it / $n_iters, loss = $loss_val")
    end

    outdir = mkpath(joinpath(@__DIR__, "output_stage2"))
    JLD2.jldsave(
        joinpath(outdir, "trunkheads_finetuned.jld2");
        Wt = model.Wt, bt = model.bt, Wcv = model.Wcv, bcv = model.bcv,
        Wc = model.Wc, bc = model.bc, scale = scale,
    )
    println("Saved ", joinpath(outdir, "trunkheads_finetuned.jld2"))

    return model
end

main()
