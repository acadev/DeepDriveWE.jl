using DeepDriveWE
using Molly
using Enzyme
using AtomsCalculators
using Random
using LinearAlgebra
using Unitful
using NNlib: relu

"""
Spike 0: de-risk the differentiable-simulation AD pipeline on alanine
dipeptide before investing in the full Track A architecture.

Builds a NoUnits `System` (required for Enzyme), adds a tiny hand-rolled MLP
"correction" general interaction (a pairwise force along a backbone vector,
mirroring the `NNBonds` example in `docs/src/differentiable.md` but with a
hand-rolled 2-layer MLP instead of `Flux.Chain` - a `Flux.Chain` general_inter
triggers an Enzyme "Type Module does not have a definite size" compile error
in this Julia 1.12 / Enzyme 0.13 / Flux 0.16 combination, confirmed via
`spike0_minimal.jl` vs `spike0_handrolled_mlp.jl`), runs a short Langevin
rollout, and computes `d(loss)/d(model params)` via `Enzyme.autodiff`. The
loss is `|phi_end - phi_target|`, computed via the existing
`compute_pcoord(config, sys)`.

Checkpoint: confirm the gradient is finite and non-zero before building out
the full trunk + CV head + correction head.
"""

struct MLP{T}
    W1::Matrix{T}
    b1::Vector{T}
    W2::Matrix{T}
    b2::Vector{T}
end

function (m::MLP)(x)
    h = relu.(m.W1 * x .+ m.b1)
    return tanh.(m.W2 * h .+ m.b2)
end

struct NNCorrection{T}
    model::T
    i::Int
    j::Int
end

function AtomsCalculators.forces!(fs, sys, inter::NNCorrection; kwargs...)
    vec_ij = vector(sys.coords[inter.i], sys.coords[inter.j], sys.boundary)
    dist = norm(vec_ij)
    f = inter.model([dist])[1] * normalize(vec_ij)
    fs[inter.i] = fs[inter.i] .+ f
    fs[inter.j] = fs[inter.j] .- f
    return fs
end

function build_nounits_system(config, model)
    ff = MolecularForceField(config.ff_files...; units=false)
    sys_ref = System(config.pdb_file, ff; units=false, nonbonded_method=:none, strictness=:nowarn)

    return System(
        atoms = sys_ref.atoms,
        coords = sys_ref.coords,
        boundary = sys_ref.boundary,
        velocities = zero(sys_ref.coords),
        specific_inter_lists = sys_ref.specific_inter_lists,
        general_inters = (NNCorrection(model, 7, 17),),
        force_units = NoUnits,
        energy_units = NoUnits,
    )
end

function loss(model, coords, velocities, sys_ref, simulator, n_steps, config, phi_target)
    sys = System(
        atoms = sys_ref.atoms,
        coords = coords,
        boundary = sys_ref.boundary,
        velocities = velocities,
        specific_inter_lists = sys_ref.specific_inter_lists,
        general_inters = (NNCorrection(model, 7, 17),),
        force_units = NoUnits,
        energy_units = NoUnits,
    )

    simulate!(sys, simulator, n_steps; rng = Random.default_rng())

    pcoord = compute_pcoord(config, sys)
    loss_val = abs(pcoord[1] - phi_target)
    return loss_val
end

function main()
    config = AlanineDipeptideConfig(n_steps = 50)

    model = MLP(
        0.01 .* randn(8, 1), zeros(8),
        0.01 .* randn(1, 8), zeros(1),
    )
    sys_ref = build_nounits_system(config, model)

    simulator = Langevin(
        dt = ustrip(u"ps", config.dt),
        temperature = ustrip(u"K", config.temperature),
        friction = ustrip(u"ps^-1", config.friction),
    )

    phi_target = -1.0
    n_steps = config.n_steps

    d_model = MLP(zero(model.W1), zero(model.b1), zero(model.W2), zero(model.b2))

    grad = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse), loss, Enzyme.Active,
        Enzyme.Duplicated(model, d_model),
        Enzyme.Duplicated(copy(sys_ref.coords), zero(sys_ref.coords)),
        Enzyme.Duplicated(copy(sys_ref.velocities), zero(sys_ref.velocities)),
        Enzyme.Const(sys_ref), Enzyme.Const(simulator), Enzyme.Const(n_steps),
        Enzyme.Const(config), Enzyme.Const(phi_target),
    )

    println("grad = ", grad)
    println("d_model.W1 = ", d_model.W1)
    println("d_model.b1 = ", d_model.b1)
    println("d_model.W2 = ", d_model.W2)
    println("d_model.b2 = ", d_model.b2)

    return grad, d_model
end

main()
