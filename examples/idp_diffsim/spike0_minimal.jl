using Molly
using Flux
using Enzyme
using AtomsCalculators
using LinearAlgebra
using StaticArrays

"""
Minimal AD pipeline check, mirroring the `NNBonds` example in
`docs/src/differentiable.md` almost verbatim (3 free atoms, no
specific_inter_lists, no pairwise inters/neighbor finder). Used to establish a
working baseline before adding alanine-dipeptide-specific complexity
(specific_inter_lists from a real PDB topology), which triggered an Enzyme
"Type Module does not have a definite size" error when run via `simulate!`.
"""

struct NNBonds{T}
    model::T
end

function AtomsCalculators.forces!(fs, sys, inter::NNBonds; kwargs...)
    vec_ij = vector(sys.coords[1], sys.coords[3], sys.boundary)
    dist = norm(vec_ij)
    f = inter.model([dist])[1] * normalize(vec_ij)
    fs .+= [f, zero(f), -f]
    return fs
end

function loss(model, coords, velocities, atoms, boundary, simulator, n_steps, dist_true)
    general_inters = (NNBonds(model),)

    sys = System(
        atoms = atoms,
        coords = coords,
        boundary = boundary,
        velocities = velocities,
        general_inters = general_inters,
        force_units = NoUnits,
        energy_units = NoUnits,
    )

    simulate!(sys, simulator, n_steps)

    dist_end = (norm(vector(sys.coords[1], sys.coords[2], boundary)) +
                norm(vector(sys.coords[2], sys.coords[3], boundary)) +
                norm(vector(sys.coords[3], sys.coords[1], boundary))) / 3
    loss_val = abs(dist_end - dist_true)
    return loss_val
end

function main()
    dist_true = 1.0f0
    n_steps = 400
    boundary = CubicBoundary(5.0f0)
    temp = 0.01f0
    coords = [
        SVector(2.3f0, 2.07f0, 0.0f0),
        SVector(2.5f0, 2.93f0, 0.0f0),
        SVector(2.7f0, 2.07f0, 0.0f0),
    ]
    n_atoms = length(coords)
    velocities = zero(coords)
    atoms = [Atom(i, 1, 10.0f0, 0.0f0, 0.0f0, 0.0f0) for i in 1:n_atoms]
    simulator = VelocityVerlet(
        dt = 0.02f0,
        coupling = BerendsenThermostat(temp, 0.5f0),
    )

    model = Chain(Dense(1, 5, relu), Dense(5, 1, tanh))
    d_model = Flux.fmap(model) do x
        x isa Array ? zero(x) : x
    end

    grad = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse), loss, Enzyme.Active,
        Enzyme.Duplicated(model, d_model),
        Enzyme.Duplicated(copy(coords), zero(coords)),
        Enzyme.Duplicated(copy(velocities), zero(velocities)),
        Enzyme.Const(atoms), Enzyme.Const(boundary), Enzyme.Const(simulator),
        Enzyme.Const(n_steps), Enzyme.Const(dist_true),
    )

    println("loss value (primal) = ", grad[2])
    println("d(loss)/d(model) = ", d_model)

    return grad, d_model
end

main()
