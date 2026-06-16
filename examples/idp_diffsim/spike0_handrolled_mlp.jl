using Molly
using Enzyme
using AtomsCalculators
using LinearAlgebra
using StaticArrays
using NNlib: relu

"""
Molly + Enzyme + a hand-rolled 2-layer MLP (no `Flux.Chain`) as a
`general_inter`: gradient of a loss w.r.t. the MLP's weight/bias arrays,
through a short VelocityVerlet simulation. `spike0_minimal.jl` (the doc's
`NNBonds` example verbatim, using a `Flux.Chain`) hit an Enzyme "Type Module
does not have a definite size" error at compile time in this Julia
1.12/Enzyme 0.13/Flux 0.16 combination; this checks whether avoiding
`Flux.Chain` (and its activation-function type parameters) sidesteps that.
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
    sys = System(
        atoms = atoms,
        coords = coords,
        boundary = boundary,
        velocities = velocities,
        general_inters = (NNBonds(model),),
        force_units = NoUnits,
        energy_units = NoUnits,
    )

    simulate!(sys, simulator, n_steps)

    dist_end = (norm(vector(sys.coords[1], sys.coords[2], boundary)) +
                norm(vector(sys.coords[2], sys.coords[3], boundary)) +
                norm(vector(sys.coords[3], sys.coords[1], boundary))) / 3
    return abs(dist_end - dist_true)
end

function main()
    dist_true = 1.0f0
    n_steps = 50
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

    model = MLP(0.1f0 .* randn(Float32, 5, 1), zeros(Float32, 5), 0.1f0 .* randn(Float32, 1, 5), zeros(Float32, 1))
    d_model = MLP(zero(model.W1), zero(model.b1), zero(model.W2), zero(model.b2))

    grad = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse), loss, Enzyme.Active,
        Enzyme.Duplicated(model, d_model),
        Enzyme.Duplicated(copy(coords), zero(coords)),
        Enzyme.Duplicated(copy(velocities), zero(velocities)),
        Enzyme.Const(atoms), Enzyme.Const(boundary), Enzyme.Const(simulator),
        Enzyme.Const(n_steps), Enzyme.Const(dist_true),
    )

    println("grad (primal loss) = ", grad)
    println("d_model.W1 = ", d_model.W1)
    println("d_model.b1 = ", d_model.b1)
    println("d_model.W2 = ", d_model.W2)
    println("d_model.b2 = ", d_model.b2)
end

main()
