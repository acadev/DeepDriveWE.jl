using Molly
using Enzyme
using LinearAlgebra
using StaticArrays

"""
Molly-only (no Flux) differentiable-simulation check: gradient of a loss
w.r.t. a scalar atom property (sigma) through a short VelocityVerlet
simulation, mirroring `test/gradients.jl`'s "Differentiable simulation"
testset. Used to isolate whether the "Type Module does not have a definite
size" Enzyme error seen in `spike0_minimal.jl` comes from Molly+Enzyme itself
or from the combination with a Flux `Chain` as a `general_inter`.
"""

function loss(σ, coords, velocities, atoms, boundary, simulator, n_steps)
    atoms2 = [Atom(a.index, a.atom_type, a.mass, a.charge, σ, a.ϵ) for a in atoms]

    sys = System(
        atoms = atoms2,
        coords = coords,
        boundary = boundary,
        velocities = velocities,
        force_units = NoUnits,
        energy_units = NoUnits,
    )

    simulate!(sys, simulator, n_steps)

    return sum(norm(c) for c in sys.coords)
end

function main()
    n_atoms = 3
    boundary = CubicBoundary(5.0f0)
    coords = [
        SVector(2.3f0, 2.07f0, 0.0f0),
        SVector(2.5f0, 2.93f0, 0.0f0),
        SVector(2.7f0, 2.07f0, 0.0f0),
    ]
    velocities = zero(coords)
    atoms = [Atom(i, 1, 10.0f0, 0.0f0, 0.0f0, 0.0f0) for i in 1:n_atoms]
    simulator = VelocityVerlet(dt = 0.02f0)

    σ = 0.0f0
    n_steps = 10

    grad = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse), loss, Enzyme.Active,
        Enzyme.Active(σ),
        Enzyme.Duplicated(copy(coords), zero(coords)),
        Enzyme.Duplicated(copy(velocities), zero(velocities)),
        Enzyme.Const(atoms), Enzyme.Const(boundary), Enzyme.Const(simulator), Enzyme.Const(n_steps),
    )

    println("grad = ", grad)
end

main()
