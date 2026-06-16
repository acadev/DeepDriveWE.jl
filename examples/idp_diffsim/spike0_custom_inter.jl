using Molly
using Enzyme
using AtomsCalculators
using LinearAlgebra
using StaticArrays

"""
Molly + Enzyme + a custom `general_inter` (no Flux): gradient of a loss
w.r.t. a scalar parameter `k` of a custom pairwise-force `general_inter`,
through a short VelocityVerlet simulation. Used to isolate whether the
"Type Module does not have a definite size" Enzyme error seen in
`spike0_minimal.jl` is specifically about `general_inters` + Enzyme, or
specifically about a Flux `Chain` inside a `general_inter`.
"""

struct ScalarBond{T}
    k::T
end

function AtomsCalculators.forces!(fs, sys, inter::ScalarBond; kwargs...)
    vec_ij = vector(sys.coords[1], sys.coords[3], sys.boundary)
    dist = norm(vec_ij)
    f = inter.k * dist * normalize(vec_ij)
    fs .+= [f, zero(f), -f]
    return fs
end

function loss(k, coords, velocities, atoms, boundary, simulator, n_steps)
    sys = System(
        atoms = atoms,
        coords = coords,
        boundary = boundary,
        velocities = velocities,
        general_inters = (ScalarBond(k),),
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

    k = 0.1f0
    n_steps = 10

    grad = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse), loss, Enzyme.Active,
        Enzyme.Active(k),
        Enzyme.Duplicated(copy(coords), zero(coords)),
        Enzyme.Duplicated(copy(velocities), zero(velocities)),
        Enzyme.Const(atoms), Enzyme.Const(boundary), Enzyme.Const(simulator), Enzyme.Const(n_steps),
    )

    println("grad = ", grad)
end

main()
