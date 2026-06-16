export build_force_field, build_system, save_restart, load_restart, basis_state_initializer,
       init_basis_state!, run_segment!

"""
    build_force_field(config)

Build the `MolecularForceField` for `config`.
"""
build_force_field(config) = MolecularForceField(physical_config(config).ff_files...; units=true)

"""
    build_system(config; coords=nothing, velocities=nothing)

Build a Molly `System` in vacuum for `config`. If `coords` and/or
`velocities` are given, they replace the values loaded from the PDB file.
"""
function build_system(config; coords=nothing, velocities=nothing)
    pc = physical_config(config)
    ff = build_force_field(config)
    sys = System(pc.pdb_file, ff; nonbonded_method=:none)

    if coords !== nothing
        sys.coords = coords
    end
    if velocities !== nothing
        sys.velocities = velocities
    end

    return sys
end

"""
    save_restart(path, sys)

Save a system's coordinates and velocities (stripped to nm and nm/ps) to a
JLD2 file at `path`.
"""
function save_restart(path::AbstractString, sys)
    coords = [ustrip.(u"nm", c) for c in sys.coords]
    velocities = [ustrip.(u"nm/ps", v) for v in sys.velocities]
    JLD2.jldsave(path; coords=coords, velocities=velocities)
    return path
end

"""
    load_restart(path) -> (coords, velocities)

Load coordinates and velocities (in nm and nm/ps) saved by [`save_restart`](@ref).
"""
function load_restart(path::AbstractString)
    data = JLD2.load(path)
    coords = [SVector{3}(c) * u"nm" for c in data["coords"]]
    velocities = [SVector{3}(v) * u"nm/ps" for v in data["velocities"]]
    return coords, velocities
end

"""
    init_basis_state!(config, path; rng=Random.default_rng(), n_equil_steps=1000) -> Vector{Float64}

Build the system for `config`, assign random velocities at
`physical_config(config).temperature`, run a short Langevin equilibration,
save the resulting state to `path`, and return `compute_pcoord(config, sys)`.
"""
function init_basis_state!(
    config,
    path::AbstractString;
    rng::Random.AbstractRNG = Random.default_rng(),
    n_equil_steps::Int = 1000,
)
    pc = physical_config(config)
    sys = build_system(config)
    random_velocities!(sys, pc.temperature; rng = rng)

    if n_equil_steps > 0
        simulator = Langevin(dt = pc.dt, temperature = pc.temperature, friction = pc.friction)
        simulate!(sys, simulator, n_equil_steps; rng = rng)
    end

    save_restart(path, sys)
    return compute_pcoord(config, sys)
end

"""
    run_segment!(config, sim::SimMetadata, restart_path; rng=Random.default_rng()) -> SimMetadata

Run a short MD segment for `sim`, starting from `sim.parent_restart_file`.
Sets `sim.pcoord` to `[parent_pcoord, final_pcoord]`, writes the final state
to `restart_path`, sets `sim.restart_file`, and records start/end walltimes.
"""
function run_segment!(
    config,
    sim::SimMetadata,
    restart_path::AbstractString;
    rng::Random.AbstractRNG = Random.default_rng(),
)
    pc = physical_config(config)
    coords, velocities = load_restart(sim.parent_restart_file)
    sys = build_system(config; coords = coords, velocities = velocities)

    mark_simulation_start!(sim)
    simulator = Langevin(dt = pc.dt, temperature = pc.temperature, friction = pc.friction)
    simulate!(sys, simulator, pc.n_steps; rng = rng)
    mark_simulation_end!(sim)

    sim.pcoord = [copy(sim.parent_pcoord), compute_pcoord(config, sys)]
    save_restart(restart_path, sys)
    sim.restart_file = restart_path

    return sim
end

"""
    basis_state_initializer(config) -> Function

Return a `(basis_file::String) -> Vector{Float64}` function suitable for
[`load_basis_states!`](@ref), which loads a saved restart file and computes
`compute_pcoord(config, sys)`.
"""
function basis_state_initializer(config)
    return function (basis_file::String)
        coords, velocities = load_restart(basis_file)
        sys = build_system(config; coords = coords, velocities = velocities)
        return compute_pcoord(config, sys)
    end
end
