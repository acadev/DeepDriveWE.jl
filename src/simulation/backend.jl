export run_segment!, init_basis_state!, simulation_steps, compute_pcoord, physical_config

"""
    Simulation backend interface

A "simulation config" (e.g. [`AlanineDipeptideConfig`](@ref) or
[`IDPFragmentConfig`](@ref)) plugs a physical system into the
[`run_we!`](@ref) driver via the generic implementations in
`simulation/common.jl`, plus one required method:

- `compute_pcoord(config, sys) -> Vector{Float64}`: the progress coordinate
  for a Molly `System` `sys`, e.g. backbone dihedral angles.

`simulation/common.jl` then provides, generically (via
[`physical_config`](@ref)):

- `build_system(config; coords=nothing, velocities=nothing)`
- `init_basis_state!(config, path; rng, n_equil_steps) -> Vector{Float64}`
- `run_segment!(config, sim::SimMetadata, restart_path; rng) -> SimMetadata`
- `basis_state_initializer(config) -> Function`
- `simulation_steps(config) -> Int`, the number of integration steps per
  segment (`physical_config(config).n_steps`).

A wrapper config that swaps in a different (e.g. learned) progress coordinate
on top of an existing physical config should override
[`physical_config`](@ref) and `compute_pcoord`, and the rest of the interface
is inherited for free.
"""
function compute_pcoord end

"""
    physical_config(config) -> config

Return the underlying "physical" simulation config (with `pdb_file`,
`ff_files`, `dt`, `temperature`, `friction`, `n_steps` fields) for `config`.
Defaults to `config` itself.
"""
physical_config(config) = config

simulation_steps(config) = physical_config(config).n_steps
