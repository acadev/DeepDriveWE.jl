export AbstractRecycler, recycle, recycle_simulations

"""
    AbstractRecycler

Abstract type for recyclers that detect simulations which have crossed a
target-state boundary and restart them from a basis state.

`pcoord_idx` fields on concrete recyclers are 1-based indices into a
progress-coordinate vector, matching [`AbstractBinner`](@ref)'s convention.
"""
abstract type AbstractRecycler end

"""
    recycle(r::AbstractRecycler, pcoords::AbstractMatrix) -> Vector{Int}

Return the (1-based) simulation indices to recycle, given `pcoords` of shape
`(n_simulations, n_dims)`.
"""
function recycle end

"""
    recycle_simulations(r::AbstractRecycler, cur_sims, next_sims) -> (cur_sims, next_sims)

Recycle simulations that have crossed the target threshold: their entry in
`next_sims` is replaced with a fresh simulation restarted from a randomly
chosen basis state, and the corresponding `cur_sims` entry has its
`endpoint_type` set to 3 (recycled).
"""
function recycle_simulations(
    r::AbstractRecycler,
    cur_sims::Vector{SimMetadata},
    next_sims::Vector{SimMetadata},
)
    pcoords = _pcoord_matrix(sim -> sim.pcoord[end], cur_sims)

    recycle_inds = recycle(r, pcoords)

    _next_sims = deepcopy(next_sims)
    _cur_sims = deepcopy(cur_sims)

    for idx in recycle_inds
        sim = _next_sims[idx]

        basis_state = rand(r.basis_states.basis_states)

        new_sim = SimMetadata(;
            weight = sim.weight,
            simulation_id = sim.simulation_id,
            iteration_id = sim.iteration_id,
            parent_restart_file = basis_state.parent_restart_file,
            parent_pcoord = basis_state.parent_pcoord,
            # Negate (and offset by 1) the simulation id to indicate that
            # this simulation was recycled from a basis state.
            parent_simulation_id = -(sim.simulation_id),
            wtg_parent_ids = sim.wtg_parent_ids,
        )

        _next_sims[idx] = new_sim
        _cur_sims[idx].endpoint_type = 3
    end

    return _cur_sims, _next_sims
end
