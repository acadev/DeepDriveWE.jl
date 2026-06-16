export AbstractResampler, resample, run_resampling, get_next_sims, add_new_simulation!,
       split_sims, merge_sims, split_by_weight, merge_by_weight, adjust_count,
       split_by_threshold, merge_by_threshold, get_pcoords

"""
    AbstractResampler

Abstract type for resamplers that split/merge simulations within a bin to
maintain target walker counts and weight thresholds.

Concrete subtypes must have a mutable `index_counter::Int` field, used to
assign `simulation_id`s to newly created (split/merged) simulations.
`pcoord_idx` fields are 1-based indices into a progress-coordinate vector.
"""
abstract type AbstractResampler end

"""
    next_index!(r::AbstractResampler) -> Int

Return the next unique simulation id and increment the resampler's counter
(post-increment, starting from 0).
"""
function next_index!(r::AbstractResampler)
    val = r.index_counter
    r.index_counter += 1
    return val
end

"""
    resample(r::AbstractResampler, cur_sims, next_sims) -> (cur_sims, next_sims)

Resample the simulations within a single bin. Implemented by concrete
resamplers.
"""
function resample end

"""
    get_next_sims(cur_sims::Vector{SimMetadata}) -> Vector{SimMetadata}

Build the metadata for the next iteration's simulations from the completed
`cur_sims` (each of which must have `restart_file` set).
"""
function get_next_sims(cur_sims::Vector{SimMetadata})
    simulations = SimMetadata[]
    for (idx, sim) in enumerate(cur_sims)
        sim.restart_file === nothing && throw(ArgumentError(
            "Simulation $(sim.simulation_id) has no restart_file; " *
            "it must have been run before resampling.",
        ))

        push!(simulations, SimMetadata(;
            weight = sim.weight,
            simulation_id = idx,
            iteration_id = sim.iteration_id + 1,
            parent_restart_file = sim.restart_file,
            parent_pcoord = sim.pcoord[end],
            parent_simulation_id = sim.simulation_id,
            wtg_parent_ids = [sim.simulation_id],
        ))
    end
    return simulations
end

"""
    add_new_simulation!(r::AbstractResampler, sim, weight, wtg_parent_ids) -> SimMetadata

Create a new simulation (used by [`split_sims`](@ref) and [`merge_sims`](@ref))
inheriting `sim`'s parent/restart info but with a fresh `simulation_id`.
"""
function add_new_simulation!(
    r::AbstractResampler,
    sim::SimMetadata,
    weight::Float64,
    wtg_parent_ids::Vector{Int},
)
    return SimMetadata(;
        weight = weight,
        simulation_id = next_index!(r),
        iteration_id = sim.iteration_id,
        restart_file = sim.restart_file,
        parent_restart_file = sim.parent_restart_file,
        parent_pcoord = sim.parent_pcoord,
        parent_simulation_id = sim.parent_simulation_id,
        wtg_parent_ids = wtg_parent_ids,
    )
end

"""
    split_sims(r, sims, indices, n_splits=2) -> Vector{SimMetadata}

Split the simulations at (1-based) `indices` into `n_splits` copies each
(equally dividing the weight). `n_splits` may be a single `Int` or a
`Vector{Int}` matching `indices`.
"""
function split_sims(
    r::AbstractResampler,
    sims::Vector{SimMetadata},
    indices::Vector{Int},
    n_splits::Union{Int, Vector{Int}} = 2,
)
    sims_to_split = [sims[i] for i in indices]
    n_splits_vec = n_splits isa Int ? fill(n_splits, length(sims_to_split)) : n_splits

    index_set = Set(indices)
    new_sims = SimMetadata[sims[i] for i in eachindex(sims) if !(i in index_set)]

    for (sim, n_split) in zip(sims_to_split, n_splits_vec)
        for _ in 1:n_split
            push!(new_sims, add_new_simulation!(
                r, sim, sim.weight / n_split, copy(sim.wtg_parent_ids),
            ))
        end
    end

    return new_sims
end

"""
    merge_sims(r, cur_sims, next_sims, indices; rng=Random.default_rng()) -> Vector{SimMetadata}

Merge the simulations in `next_sims` at (1-based) `indices` into a single new
simulation, choosing the parent randomly with probability proportional to
weight. Mutates `cur_sims` in place, setting `endpoint_type = 2` for the
merged-away parents.
"""
function merge_sims(
    r::AbstractResampler,
    cur_sims::Vector{SimMetadata},
    next_sims::Vector{SimMetadata},
    indices::Vector{Int};
    rng::Random.AbstractRNG = Random.default_rng(),
)
    to_merge = [next_sims[i] for i in indices]
    weights = [sim.weight for sim in to_merge]
    norm_weights = weights ./ sum(weights)

    select = _weighted_choice(rng, norm_weights)

    wtg_parent_ids = collect(union((Set(sim.wtg_parent_ids) for sim in to_merge)...))

    new_sim = add_new_simulation!(r, to_merge[select], sum(weights), wtg_parent_ids)

    index_set = Set(indices)
    new_sims = SimMetadata[next_sims[i] for i in eachindex(next_sims) if !(i in index_set)]
    push!(new_sims, new_sim)

    merged_parents = Set(x.parent_simulation_id for x in to_merge)
    delete!(merged_parents, new_sim.parent_simulation_id)

    for sim in cur_sims
        if sim.simulation_id >= 0 && sim.simulation_id in merged_parents
            sim.endpoint_type = 2
        end
    end

    return new_sims
end

function _weighted_choice(rng::Random.AbstractRNG, probs::Vector{Float64})
    r = rand(rng)
    cum = 0.0
    for (i, p) in enumerate(probs)
        cum += p
        r <= cum && return i
    end
    return length(probs)
end

"""
    split_by_weight(r, sims, ideal_weight) -> Vector{SimMetadata}

Split every simulation whose weight exceeds `ideal_weight` into
`ceil(weight / ideal_weight)` copies.
"""
function split_by_weight(r::AbstractResampler, sims::Vector{SimMetadata}, ideal_weight::Float64)
    weights = [sim.weight for sim in sims]
    indices = findall(w -> w > ideal_weight, weights)
    num_splits = [ceil(Int, weights[i] / ideal_weight) for i in indices]
    return split_sims(r, sims, indices, num_splits)
end

"""
    merge_by_weight(r, cur_sims, next_sims, ideal_weight) -> Vector{SimMetadata}

Repeatedly merge the lowest-weight simulations together while their combined
weight stays below `ideal_weight`.
"""
function merge_by_weight(
    r::AbstractResampler,
    cur_sims::Vector{SimMetadata},
    next_sims::Vector{SimMetadata},
    ideal_weight::Float64,
)
    while true
        sorted_sims = sort(next_sims; by = sim -> sim.weight)
        weights = [sim.weight for sim in sorted_sims]
        cumul = cumsum(weights)

        to_merge = findall(<=(ideal_weight), cumul)

        length(to_merge) < 2 && return next_sims

        next_sims = merge_sims(r, cur_sims, sorted_sims, to_merge)
    end
end

"""
    adjust_count(r, cur_sims, next_sims, target_count) -> Vector{SimMetadata}

Split or merge simulations until `length(next_sims) == target_count`.
"""
function adjust_count(
    r::AbstractResampler,
    cur_sims::Vector{SimMetadata},
    next_sims::Vector{SimMetadata},
    target_count::Int,
)
    while length(next_sims) < target_count
        index = argmax([sim.weight for sim in next_sims])
        next_sims = split_sims(r, next_sims, [index], 2)
        length(next_sims) == target_count && break
    end

    while length(next_sims) > target_count
        sorted_indices = sortperm([sim.weight for sim in next_sims])
        indices = sorted_indices[1:2]
        next_sims = merge_sims(r, cur_sims, next_sims, indices)
        length(next_sims) == target_count && break
    end

    return next_sims
end

"""
    split_by_threshold(r, sims, max_allowed_weight) -> Vector{SimMetadata}

Split simulations exceeding `max_allowed_weight`.
"""
split_by_threshold(r::AbstractResampler, sims::Vector{SimMetadata}, max_allowed_weight::Float64) =
    split_by_weight(r, sims, max_allowed_weight)

"""
    merge_by_threshold(r, cur_sims, next_sims, min_allowed_weight) -> Vector{SimMetadata}

Repeatedly merge all simulations under `min_allowed_weight` into a single
simulation.
"""
function merge_by_threshold(
    r::AbstractResampler,
    cur_sims::Vector{SimMetadata},
    next_sims::Vector{SimMetadata},
    min_allowed_weight::Float64,
)
    while true
        sorted_sims = sort(next_sims; by = sim -> sim.weight)
        weights = [sim.weight for sim in sorted_sims]

        to_merge = findall(<(min_allowed_weight), weights)

        length(to_merge) < 2 && return next_sims

        next_sims = merge_sims(r, cur_sims, sorted_sims, to_merge)
    end
end

"""
    get_pcoords(next_sims, pcoord_idx=1) -> Vector{Float64}

Extract the `parent_pcoord[pcoord_idx]` value from each simulation.
"""
get_pcoords(next_sims::Vector{SimMetadata}, pcoord_idx::Int = 1) =
    [sim.parent_pcoord[pcoord_idx] for sim in next_sims]

"""
    run_resampling(r, cur_sims, binner, recycler) -> (cur_sims, new_sims, metadata)

Top-level weighted-ensemble resampling step: build next-iteration metadata,
recycle simulations crossing the target threshold, bin them, and resample
each bin.
"""
function run_resampling(
    r::AbstractResampler,
    cur_sims::Vector{SimMetadata},
    binner::AbstractBinner,
    recycler::AbstractRecycler,
)
    next_sims = get_next_sims(cur_sims)

    cur_sims, next_sims = recycle_simulations(recycler, cur_sims, next_sims)

    bins = bin_simulations(binner, next_sims)

    metadata = compute_iteration_metadata(binner, cur_sims)

    new_sims = SimMetadata[]
    for sim_indices in values(bins)
        binned_sims = [next_sims[i] for i in sim_indices]
        cur_sims, resampled_sims = resample(r, cur_sims, binned_sims)
        append!(new_sims, resampled_sims)
    end

    return cur_sims, new_sims, metadata
end
