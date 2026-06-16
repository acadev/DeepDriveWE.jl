export SplitLowResampler

"""
    SplitLowResampler(; num_resamples=1, n_split=2, pcoord_idx=1)

Split the simulation(s) with the lowest progress coordinate, and merge the
simulations with the highest progress coordinate to keep the walker count
constant.
"""
mutable struct SplitLowResampler <: AbstractResampler
    num_resamples::Int
    n_split::Int
    pcoord_idx::Int
    index_counter::Int
end

SplitLowResampler(; num_resamples::Int = 1, n_split::Int = 2, pcoord_idx::Int = 1) =
    SplitLowResampler(num_resamples, n_split, pcoord_idx, 0)

function _split_low(r::SplitLowResampler, next_sims::Vector{SimMetadata})
    pcoords = get_pcoords(next_sims, r.pcoord_idx)
    sorted_indices = sortperm(pcoords)
    indices = sorted_indices[1:r.num_resamples]
    return split_sims(r, next_sims, indices, r.n_split)
end

function _merge_low(
    r::SplitLowResampler,
    cur_sims::Vector{SimMetadata},
    next_sims::Vector{SimMetadata},
)
    pcoords = get_pcoords(next_sims, r.pcoord_idx)
    sorted_indices = sortperm(pcoords)

    # n_split new simulations were created from the split, so n_split - 1 + 1
    # = num_resamples + 1 simulations need to be merged to keep the count
    # constant (matches upstream's `num_resamples + 1`).
    num_merges = r.num_resamples + 1
    indices = sorted_indices[(end - num_merges + 1):end]
    return merge_sims(r, cur_sims, next_sims, indices)
end

function resample(
    r::SplitLowResampler,
    cur_sims::Vector{SimMetadata},
    next_sims::Vector{SimMetadata},
)
    cur = deepcopy(cur_sims)
    nxt = deepcopy(next_sims)

    nxt = _split_low(r, nxt)
    nxt = _merge_low(r, cur, nxt)

    return cur, nxt
end
