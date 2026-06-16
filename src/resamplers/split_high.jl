export SplitHighResampler

"""
    SplitHighResampler(; num_resamples=1, n_split=2, pcoord_idx=1)

Split the simulation(s) with the highest progress coordinate, and merge the
simulations with the lowest progress coordinate to keep the walker count
constant.
"""
mutable struct SplitHighResampler <: AbstractResampler
    num_resamples::Int
    n_split::Int
    pcoord_idx::Int
    index_counter::Int
end

SplitHighResampler(; num_resamples::Int = 1, n_split::Int = 2, pcoord_idx::Int = 1) =
    SplitHighResampler(num_resamples, n_split, pcoord_idx, 0)

function _split_high(r::SplitHighResampler, next_sims::Vector{SimMetadata})
    pcoords = get_pcoords(next_sims, r.pcoord_idx)
    sorted_indices = sortperm(pcoords)
    indices = sorted_indices[(end - r.num_resamples + 1):end]
    return split_sims(r, next_sims, indices, r.n_split)
end

function _merge_high(
    r::SplitHighResampler,
    cur_sims::Vector{SimMetadata},
    next_sims::Vector{SimMetadata},
)
    pcoords = get_pcoords(next_sims, r.pcoord_idx)
    sorted_indices = sortperm(pcoords)

    num_merges = r.num_resamples + 1
    indices = sorted_indices[1:num_merges]
    return merge_sims(r, cur_sims, next_sims, indices)
end

function resample(
    r::SplitHighResampler,
    cur_sims::Vector{SimMetadata},
    next_sims::Vector{SimMetadata},
)
    cur = deepcopy(cur_sims)
    nxt = deepcopy(next_sims)

    nxt = _split_high(r, nxt)
    nxt = _merge_high(r, cur, nxt)

    return cur, nxt
end
