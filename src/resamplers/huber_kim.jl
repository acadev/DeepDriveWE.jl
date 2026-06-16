export HuberKimResampler

"""
    HuberKimResampler(; sims_per_bin=5, max_allowed_weight=1.0, min_allowed_weight=10e-40)

Huber & Kim (1996) resampling, with WESTPA-style fixed-bin adjust-count and
weight-threshold steps:

1. Resample based on weight (split/merge towards `ideal_weight = total_weight / sims_per_bin`).
2. Adjust the number of simulations in the bin to `sims_per_bin`.
3. Split/merge to keep simulations within `[min_allowed_weight, max_allowed_weight]`.
"""
mutable struct HuberKimResampler <: AbstractResampler
    sims_per_bin::Int
    max_allowed_weight::Float64
    min_allowed_weight::Float64
    index_counter::Int
end

HuberKimResampler(;
    sims_per_bin::Int = 5,
    max_allowed_weight::Real = 1.0,
    min_allowed_weight::Real = 10e-40,
) = HuberKimResampler(sims_per_bin, Float64(max_allowed_weight), Float64(min_allowed_weight), 0)

function resample(
    r::HuberKimResampler,
    cur_sims::Vector{SimMetadata},
    next_sims::Vector{SimMetadata},
)
    cur = deepcopy(cur_sims)
    nxt = deepcopy(next_sims)

    weights = [sim.weight for sim in nxt]
    ideal_weight = sum(weights) / r.sims_per_bin

    nxt = split_by_weight(r, nxt, ideal_weight)
    nxt = merge_by_weight(r, cur, nxt, ideal_weight)
    nxt = adjust_count(r, cur, nxt, r.sims_per_bin)
    nxt = split_by_threshold(r, nxt, r.max_allowed_weight)
    nxt = merge_by_threshold(r, cur, nxt, r.min_allowed_weight)

    return cur, nxt
end
