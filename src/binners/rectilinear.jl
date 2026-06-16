export RectilinearBinner

"""
    RectilinearBinner(bins, bin_target_counts; target_state_inds=[1], pcoord_idx=1)

Rectilinear binner for the progress coordinate.

# Arguments
- `bins::Vector{Float64}`: the bin edges, sorted in ascending order.
- `bin_target_counts::Union{Int, Vector{Int}}`: the target walker count for
  each bin. If an `Int`, the same target count is used for every bin (with
  `target_state_inds` zeroed out, since those bins are recycled).
- `target_state_inds::Vector{Int}`: 1-based bin indices that correspond to
  target states (recycled, target count 0). Only used when
  `bin_target_counts` is an `Int`.
- `pcoord_idx::Int`: 1-based index into the progress-coordinate vector to use
  for binning.
"""
mutable struct RectilinearBinner <: AbstractBinner
    bins::Vector{Float64}
    bin_target_counts::Union{Int, Vector{Int}}
    target_state_inds::Vector{Int}
    pcoord_idx::Int

    function RectilinearBinner(
        bins::AbstractVector{<:Real},
        bin_target_counts::Union{Int, Vector{Int}};
        target_state_inds::Union{Int, Vector{Int}} = [1],
        pcoord_idx::Int = 1,
    )
        bins = Float64.(bins)
        if !all(diff(bins) .> 0)
            throw(ArgumentError("Bins must be sorted in ascending order."))
        end
        inds = target_state_inds isa Int ? [target_state_inds] : target_state_inds
        new(bins, bin_target_counts, inds, pcoord_idx)
    end
end

nbins(b::RectilinearBinner) = length(b.bins) - 1

function get_bin_target_counts(b::RectilinearBinner)
    if b.bin_target_counts isa Int
        counts = fill(b.bin_target_counts, nbins(b))
        for i in b.target_state_inds
            counts[i] = 0
        end
        b.bin_target_counts = counts
    end
    return b.bin_target_counts
end

function assign_bins(b::RectilinearBinner, pcoords::AbstractMatrix)
    return [digitize_right(pcoords[i, b.pcoord_idx], b.bins) for i in axes(pcoords, 1)]
end

StructTypes.StructType(::Type{RectilinearBinner}) = StructTypes.Mutable()
