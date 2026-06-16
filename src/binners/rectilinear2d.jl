export RectilinearBinner2D

"""
    RectilinearBinner2D(bins_x, bins_y, bin_target_count; target_state_inds=Int[], pcoord_idxs=(1, 2))

Two-dimensional rectilinear binner over a `(pcoord_idxs[1], pcoord_idxs[2])`
progress-coordinate pair, e.g. `(phi, psi)`.

# Arguments
- `bins_x`, `bins_y::Vector{Float64}`: bin edges along each dimension, sorted
  in ascending order.
- `bin_target_count::Int`: target walker count for every occupied cell (with
  `target_state_inds` zeroed out, since those bins are recycled).
- `target_state_inds::Vector{Int}`: 1-based linear bin indices that
  correspond to target states (target count 0).
- `pcoord_idxs::Tuple{Int, Int}`: 1-based indices into the progress-coordinate
  vector for the x and y dimensions.

Bins are laid out on a `nx x ny` grid, with linear bin index
`(iy - 1) * nx + ix` for grid cell `(ix, iy)`, `ix in 1:nx`, `iy in 1:ny`.
"""
mutable struct RectilinearBinner2D <: AbstractBinner
    bins_x::Vector{Float64}
    bins_y::Vector{Float64}
    bin_target_count::Int
    target_state_inds::Vector{Int}
    pcoord_idxs::Tuple{Int, Int}
    bin_target_counts::Union{Nothing, Vector{Int}}

    function RectilinearBinner2D(
        bins_x::AbstractVector{<:Real},
        bins_y::AbstractVector{<:Real},
        bin_target_count::Int;
        target_state_inds::Vector{Int} = Int[],
        pcoord_idxs::Tuple{Int, Int} = (1, 2),
    )
        bins_x = Float64.(bins_x)
        bins_y = Float64.(bins_y)
        if !all(diff(bins_x) .> 0) || !all(diff(bins_y) .> 0)
            throw(ArgumentError("Bins must be sorted in ascending order."))
        end
        new(bins_x, bins_y, bin_target_count, target_state_inds, pcoord_idxs, nothing)
    end
end

nx(b::RectilinearBinner2D) = length(b.bins_x) - 1
ny(b::RectilinearBinner2D) = length(b.bins_y) - 1
nbins(b::RectilinearBinner2D) = nx(b) * ny(b)

function get_bin_target_counts(b::RectilinearBinner2D)
    if b.bin_target_counts === nothing
        counts = fill(b.bin_target_count, nbins(b))
        for i in b.target_state_inds
            counts[i] = 0
        end
        b.bin_target_counts = counts
    end
    return b.bin_target_counts
end

function assign_bins(b::RectilinearBinner2D, pcoords::AbstractMatrix)
    ix_idx, iy_idx = b.pcoord_idxs
    n, m = nx(b), ny(b)
    return [
        begin
            ix = min(digitize_right(pcoords[i, ix_idx], b.bins_x), n)
            iy = min(digitize_right(pcoords[i, iy_idx], b.bins_y), m)
            (iy - 1) * n + ix
        end
        for i in axes(pcoords, 1)
    ]
end

StructTypes.StructType(::Type{RectilinearBinner2D}) = StructTypes.Mutable()
