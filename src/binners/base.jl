export AbstractBinner, nbins, get_bin_target_counts, assign_bins, bin_labels,
       bin_assignments, bin_probs, compute_iteration_metadata, bin_simulations,
       digitize_right

"""
    AbstractBinner

Abstract type for binners that assign simulations to bins based on their
progress coordinate.

# Indexing convention

Bin indices are **1-based**, in the range `1:nbins(binner)+1` for
[`digitize_right`](@ref) (the +1 only occurs for out-of-range progress
coordinates). `bin_target_counts` is a `Vector{Int}` of length `nbins(binner)`
indexed `1:nbins(binner)`. `pcoord_idx` fields are 1-based indices into a
progress-coordinate vector.
"""
abstract type AbstractBinner end

"""
    nbins(b::AbstractBinner) -> Int

The number of bins.
"""
function nbins end

"""
    get_bin_target_counts(b::AbstractBinner) -> Vector{Int}

The target walker counts for each bin.
"""
function get_bin_target_counts end

"""
    assign_bins(b::AbstractBinner, pcoords::AbstractMatrix) -> Vector{Int}

Assign each row of `pcoords` (shape `(n_simulations, n_dims)`) to a bin index.
"""
function assign_bins end

"""
    bin_labels(b::AbstractBinner) -> Vector{String}

WESTPA-style bin labels, e.g. `["state1", "state2", ...]`.
"""
bin_labels(b::AbstractBinner) = ["state$i" for i in 1:nbins(b)]

"""
    digitize_right(x, bins) -> Int

1-based bin index for `x` given bin edges `bins` (length `nbins + 1`),
matching `numpy.digitize(x, bins, right=True)` semantics shifted so that bin
`k` (`1 <= k <= nbins`) covers `bins[k] < x <= bins[k+1]`. Values `x <=
bins[1]` fall into bin `1`, and values `x > bins[end]` return `nbins + 1`
(out of range).
"""
digitize_right(x::Real, bins::AbstractVector{<:Real}) =
    max(searchsortedfirst(bins, x) - 1, 1)

"""
    bin_assignments(b::AbstractBinner, pcoords::AbstractMatrix) -> Dict{Int, Vector{Int}}

Map bin index to the (1-based) simulation indices assigned to that bin.
"""
function bin_assignments(b::AbstractBinner, pcoords::AbstractMatrix)
    assignments = assign_bins(b, pcoords)

    if length(assignments) != size(pcoords, 1)
        throw(ArgumentError(
            "Number of assignments must match the number of simulations.",
        ))
    end

    result = Dict{Int, Vector{Int}}()
    for (sim_idx, bin_idx) in enumerate(assignments)
        push!(get!(result, bin_idx, Int[]), sim_idx)
    end
    return result
end

"""
    bin_probs(assignments, cur_sims) -> Vector{Float64}

The total weight (probability) in each bin.
"""
function bin_probs(assignments::Dict{Int, Vector{Int}}, cur_sims::Vector{SimMetadata})
    return [sum(cur_sims[i].weight for i in idxs) for idxs in values(assignments)]
end

"""
    compute_iteration_metadata(b::AbstractBinner, cur_sims) -> IterationMetadata

Compute the [`IterationMetadata`](@ref) for the current iteration using the
last frame's progress coordinate of each simulation.
"""
function compute_iteration_metadata(b::AbstractBinner, cur_sims::Vector{SimMetadata})
    pcoords = _pcoord_matrix(sim -> sim.pcoord[end], cur_sims)

    assignments = bin_assignments(b, pcoords)
    probs = bin_probs(assignments, cur_sims)

    binner_hash = bytes2hex(sha256(JSON3.write(b)))

    return IterationMetadata(;
        iteration_id = cur_sims[1].iteration_id,
        binner_hash = binner_hash,
        min_bin_prob = minimum(probs),
        max_bin_prob = maximum(probs),
        bin_target_counts = get_bin_target_counts(b),
    )
end

"""
    bin_simulations(b::AbstractBinner, next_sims) -> Dict{Int, Vector{Int}}

Assign the next-iteration simulations to bins based on `parent_pcoord`.
"""
function bin_simulations(b::AbstractBinner, next_sims::Vector{SimMetadata})
    pcoords = _pcoord_matrix(sim -> sim.parent_pcoord, next_sims)
    return bin_assignments(b, pcoords)
end

function _pcoord_matrix(getter, sims::Vector{SimMetadata})
    n = length(sims)
    d = length(getter(sims[1]))
    mat = Matrix{Float64}(undef, n, d)
    for (i, sim) in enumerate(sims)
        mat[i, :] .= getter(sim)
    end
    return mat
end
