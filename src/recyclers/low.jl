export LowRecycler

"""
    LowRecycler(basis_states, target_threshold; pcoord_idx=1)

Recycle simulations whose progress coordinate falls *below* `target_threshold`.
"""
struct LowRecycler <: AbstractRecycler
    basis_states::BasisStates
    target_threshold::Float64
    pcoord_idx::Int
end

LowRecycler(basis_states::BasisStates, target_threshold::Real; pcoord_idx::Int=1) =
    LowRecycler(basis_states, Float64(target_threshold), pcoord_idx)

function recycle(r::LowRecycler, pcoords::AbstractMatrix)
    return findall(<(r.target_threshold), @view pcoords[:, r.pcoord_idx])
end
