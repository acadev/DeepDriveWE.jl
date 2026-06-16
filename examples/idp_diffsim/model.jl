using DeepDriveWE
using Molly
using AtomsCalculators
using LinearAlgebra
using Functors
using NNlib: relu

"""
Track A architecture for alanine dipeptide: a shared trunk operating on
backbone-dihedral features (`dihedral_features(compute_pcoord(config, sys))`,
4-dim for alanine: `[sin phi, cos phi, sin psi, cos psi]`), with two heads:

- `cv_latent`: 2D latent CV (here trained to recover `[phi, psi]` itself,
  i.e. a representation-learning sanity check - alanine's CVs are already
  2D, so the "head" learns an invertible-ish encoding of them).
- `correction_magnitudes`: a scalar force-magnitude correction per backbone
  atom pair in `pairs`, applied as a [`NNCorrection2`](@ref) `general_inter`
  (same pattern as the validated `spike_alanine.jl` `NNCorrection`, but the
  magnitude is now a function of the full dihedral-feature embedding rather
  than a single pairwise distance).

Implemented as hand-rolled dense layers (not `Flux.Chain`) - a `Flux.Chain`
`general_inter` triggers an Enzyme "Type Module does not have a definite
size" compile error under `simulate!` in this Julia 1.12 / Enzyme 0.13 / Flux
0.16 combination (see `spike0_minimal.jl` vs `spike0_handrolled_mlp.jl`).
`Functors.@functor` makes the struct compatible with `Optimisers.jl`.
"""
struct TrunkHeads{T}
    Wt::Matrix{T}
    bt::Vector{T}
    Wcv::Matrix{T}
    bcv::Vector{T}
    Wc::Matrix{T}
    bc::Vector{T}
end

Functors.@functor TrunkHeads

function TrunkHeads(::Type{T}, n_features::Int, n_hidden::Int, n_pairs::Int) where T
    return TrunkHeads(
        T(0.1) .* randn(T, n_hidden, n_features), zeros(T, n_hidden),
        T(0.1) .* randn(T, 2, n_hidden), zeros(T, 2),
        T(0.1) .* randn(T, n_pairs, n_hidden), zeros(T, n_pairs),
    )
end

trunk_embed(m::TrunkHeads, features) = relu.(m.Wt * features .+ m.bt)
cv_latent(m::TrunkHeads, h) = m.Wcv * h .+ m.bcv
correction_magnitudes(m::TrunkHeads, h) = tanh.(m.Wc * h .+ m.bc)

function (m::TrunkHeads)(features)
    h = trunk_embed(m, features)
    return cv_latent(m, h), correction_magnitudes(m, h)
end

"""
    NNCorrection2{T, C}

`general_inter` that adds a learned force correction along each backbone atom
pair in `pairs`, with magnitudes given by `TrunkHeads.correction_magnitudes`
applied to the current dihedral-feature embedding (computed from
`compute_pcoord(config, sys)`).
"""
struct NNCorrection2{T, C}
    model::TrunkHeads{T}
    config::C
    pairs::Vector{Tuple{Int, Int}}
    scale::Float64
end

function AtomsCalculators.forces!(fs, sys, inter::NNCorrection2; kwargs...)
    pcoord = compute_pcoord(inter.config, sys)
    features = dihedral_features(pcoord)
    h = trunk_embed(inter.model, features)
    mags = inter.scale .* correction_magnitudes(inter.model, h)

    for (idx, (i, j)) in enumerate(inter.pairs)
        vec_ij = vector(sys.coords[i], sys.coords[j], sys.boundary)
        f = mags[idx] * normalize(vec_ij)
        fs[i] = fs[i] .+ f
        fs[j] = fs[j] .- f
    end
    return fs
end

"""
    backbone_pairs(config::AlanineDipeptideConfig)

The two backbone atom pairs (phi-spanning and psi-spanning) that
[`NNCorrection2`](@ref) applies its learned force correction along.
"""
backbone_pairs(::AlanineDipeptideConfig) = [
    (DeepDriveWE.PHI_ATOMS[1], DeepDriveWE.PHI_ATOMS[4]),
    (DeepDriveWE.PSI_ATOMS[1], DeepDriveWE.PSI_ATOMS[4]),
]
