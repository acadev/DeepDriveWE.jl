using DeepDriveWE
using JLD2

include("cvae_model.jl")

"""
    CVAELatentConfig(physical, cvae_path)

Wrapper around an [`IDPFragmentConfig`](@ref) (`physical`) that overrides the
progress coordinate with the 2D latent code of a trained CVAE encoder loaded
from `cvae_path` (as saved by `train_cvae.jl`).

All MD plumbing (`build_system`, `init_basis_state!`, `run_segment!`, ...) is
inherited generically via [`DeepDriveWE.physical_config`](@ref).
"""
struct CVAELatentConfig
    physical::IDPFragmentConfig
    encoder::Encoder
    ps::NamedTuple
    st::NamedTuple
    mu_x::Matrix{Float32}
    sigma_x::Matrix{Float32}
end

function CVAELatentConfig(physical::IDPFragmentConfig, cvae_path::AbstractString)
    cvae = JLD2.load(cvae_path)
    return CVAELatentConfig(
        physical,
        cvae["encoder"],
        cvae["ps_encoder"],
        cvae["st_encoder"],
        cvae["mu_x"],
        cvae["sigma_x"],
    )
end

DeepDriveWE.physical_config(c::CVAELatentConfig) = c.physical

function DeepDriveWE.compute_pcoord(c::CVAELatentConfig, sys)
    pcoord = compute_pcoord(c.physical, sys)
    features = Float32.(dihedral_features(pcoord))
    features_norm = (features .- vec(c.mu_x)) ./ vec(c.sigma_x)
    (mu, _), _ = c.encoder(reshape(features_norm, :, 1), c.ps, c.st)
    return Float64.(vec(mu))
end
