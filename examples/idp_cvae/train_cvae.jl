using DeepDriveWE
using JLD2
using Lux
using Optimisers
using Random
using Statistics
using Zygote

include("cvae_model.jl")

"""
Train a small convolutional-free VAE ("CVAE") on the backbone-dihedral
features collected by `examples/idp_fragment/run_we_driver.jl`.

Each 18-element pcoord (9 phi + 9 psi angles, radians) is mapped to a
36-element periodicity-safe feature vector via `dihedral_features`
(`[sin θ_1, cos θ_1, ..., sin θ_n, cos θ_n]`). The encoder maps these 36
features to a 2D latent space; the decoder reconstructs the 36 features from
the latent code. The trained encoder weights are saved to `cvae.jld2` for use
as a learned progress coordinate in the CVAE-driven WE run.
"""

function load_features(path::AbstractString)
    data = JLD2.load(path)
    records = data["records"]
    features = reduce(hcat, dihedral_features(r.pcoord) for r in records)
    return Float32.(features)  # FEATURE_DIM x N
end

# --- VAE loss -----------------------------------------------------------

function vae_loss(encoder, decoder, ps, st, x, rng)
    (mu, logvar), st_enc = encoder(x, ps.encoder, st.encoder)
    eps = randn(rng, Float32, size(mu))
    z = mu .+ exp.(0.5f0 .* logvar) .* eps

    x_hat, st_dec = decoder(z, ps.decoder, st.decoder)

    recon_loss = mean(sum((x_hat .- x) .^ 2; dims = 1))
    kl_loss = mean(-0.5f0 .* sum(1 .+ logvar .- mu .^ 2 .- exp.(logvar); dims = 1))

    st_new = (encoder = st_enc, decoder = st_dec)
    return recon_loss + 0.1f0 * kl_loss, st_new, recon_loss, kl_loss
end

function main()
    rng = Random.MersenneTwister(42)

    data_path = joinpath(@__DIR__, "..", "idp_fragment", "output_driver", "we_data.jld2")
    features = load_features(data_path)
    println("Loaded features: ", size(features))

    # Normalize per-feature (sin/cos already in [-1, 1], but standardize anyway)
    mu_x = mean(features; dims = 2)
    sigma_x = std(features; dims = 2) .+ 1f-6
    features_norm = (features .- mu_x) ./ sigma_x

    encoder = Encoder()
    decoder = build_decoder()

    ps_enc, st_enc = Lux.setup(rng, encoder)
    ps_dec, st_dec = Lux.setup(rng, decoder)
    ps = (encoder = ps_enc, decoder = ps_dec)
    st = (encoder = st_enc, decoder = st_dec)

    opt = Optimisers.Adam(1f-3)
    opt_state = Optimisers.setup(opt, ps)

    n_epochs = 500
    for epoch in 1:n_epochs
        loss, st, recon_loss, kl_loss = nothing, st, nothing, nothing
        grads = Zygote.gradient(ps) do p
            l, st_new, _, _ = vae_loss(encoder, decoder, p, st, features_norm, rng)
            st = st_new
            l
        end
        opt_state, ps = Optimisers.update(opt_state, ps, grads[1])

        if epoch % 50 == 0 || epoch == 1
            l, _, recon_loss, kl_loss = vae_loss(encoder, decoder, ps, st, features_norm, rng)
            println("epoch $epoch: loss = $(round(l, digits=4)), recon = $(round(recon_loss, digits=4)), kl = $(round(kl_loss, digits=4))")
        end
    end

    outpath = joinpath(@__DIR__, "cvae.jld2")
    JLD2.jldsave(
        outpath;
        encoder = encoder,
        ps_encoder = ps.encoder,
        st_encoder = st.encoder,
        mu_x = mu_x,
        sigma_x = sigma_x,
    )
    println("Saved trained encoder to $outpath")
    return ps, st
end

main()
