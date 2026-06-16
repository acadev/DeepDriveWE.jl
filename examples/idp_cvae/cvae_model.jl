using Lux
using Random

const FEATURE_DIM = 36
const HIDDEN_DIM = 32
const LATENT_DIM = 2

struct Encoder <: Lux.AbstractLuxLayer
    fc1::Lux.Dense
    fc_mu::Lux.Dense
    fc_logvar::Lux.Dense
end

function Encoder()
    return Encoder(
        Dense(FEATURE_DIM => HIDDEN_DIM, tanh),
        Dense(HIDDEN_DIM => LATENT_DIM),
        Dense(HIDDEN_DIM => LATENT_DIM),
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, m::Encoder)
    return (
        fc1 = Lux.initialparameters(rng, m.fc1),
        fc_mu = Lux.initialparameters(rng, m.fc_mu),
        fc_logvar = Lux.initialparameters(rng, m.fc_logvar),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, m::Encoder)
    return (
        fc1 = Lux.initialstates(rng, m.fc1),
        fc_mu = Lux.initialstates(rng, m.fc_mu),
        fc_logvar = Lux.initialstates(rng, m.fc_logvar),
    )
end

function (m::Encoder)(x, ps, st)
    h, st_fc1 = m.fc1(x, ps.fc1, st.fc1)
    mu, st_mu = m.fc_mu(h, ps.fc_mu, st.fc_mu)
    logvar, st_logvar = m.fc_logvar(h, ps.fc_logvar, st.fc_logvar)
    return (mu, logvar), (fc1 = st_fc1, fc_mu = st_mu, fc_logvar = st_logvar)
end

function build_decoder()
    return Chain(
        Dense(LATENT_DIM => HIDDEN_DIM, tanh),
        Dense(HIDDEN_DIM => FEATURE_DIM),
    )
end
