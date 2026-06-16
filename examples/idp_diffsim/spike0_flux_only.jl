using Flux
using Enzyme

function loss(model, x)
    return sum(model(x))
end

function main()
    model = Chain(Dense(1, 5, relu), Dense(5, 1, tanh))
    x = Float32[1.0;;]

    d_model = Flux.fmap(model) do p
        p isa Array ? zero(p) : p
    end

    grad = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse), loss, Enzyme.Active,
        Enzyme.Duplicated(model, d_model),
        Enzyme.Const(x),
    )

    println("grad = ", grad)
    println("d_model = ", d_model)
end

main()
