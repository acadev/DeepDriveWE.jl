using DeepDriveWE
using Molly
using Random
using Unitful
using JLD2

"""
Plain MD baseline for the 5AWL IDP fragment, matching the total step count of
`run_we_driver.jl`. Records the full 18-element backbone dihedral pcoord for
each 1 ps segment.
"""
function main()
    rng = Random.MersenneTwister(123)
    config = IDPFragmentConfig(
        n_steps = 500,           # 1 ps segments, matches the WE run
        dt = 0.002u"ps",
        temperature = 300.0u"K",
        friction = 1.0u"ps^-1",
    )

    outdir = mkpath(joinpath(@__DIR__, "output_driver"))
    we_data = JLD2.load(joinpath(outdir, "we_data.jld2"))
    total_steps = we_data["total_steps"]
    n_segments = total_steps ÷ config.n_steps

    println("Running plain MD baseline: $n_segments segments x $(config.n_steps) steps = $(n_segments * config.n_steps) steps")

    sys = build_system(config)
    random_velocities!(sys, config.temperature; rng = rng)

    simulator = Langevin(dt = config.dt, temperature = config.temperature, friction = config.friction)
    simulate!(sys, simulator, 5000; rng = rng)  # equilibration, matches WE basis state

    records = NamedTuple{(:pcoord,), Tuple{Vector{Float64}}}[]
    for i in 1:n_segments
        simulate!(sys, simulator, config.n_steps; rng = rng)
        push!(records, (pcoord = compute_pcoord(config, sys),))
        if i % 100 == 0
            println("segment $i / $n_segments")
        end
    end

    JLD2.jldsave(joinpath(outdir, "md_data.jld2"); records = records)
    return records
end

main()
