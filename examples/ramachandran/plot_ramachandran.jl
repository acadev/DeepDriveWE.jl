using JLD2
using Plots
using Statistics

function main()
    outdir = mkpath(joinpath(@__DIR__, "output"))

    we_data = JLD2.load(joinpath(outdir, "we_data.jld2"))
    md_data = JLD2.load(joinpath(outdir, "md_data.jld2"))

    we_records = we_data["records"]
    md_records = md_data["records"]

    rad2deg_ = x -> x * 180 / pi

    we_phi = rad2deg_.([r.phi for r in we_records])
    we_psi = rad2deg_.([r.psi for r in we_records])
    we_w = [r.weight for r in we_records]

    md_phi = rad2deg_.([r.phi for r in md_records])
    md_psi = rad2deg_.([r.psi for r in md_records])

    edges = -180:6:180

    p1 = histogram2d(
        we_phi, we_psi;
        weights = we_w,
        bins = (edges, edges),
        normalize = :probability,
        color = :viridis,
        xlabel = "phi (deg)",
        ylabel = "psi (deg)",
        title = "Weighted Ensemble (80 iter, 582k steps)",
        xlims = (-180, 180),
        ylims = (-180, 180),
        colorbar_title = "weight",
    )

    p2 = histogram2d(
        md_phi, md_psi;
        bins = (edges, edges),
        normalize = :probability,
        color = :viridis,
        xlabel = "phi (deg)",
        ylabel = "psi (deg)",
        title = "Plain MD baseline (582k steps)",
        xlims = (-180, 180),
        ylims = (-180, 180),
        colorbar_title = "probability",
    )

    plt = plot(p1, p2; layout = (1, 2), size = (1100, 480), margin = 5Plots.mm)
    savefig(plt, joinpath(outdir, "ramachandran.png"))
    println("Saved ", joinpath(outdir, "ramachandran.png"))

    # Scatter overlay showing WE walker coverage vs MD trajectory
    p3 = scatter(
        md_phi, md_psi;
        markersize = 1.5, markerstrokewidth = 0, alpha = 0.3, label = "plain MD",
        xlabel = "phi (deg)", ylabel = "psi (deg)",
        xlims = (-180, 180), ylims = (-180, 180),
        title = "WE walkers vs plain MD trajectory",
        size = (600, 550),
    )
    scatter!(
        p3, we_phi, we_psi;
        markersize = 3.0, markerstrokewidth = 0, alpha = 0.6,
        marker_z = log10.(we_w), color = :plasma, label = "WE walkers (log10 weight)",
    )
    savefig(p3, joinpath(outdir, "ramachandran_scatter.png"))
    println("Saved ", joinpath(outdir, "ramachandran_scatter.png"))

    return nothing
end

main()
