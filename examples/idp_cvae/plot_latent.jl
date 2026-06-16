using DeepDriveWE
using JLD2
using Plots
using StatsBase

include("cvae_model.jl")

"""
Visualize the CVAE latent space:

1. Coverage heatmaps: CVAE-driven WE (`output_cvae/we_data.jld2`) vs the
   plain MD baseline (`../idp_fragment/output_driver/md_data.jld2`), both
   encoded into the trained 2D latent space.
2. Mapping back to physical dihedrals: scatter of the latent space colored by
   the (phi_2, psi_1) backbone dihedral angles, showing what physical motion
   each latent dimension captures.
"""
function encode(encoder, ps, st, mu_x, sigma_x, pcoords)
    features = reduce(hcat, dihedral_features(p) for p in pcoords)
    features_norm = (Float32.(features) .- mu_x) ./ sigma_x
    (mu, _), _ = encoder(features_norm, ps, st)
    return mu  # 2 x N
end

function main()
    cvae = JLD2.load(joinpath(@__DIR__, "cvae.jld2"))
    encoder, ps, st, mu_x, sigma_x = cvae["encoder"], cvae["ps_encoder"], cvae["st_encoder"], cvae["mu_x"], cvae["sigma_x"]

    we_cvae = JLD2.load(joinpath(@__DIR__, "output_cvae", "we_data.jld2"))["records"]
    md_data = JLD2.load(joinpath(@__DIR__, "..", "idp_fragment", "output_driver", "md_data.jld2"))["records"]

    we_latent = reduce(hcat, r.latent for r in we_cvae)
    we_w = [r.weight for r in we_cvae]

    md_pcoords = [r.pcoord for r in md_data]
    md_latent = encode(encoder, ps, st, mu_x, sigma_x, md_pcoords)

    edges = -5:0.4:5

    we_hist = fit(Histogram, (we_latent[1, :], we_latent[2, :]), Weights(we_w), (edges, edges))
    md_hist = fit(Histogram, (md_latent[1, :], md_latent[2, :]), (edges, edges))
    md_prob = md_hist.weights ./ sum(md_hist.weights)

    floor_val = 1e-8
    we_log = log10.(max.(we_hist.weights, floor_val))
    md_log = log10.(max.(md_prob, floor_val))

    centers = collect(edges)[1:end-1] .+ step(edges) / 2

    p1 = heatmap(
        centers, centers, we_log';
        color = :viridis,
        xlabel = "z1", ylabel = "z2",
        title = "CVAE-driven WE (100 iter, latent-space binning)",
        xlims = (-5, 5), ylims = (-5, 5),
        clims = (-8, -1),
        colorbar_title = "log10(weight)",
    )

    p2 = heatmap(
        centers, centers, md_log';
        color = :viridis,
        xlabel = "z1", ylabel = "z2",
        title = "Plain MD baseline (same total steps as data-collection WE)",
        xlims = (-5, 5), ylims = (-5, 5),
        clims = (-8, -1),
        colorbar_title = "log10(probability)",
    )

    plt = plot(p1, p2; layout = (1, 2), size = (1150, 480), margin = 5Plots.mm)
    outdir = mkpath(joinpath(@__DIR__, "output_cvae"))
    savefig(plt, joinpath(outdir, "latent_coverage.png"))
    println("Saved ", joinpath(outdir, "latent_coverage.png"))

    # Map latent space back to physical backbone dihedrals (phi_2, psi_1 -
    # pcoord indices 1 and 10).
    we_pcoords = [r.pcoord for r in we_cvae]
    we_phi2 = rad2deg.([p[1] for p in we_pcoords])
    we_psi1 = rad2deg.([p[10] for p in we_pcoords])

    p3 = scatter(
        we_latent[1, :], we_latent[2, :];
        marker_z = we_phi2, color = :twilight,
        markersize = 4, markerstrokewidth = 0, label = false,
        xlabel = "z1", ylabel = "z2",
        title = "Latent space colored by phi_2 (deg)",
        xlims = (-5, 5), ylims = (-5, 5),
        colorbar_title = "phi_2 (deg)",
    )

    p4 = scatter(
        we_latent[1, :], we_latent[2, :];
        marker_z = we_psi1, color = :twilight,
        markersize = 4, markerstrokewidth = 0, label = false,
        xlabel = "z1", ylabel = "z2",
        title = "Latent space colored by psi_1 (deg)",
        xlims = (-5, 5), ylims = (-5, 5),
        colorbar_title = "psi_1 (deg)",
    )

    plt2 = plot(p3, p4; layout = (1, 2), size = (1150, 480), margin = 5Plots.mm)
    savefig(plt2, joinpath(outdir, "latent_to_dihedral.png"))
    println("Saved ", joinpath(outdir, "latent_to_dihedral.png"))

    return nothing
end

main()
