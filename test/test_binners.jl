@testset "binners" begin
    @testset "digitize_right" begin
        bins = [0.0, 0.25, 0.5, 0.75, 1.0]

        @test digitize_right(0.0, bins) == 1
        @test digitize_right(0.1, bins) == 1
        @test digitize_right(0.25, bins) == 1
        @test digitize_right(0.3, bins) == 2
        @test digitize_right(1.0, bins) == 4
        @test digitize_right(1.5, bins) == 5
    end

    @testset "RectilinearBinner" begin
        bins = [0.0, 0.25, 0.5, 0.75, 1.0]

        @test_throws ArgumentError RectilinearBinner([1.0, 0.0], 5)

        b = RectilinearBinner(bins, 5; target_state_inds = [1])
        @test nbins(b) == 4
        @test bin_labels(b) == ["state1", "state2", "state3", "state4"]

        # bin_target_counts: int expanded, target state bin zeroed
        counts = get_bin_target_counts(b)
        @test counts == [0, 5, 5, 5]
        # cached
        @test b.bin_target_counts == [0, 5, 5, 5]

        pcoords = reshape([0.0, 0.1, 0.3, 0.6, 0.9, 1.0], :, 1)
        assigned = assign_bins(b, pcoords)
        @test assigned == [1, 1, 2, 3, 4, 4]

        assignments = bin_assignments(b, pcoords)
        @test sort(assignments[1]) == [1, 2]
        @test assignments[2] == [3]
        @test assignments[3] == [4]
        @test sort(assignments[4]) == [5, 6]
    end

    @testset "RectilinearBinner2D" begin
        bins_x = [-1.0, 0.0, 1.0]
        bins_y = [-1.0, 0.0, 1.0]

        @test_throws ArgumentError RectilinearBinner2D([1.0, 0.0], bins_y, 3)

        b = RectilinearBinner2D(bins_x, bins_y, 3; target_state_inds = [1])
        @test nbins(b) == 4

        counts = get_bin_target_counts(b)
        @test counts == [0, 3, 3, 3]
        @test b.bin_target_counts == [0, 3, 3, 3]

        # grid layout: linear index = (iy - 1) * nx + ix, nx = 2
        # (x, y) -> (ix, iy) -> linear index
        pcoords = [
            -0.5 -0.5;  # ix=1, iy=1 -> 1
             0.5 -0.5;  # ix=2, iy=1 -> 2
            -0.5  0.5;  # ix=1, iy=2 -> 3
             0.5  0.5;  # ix=2, iy=2 -> 4
             1.0  1.0;  # at upper edge -> clamped to (2, 2) -> 4
            -2.0 -2.0;  # below range -> clamped to (1, 1) -> 1
        ]
        assigned = assign_bins(b, pcoords)
        @test assigned == [1, 2, 3, 4, 4, 1]
    end

    @testset "bin_simulations and compute_iteration_metadata" begin
        b = RectilinearBinner([0.0, 0.5, 1.0], 2; target_state_inds = [1])

        sims = [
            SimMetadata(;
                weight = 0.25, simulation_id = i, iteration_id = 1,
                parent_restart_file = "x", parent_pcoord = [p],
                pcoord = [[p]],
            )
            for (i, p) in enumerate([0.1, 0.4, 0.6, 0.9])
        ]

        bins = bin_simulations(b, sims)
        @test sort(bins[1]) == [1, 2]
        @test sort(bins[2]) == [3, 4]

        meta = compute_iteration_metadata(b, sims)
        @test meta.iteration_id == 1
        @test meta.bin_target_counts == [0, 2]
        @test meta.min_bin_prob ≈ 0.5
        @test meta.max_bin_prob ≈ 0.5
        @test !isempty(meta.binner_hash)
    end
end
