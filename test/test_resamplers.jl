@testset "resamplers" begin
    function _make_next_sims(weights)
        [
            SimMetadata(;
                weight = w,
                simulation_id = i,
                iteration_id = 1,
                parent_restart_file = "restart_$i.jld2",
                parent_pcoord = [p],
                parent_simulation_id = i,
                wtg_parent_ids = [i],
                restart_file = "restart_$i.jld2",
            )
            for (i, (w, p)) in enumerate(zip(weights, range(0.0, 1.0; length = length(weights))))
        ]
    end

    function _make_cur_sims(n)
        [
            SimMetadata(;
                weight = 1.0 / n,
                simulation_id = i,
                iteration_id = 1,
                parent_restart_file = "x",
                parent_pcoord = [0.0],
                pcoord = [[0.0]],
            )
            for i in 1:n
        ]
    end

    @testset "HuberKimResampler" begin
        weights = [0.5, 0.3, 0.1, 0.05, 0.05]
        next_sims = _make_next_sims(weights)
        cur_sims = _make_cur_sims(length(weights))

        r = HuberKimResampler(; sims_per_bin = 3, max_allowed_weight = 1.0, min_allowed_weight = 1e-10)

        cur, nxt = resample(r, cur_sims, next_sims)

        @test length(nxt) == r.sims_per_bin
        @test sum(sim.weight for sim in nxt) ≈ 1.0
        @test r.index_counter > 0
        @test all(sim.weight <= r.max_allowed_weight for sim in nxt)
        @test all(sim.weight >= r.min_allowed_weight for sim in nxt)
    end

    @testset "SplitLowResampler" begin
        weights = [1 / 3, 1 / 3, 1 / 3]
        next_sims = _make_next_sims(weights)
        cur_sims = _make_cur_sims(length(weights))

        r = SplitLowResampler(; num_resamples = 1, n_split = 2, pcoord_idx = 1)

        cur, nxt = resample(r, cur_sims, next_sims)

        # Split adds one walker, merge removes one => count unchanged.
        @test length(nxt) == length(next_sims)
        @test sum(sim.weight for sim in nxt) ≈ 1.0
        @test r.index_counter > 0

        # The merged walker should carry the union of wtg_parent_ids from
        # the two highest-pcoord sims (ids 2 and 3).
        merged = only(filter(sim -> sim.simulation_id == r.index_counter - 1, nxt))
        @test sort(merged.wtg_parent_ids) == [2, 3]
    end

    @testset "SplitHighResampler" begin
        weights = [1 / 3, 1 / 3, 1 / 3]
        next_sims = _make_next_sims(weights)
        cur_sims = _make_cur_sims(length(weights))

        r = SplitHighResampler(; num_resamples = 1, n_split = 2, pcoord_idx = 1)

        cur, nxt = resample(r, cur_sims, next_sims)

        @test length(nxt) == length(next_sims)
        @test sum(sim.weight for sim in nxt) ≈ 1.0
        @test r.index_counter > 0

        # The merged walker should carry the union of wtg_parent_ids from
        # the two lowest-pcoord sims (ids 1 and 2).
        merged = only(filter(sim -> sim.simulation_id == r.index_counter - 1, nxt))
        @test sort(merged.wtg_parent_ids) == [1, 2]
    end

    @testset "run_resampling end-to-end" begin
        binner = RectilinearBinner([0.0, 0.5, 1.0], 2; target_state_inds = Int[])
        basis_states = BasisStates(;
            basis_state_dir = "basis",
            initial_ensemble_members = 1,
            num_basis_files = 1,
            basis_states = [
                SimMetadata(;
                    weight = 1.0,
                    simulation_id = 1,
                    iteration_id = 1,
                    parent_restart_file = "basis/state1.jld2",
                    parent_pcoord = [0.5],
                    parent_simulation_id = -1,
                    wtg_parent_ids = [-1],
                ),
            ],
        )
        recycler = LowRecycler(basis_states, -1.0)
        resampler = HuberKimResampler(; sims_per_bin = 2)

        pcoords = [0.1, 0.4, 0.6, 0.9]
        cur_sims = [
            SimMetadata(;
                weight = 0.25,
                simulation_id = i,
                iteration_id = 1,
                parent_restart_file = "x",
                parent_pcoord = [0.0],
                pcoord = [[p]],
                restart_file = "restart_$i.jld2",
            )
            for (i, p) in enumerate(pcoords)
        ]

        new_cur, new_sims, metadata = run_resampling(resampler, cur_sims, binner, recycler)

        @test sum(sim.weight for sim in new_sims) ≈ 1.0
        @test length(new_sims) == sum(get_bin_target_counts(binner))
        @test metadata.bin_target_counts == [2, 2]
    end
end
