@testset "recyclers" begin
    function _make_basis_states()
        BasisStates(;
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
    end

    function _make_sims(pcoords)
        [
            SimMetadata(;
                weight = 1.0 / length(pcoords),
                simulation_id = i,
                iteration_id = 1,
                parent_restart_file = "x",
                parent_pcoord = [0.0],
                pcoord = [[p]],
                restart_file = "restart_$i.jld2",
            )
            for (i, p) in enumerate(pcoords)
        ]
    end

    @testset "LowRecycler" begin
        r = LowRecycler(_make_basis_states(), 0.2)

        pcoords = reshape([0.1, 0.5, 0.05], :, 1)
        @test recycle(r, pcoords) == [1, 3]
    end

    @testset "HighRecycler" begin
        r = HighRecycler(_make_basis_states(), 0.8)

        pcoords = reshape([0.1, 0.9, 0.95], :, 1)
        @test recycle(r, pcoords) == [2, 3]
    end

    @testset "recycle_simulations" begin
        bs = _make_basis_states()
        r = LowRecycler(bs, 0.2)

        cur_sims = _make_sims([0.1, 0.5, 0.9])
        next_sims = _make_sims([0.15, 0.55, 0.95])

        new_cur, new_next = recycle_simulations(r, cur_sims, next_sims)

        # Only the first walker (pcoord 0.1 < 0.2) is recycled.
        @test new_cur[1].endpoint_type == 3
        @test new_cur[2].endpoint_type == 1
        @test new_cur[3].endpoint_type == 1

        recycled = new_next[1]
        basis_state = bs.basis_states[1]
        @test recycled.parent_restart_file == basis_state.parent_restart_file
        @test recycled.parent_pcoord == basis_state.parent_pcoord
        @test recycled.parent_simulation_id == -(cur_sims[1].simulation_id)
        @test recycled.simulation_id == next_sims[1].simulation_id
        @test recycled.weight == next_sims[1].weight

        # Non-recycled entries are untouched.
        @test new_next[2].parent_pcoord == next_sims[2].parent_pcoord
        @test new_next[3].parent_pcoord == next_sims[3].parent_pcoord

        # Original inputs are not mutated.
        @test cur_sims[1].endpoint_type == 1
    end
end
