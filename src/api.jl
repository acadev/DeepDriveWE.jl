export SimMetadata, IterationMetadata, TargetState, BasisStates, WeightedEnsemble
export mark_simulation_start!, mark_simulation_end!, walltime, simulation_name,
       num_frames, append_pcoord!
export unique_basis_states, load_basis_states!
export initialize_basis_states!, iteration, advance_iteration!

"""
    SimMetadata

Metadata for a single simulation (walker) in the weighted ensemble.

`endpoint_type` follows the WESTPA convention: 1 = continue, 2 = merged,
3 = recycled.
"""
Base.@kwdef mutable struct SimMetadata
    weight::Float64 = 0.0
    simulation_id::Int = 0
    iteration_id::Int = 0
    parent_restart_file::String = ""
    parent_pcoord::Vector{Float64} = Float64[]
    parent_simulation_id::Union{Int, Nothing} = nothing
    wtg_parent_ids::Vector{Int} = Int[]
    restart_file::Union{String, Nothing} = nothing
    pcoord::Vector{Vector{Float64}} = Vector{Float64}[]
    auxdata::Dict{String, Any} = Dict{String, Any}()
    endpoint_type::Int = 1
    simulation_start_time::Float64 = 0.0
    simulation_end_time::Float64 = 0.0
end

"""
    mark_simulation_start!(sim::SimMetadata)

Mark the start time of the simulation.
"""
mark_simulation_start!(sim::SimMetadata) = (sim.simulation_start_time = time(); sim)

"""
    mark_simulation_end!(sim::SimMetadata)

Mark the end time of the simulation.
"""
mark_simulation_end!(sim::SimMetadata) = (sim.simulation_end_time = time(); sim)

"""
    walltime(sim::SimMetadata)

Return the walltime (seconds) of the simulation.
"""
walltime(sim::SimMetadata) = sim.simulation_end_time - sim.simulation_start_time

"""
    simulation_name(sim::SimMetadata)

Return the simulation name (used to create the output directory), e.g. "000001/000002".
"""
simulation_name(sim::SimMetadata) =
    string(lpad(sim.iteration_id, 6, '0'), "/", lpad(sim.simulation_id, 6, '0'))

"""
    num_frames(sim::SimMetadata)

Return the number of frames in the simulation.
"""
num_frames(sim::SimMetadata) = length(sim.pcoord)

"""
    append_pcoord!(sim::SimMetadata, pcoords::Vector{Float64})

Append a new progress-coordinate component to each frame's pcoord vector.

`pcoords` must have the same length as `sim.pcoord` (one value per frame).
"""
function append_pcoord!(sim::SimMetadata, pcoords::Vector{Float64})
    if length(pcoords) != length(sim.pcoord)
        throw(ArgumentError(
            "The number of frames in the progress coordinate does not " *
            "match the number of frames in the simulation metadata.",
        ))
    end
    for (orig_pcoord, p) in zip(sim.pcoord, pcoords)
        push!(orig_pcoord, p)
    end
    return sim
end

"""
    IterationMetadata

Metadata for an iteration of the weighted ensemble.

Note: unlike the Python implementation, the binner is not pickled into this
struct. Instead `binner_hash` is a SHA-256 hash of the binner's JSON
serialization, used to detect topology changes between iterations.
"""
Base.@kwdef mutable struct IterationMetadata
    iteration_id::Int = 1
    binner_hash::String = ""
    min_bin_prob::Float64 = 0.0
    max_bin_prob::Float64 = 0.0
    bin_target_counts::Vector{Int} = Int[]
end

"""
    TargetState

Target state for the weighted ensemble.
"""
Base.@kwdef mutable struct TargetState
    label::String = ""
    pcoord::Vector{Float64} = Float64[]
end

"""
    BasisStates

Basis states for the weighted ensemble.

`basis_state_dir` is a nested directory storing initial simulation start
files, e.g. `pdb_dir/system1/`, `pdb_dir/system2/`, ..., where `system<i>`
might store coordinate/restart files needed to start the simulation.
"""
Base.@kwdef mutable struct BasisStates
    basis_state_dir::String = ""
    basis_state_ext::String = ".jld2"
    initial_ensemble_members::Int = 0
    randomly_initialize::Bool = false
    random_seed::Int = 0
    num_basis_files::Int = 0
    basis_states::Vector{SimMetadata} = SimMetadata[]
end

Base.length(bs::BasisStates) = length(bs.basis_states)
Base.getindex(bs::BasisStates, idx::Int) = bs.basis_states[idx]
Base.iterate(bs::BasisStates, state...) = iterate(bs.basis_states, state...)

"""
    unique_basis_states(bs::BasisStates)

Return the unique basis states (the first `num_basis_files` entries).
"""
unique_basis_states(bs::BasisStates) = bs.basis_states[1:bs.num_basis_files]

"""
    load_basis_states!(bs::BasisStates, basis_state_initializer)

Load the basis states for the weighted ensemble.

`basis_state_initializer` is a function `(basis_file::String) -> Vector{Float64}`
that computes the initial progress coordinate for a basis state file.
"""
function load_basis_states!(bs::BasisStates, basis_state_initializer)
    basis_files = _glob_basis_states(bs)

    basis_pcoords = [basis_state_initializer(f) for f in basis_files]

    bs.basis_states = _uniform_init(bs, basis_files, basis_pcoords)
    bs.num_basis_files = length(basis_files)

    @info "Loaded $(length(bs.basis_states)) basis states"
    return bs
end

function _glob_basis_states(bs::BasisStates)
    sim_input_dirs = filter(isdir, readdir(bs.basis_state_dir; join=true))
    sort!(sim_input_dirs)

    if length(sim_input_dirs) > bs.initial_ensemble_members && bs.randomly_initialize
        rng = Random.MersenneTwister(bs.random_seed)
        Random.shuffle!(rng, sim_input_dirs)
    end

    sim_input_dirs = sim_input_dirs[1:min(end, bs.initial_ensemble_members)]

    basis_states = String[]
    for input_dir in sim_input_dirs
        candidates = filter(
            f -> endswith(f, bs.basis_state_ext),
            readdir(input_dir; join=true),
        )
        if isempty(candidates)
            throw(ArgumentError(
                "No basis state in $input_dir found with extension: $(bs.basis_state_ext)",
            ))
        end
        push!(basis_states, first(sort(candidates)))
    end

    return basis_states
end

function _uniform_init(
    bs::BasisStates,
    basis_files::Vector{String},
    basis_pcoords::Vector{Vector{Float64}},
)
    weight = 1.0 / bs.initial_ensemble_members

    # Map each unique basis file to a negative index (1-based, so files get
    # parent IDs -1, -2, ... which never collide with real (>=1) sim ids).
    index_map = Dict(file => -idx for (idx, file) in enumerate(basis_files))

    n_files = length(basis_files)
    simulations = SimMetadata[]
    for idx in 1:bs.initial_ensemble_members
        cyc = mod1(idx, n_files)
        file = basis_files[cyc]
        pcoord = basis_pcoords[cyc]

        push!(simulations, SimMetadata(;
            weight=weight,
            simulation_id=idx,
            iteration_id=1,
            parent_restart_file=file,
            parent_pcoord=pcoord,
            parent_simulation_id=index_map[file],
            wtg_parent_ids=[index_map[file]],
        ))
    end

    return simulations
end

"""
    WeightedEnsemble

Top-level container for a weighted-ensemble simulation campaign.
"""
Base.@kwdef mutable struct WeightedEnsemble
    basis_states::BasisStates = BasisStates()
    target_states::Vector{TargetState} = TargetState[]
    metadata::IterationMetadata = IterationMetadata()
    cur_sims::Vector{SimMetadata} = SimMetadata[]
    next_sims::Vector{SimMetadata} = SimMetadata[]
end

"""
    initialize_basis_states!(we::WeightedEnsemble, basis_state_initializer)

Load the basis states and initialize `next_sims` with them.
"""
function initialize_basis_states!(we::WeightedEnsemble, basis_state_initializer)
    load_basis_states!(we.basis_states, basis_state_initializer)
    we.next_sims = deepcopy(we.basis_states.basis_states)
    return we
end

"""
    iteration(we::WeightedEnsemble)

Return the current iteration number of the weighted ensemble.
"""
iteration(we::WeightedEnsemble) = we.metadata.iteration_id

"""
    advance_iteration!(we::WeightedEnsemble, cur_sims, next_sims, metadata)

Advance the weighted ensemble to the next iteration.
"""
function advance_iteration!(
    we::WeightedEnsemble,
    cur_sims::Vector{SimMetadata},
    next_sims::Vector{SimMetadata},
    metadata::IterationMetadata,
)
    we.metadata = metadata
    we.cur_sims = cur_sims
    we.next_sims = next_sims
    return we
end

StructTypes.StructType(::Type{SimMetadata}) = StructTypes.Mutable()
StructTypes.StructType(::Type{IterationMetadata}) = StructTypes.Mutable()
StructTypes.StructType(::Type{TargetState}) = StructTypes.Mutable()
StructTypes.StructType(::Type{BasisStates}) = StructTypes.Mutable()
StructTypes.StructType(::Type{WeightedEnsemble}) = StructTypes.Mutable()
