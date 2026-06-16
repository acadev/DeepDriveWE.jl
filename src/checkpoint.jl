export EnsembleCheckpointer, save_checkpoint, load_checkpoint, latest_checkpoint

"""
    EnsembleCheckpointer(output_dir::AbstractString)

Checkpointer for a [`WeightedEnsemble`](@ref). Writes JSON checkpoints to
`output_dir/checkpoints/checkpoint-NNNNNN.json`.

Note: unlike the upstream Python implementation, this is JSON-only (no
`west.h5`/HDF5 output) for v1.
"""
struct EnsembleCheckpointer
    checkpoint_dir::String

    function EnsembleCheckpointer(output_dir::AbstractString)
        checkpoint_dir = joinpath(output_dir, "checkpoints")
        mkpath(checkpoint_dir)
        new(checkpoint_dir)
    end
end

"""
    save_checkpoint(c::EnsembleCheckpointer, we::WeightedEnsemble)

Write `we` to `checkpoint-NNNNNN.json`, where `NNNNNN` is the current
iteration number.
"""
function save_checkpoint(c::EnsembleCheckpointer, we::WeightedEnsemble)
    filename = "checkpoint-$(lpad(we.metadata.iteration_id, 6, '0')).json"
    open(joinpath(c.checkpoint_dir, filename), "w") do io
        JSON3.write(io, we)
    end
    return nothing
end

"""
    latest_checkpoint(c::EnsembleCheckpointer) -> Union{String, Nothing}

Return the path to the most recent checkpoint file, or `nothing` if none
exist.
"""
function latest_checkpoint(c::EnsembleCheckpointer)
    files = filter(
        f -> startswith(f, "checkpoint-") && endswith(f, ".json"),
        readdir(c.checkpoint_dir),
    )
    isempty(files) && return nothing
    return joinpath(c.checkpoint_dir, maximum(files))
end

"""
    load_checkpoint(c::EnsembleCheckpointer, path=nothing) -> WeightedEnsemble

Load a [`WeightedEnsemble`](@ref) from `path`, defaulting to the latest
checkpoint.
"""
function load_checkpoint(c::EnsembleCheckpointer, path::Union{AbstractString, Nothing} = nothing)
    if path === nothing
        path = latest_checkpoint(c)
        path === nothing && throw(ArgumentError("No checkpoint file found in $(c.checkpoint_dir)"))
    end
    return open(io -> JSON3.read(io, WeightedEnsemble), path, "r")
end
