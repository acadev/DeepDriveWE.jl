export AlanineDipeptideConfig

const ALANINE_DIPEPTIDE_DATA_DIR = normpath(@__DIR__, "..", "..", "data", "alanine_dipeptide")

# 1-based atom indices into the ACE-ALA-NME topology of `dipeptide_nowater.pdb`.
# phi = ACE(C) - ALA(N) - ALA(CA) - ALA(C); psi = ALA(N) - ALA(CA) - ALA(C) - NME(N).
const PHI_ATOMS = (5, 7, 9, 15)
const PSI_ATOMS = (7, 9, 15, 17)

"""
    AlanineDipeptideConfig

Configuration for running short alanine dipeptide MD segments with Molly.jl,
using the bundled `dipeptide_nowater.pdb` structure and `ff99SBildn` force
field.

The progress coordinate is the backbone `(phi, psi)` dihedral pair, in
radians.
"""
Base.@kwdef struct AlanineDipeptideConfig
    pdb_file::String = joinpath(ALANINE_DIPEPTIDE_DATA_DIR, "dipeptide_nowater.pdb")
    ff_files::Vector{String} = [joinpath(ALANINE_DIPEPTIDE_DATA_DIR, "ff99SBildn.xml")]
    dt::typeof(1.0u"ps") = 0.001u"ps"
    temperature::typeof(1.0u"K") = 300.0u"K"
    friction::typeof(1.0u"ps^-1") = 1.0u"ps^-1"
    n_steps::Int = 100
end

"""
    compute_pcoord(config::AlanineDipeptideConfig, sys) -> Vector{Float64}

Compute the `(phi, psi)` backbone dihedral progress coordinate (radians) for
an alanine dipeptide system.
"""
function compute_pcoord(config::AlanineDipeptideConfig, sys)
    phi = torsion_angle(
        sys.coords[PHI_ATOMS[1]], sys.coords[PHI_ATOMS[2]],
        sys.coords[PHI_ATOMS[3]], sys.coords[PHI_ATOMS[4]], sys.boundary,
    )
    psi = torsion_angle(
        sys.coords[PSI_ATOMS[1]], sys.coords[PSI_ATOMS[2]],
        sys.coords[PSI_ATOMS[3]], sys.coords[PSI_ATOMS[4]], sys.boundary,
    )
    return [Float64(phi), Float64(psi)]
end
