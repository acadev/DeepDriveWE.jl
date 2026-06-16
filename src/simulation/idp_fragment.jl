export IDPFragmentConfig, dihedral_features

const IDP_FRAGMENT_DATA_DIR = normpath(@__DIR__, "..", "..", "data", "idp_fragment")

"""
    IDPFragmentConfig

Configuration for running short MD segments of a small (10-residue)
intrinsically-disordered peptide fragment with Molly.jl, using the bundled
`5AWL_A_noHET.pdb` structure and `a99SB-disp` force field.

The progress coordinate is the full set of backbone `(phi, psi)` dihedral
angles, in radians: for an `n`-residue chain there are `n - 1` phi angles
(residues `2:n`) followed by `n - 1` psi angles (residues `1:n-1`), so
`length(compute_pcoord(config, sys)) == 2 * (n - 1)`.
"""
struct IDPFragmentConfig
    pdb_file::String
    ff_files::Vector{String}
    dt::typeof(1.0u"ps")
    temperature::typeof(1.0u"K")
    friction::typeof(1.0u"ps^-1")
    n_steps::Int
    phi_atoms::Vector{NTuple{4, Int}}
    psi_atoms::Vector{NTuple{4, Int}}
end

function IDPFragmentConfig(;
    pdb_file::String = joinpath(IDP_FRAGMENT_DATA_DIR, "5AWL_A_noHET.pdb"),
    ff_files::Vector{String} = [joinpath(IDP_FRAGMENT_DATA_DIR, "a99SB-disp.xml")],
    dt::typeof(1.0u"ps") = 0.002u"ps",
    temperature::typeof(1.0u"K") = 300.0u"K",
    friction::typeof(1.0u"ps^-1") = 1.0u"ps^-1",
    n_steps::Int = 500,
)
    phi_atoms, psi_atoms = _backbone_dihedral_atoms(pdb_file, ff_files)
    return IDPFragmentConfig(pdb_file, ff_files, dt, temperature, friction, n_steps, phi_atoms, psi_atoms)
end

function _backbone_dihedral_atoms(pdb_file::String, ff_files::Vector{String})
    ff = MolecularForceField(ff_files...; units=true)
    sys = System(pdb_file, ff; nonbonded_method=:none)
    atoms_data = sys.atoms_data

    n_idx = Dict{Int, Int}()
    ca_idx = Dict{Int, Int}()
    c_idx = Dict{Int, Int}()
    for (i, a) in enumerate(atoms_data)
        if a.atom_name == "N"
            n_idx[a.res_number] = i
        elseif a.atom_name == "CA"
            ca_idx[a.res_number] = i
        elseif a.atom_name == "C"
            c_idx[a.res_number] = i
        end
    end

    res_numbers = sort(collect(keys(n_idx)))

    phi_atoms = NTuple{4, Int}[]
    for k in 2:length(res_numbers)
        r, rprev = res_numbers[k], res_numbers[k - 1]
        push!(phi_atoms, (c_idx[rprev], n_idx[r], ca_idx[r], c_idx[r]))
    end

    psi_atoms = NTuple{4, Int}[]
    for k in 1:(length(res_numbers) - 1)
        r, rnext = res_numbers[k], res_numbers[k + 1]
        push!(psi_atoms, (n_idx[r], ca_idx[r], c_idx[r], n_idx[rnext]))
    end

    return phi_atoms, psi_atoms
end

"""
    compute_pcoord(config::IDPFragmentConfig, sys) -> Vector{Float64}

Compute all backbone `(phi, psi)` dihedral angles (radians), as
`[phi_2, ..., phi_n, psi_1, ..., psi_{n-1}]`.
"""
function compute_pcoord(config::IDPFragmentConfig, sys)
    angle_at(atoms) = Float64(torsion_angle(
        sys.coords[atoms[1]], sys.coords[atoms[2]],
        sys.coords[atoms[3]], sys.coords[atoms[4]], sys.boundary,
    ))
    return vcat(angle_at.(config.phi_atoms), angle_at.(config.psi_atoms))
end

"""
    dihedral_features(pcoord::AbstractVector{<:Real}) -> Vector{Float64}

Map a vector of `n` dihedral angles (radians) to `2n` periodicity-safe
features `[sin(theta_1), cos(theta_1), ..., sin(theta_n), cos(theta_n)]`,
suitable as CVAE input/output.
"""
function dihedral_features(pcoord::AbstractVector{<:Real})
    features = Vector{Float64}(undef, 2 * length(pcoord))
    for (i, theta) in enumerate(pcoord)
        features[2i - 1] = sin(theta)
        features[2i] = cos(theta)
    end
    return features
end
