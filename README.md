# DeepDriveWE.jl

[![Julia](https://img.shields.io/badge/Julia-1.10+-blueviolet)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A pure-Julia implementation of the [WESTPA](https://github.com/westpa/westpa)-style
weighted ensemble (WE) algorithm for enhanced sampling of molecular systems, coupled to
[Molly.jl](https://github.com/JuliaMolSim/Molly.jl) for Langevin MD and
[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) for differentiable end-to-end
potential learning.

The library covers three layers:

| Layer | What it does |
|-------|-------------|
| **WE data model** (`src/`) | Walker metadata, binners, resamplers, recyclers, checkpointing, sequential `run_we!` driver |
| **Simulation backends** (`src/simulation/`) | Molly.jl configs for alanine dipeptide and the 5AWL IDP fragment; pluggable `compute_pcoord` interface |
| **Differentiable pipeline** (`examples/idp_diffsim/`) | Track A: hand-rolled transformer-style trunk + CV head + energy-correction head, trained via Enzyme reverse-mode AD through short Langevin rollouts |

---

## Installation

DeepDriveWE.jl is not yet registered. Clone the repository and instantiate:

```julia
julia> using Pkg
julia> Pkg.develop(path = "/path/to/DeepDriveWE.jl")
julia> Pkg.instantiate()
```

**Julia ≥ 1.10** is required. Julia 1.12 is recommended for the differentiable
pipeline (needed for Enzyme 0.13 compatibility).

---

## Quick start

```julia
using DeepDriveWE
using Random

rng  = Random.MersenneTwister(1)
config = AlanineDipeptideConfig(n_steps = 100)

# Basis states (equilibrated PDB restart)
basis_dir = mkpath("output/basis/system1")
init_basis_state!(config, joinpath(basis_dir, "basis.jld2"); rng, n_equil_steps = 5000)

bs = BasisStates(; basis_state_dir = "output/basis",
                   basis_state_ext  = ".jld2",
                   initial_ensemble_members = 10)

we = WeightedEnsemble(; basis_states = bs, target_states = TargetState[])
initialize_basis_states!(we, basis_state_initializer(config))

# 2-D Ramachandran binner (phi/psi, 20 x 20 grid)
edges   = collect(range(-π, π; length = 21))
binner  = RectilinearBinner2D(edges, edges, 2; pcoord_idxs = (1, 2))
recycler   = LowRecycler(bs, -100.0)   # never recycle in a sampling run
resampler  = HuberKimResampler(; sims_per_bin = 2)
run_config = WERunConfig(; output_dir = "output", n_iterations = 50, rng)

run_we!(we, config, binner, recycler, resampler, run_config)
```

---

## Package structure

```
DeepDriveWE.jl/
├── src/
│   ├── api.jl                  # SimMetadata, WeightedEnsemble, BasisStates, TargetState
│   ├── driver.jl               # run_we!, WERunConfig
│   ├── checkpoint.jl           # EnsembleCheckpointer, save/load_checkpoint
│   ├── binners/
│   │   ├── base.jl             # AbstractBinner, digitize_right, bin_probs
│   │   ├── rectilinear.jl      # RectilinearBinner (1-D)
│   │   └── rectilinear2d.jl    # RectilinearBinner2D (2-D)
│   ├── resamplers/
│   │   ├── base.jl             # AbstractResampler
│   │   ├── huber_kim.jl        # HuberKimResampler (split/merge to target count)
│   │   ├── split_low.jl        # SplitLowResampler
│   │   └── split_high.jl       # SplitHighResampler
│   ├── recyclers/
│   │   ├── base.jl             # AbstractRecycler
│   │   ├── low.jl              # LowRecycler  (recycle below threshold)
│   │   └── high.jl             # HighRecycler (recycle above threshold)
│   └── simulation/
│       ├── backend.jl          # compute_pcoord, physical_config, run_segment! interface
│       ├── common.jl           # build_system, init_basis_state!, basis_state_initializer
│       ├── alanine_dipeptide.jl# AlanineDipeptideConfig, (phi, psi) pcoord
│       └── idp_fragment.jl     # IDPFragmentConfig, dihedral_features, 18-dim pcoord
├── data/
│   ├── alanine_dipeptide/      # dipeptide_nowater.pdb + ff99SBildn.xml
│   └── idp_fragment/           # 5AWL_A_noHET.pdb + a99SB-disp.xml
├── examples/
│   ├── ramachandran/           # Alanine dipeptide WE: 1-D and 2-D Ramachandran binning
│   ├── idp_fragment/           # 5AWL IDP fragment WE + plain MD baseline
│   ├── idp_cvae/               # CVAE-driven WE (Lux/Zygote trained encoder)
│   └── idp_diffsim/            # Track A: differentiable potential learning (Enzyme)
└── test/                       # 180 unit + integration tests
```

---

## Simulation backends

### Alanine dipeptide (`AlanineDipeptideConfig`)

A 22-atom vacuum model (ACE–ALA–NME) using the AMBER ff99SB-ILDN force field.
The progress coordinate is the backbone `(phi, psi)` dihedral pair (radians).

```julia
config = AlanineDipeptideConfig(
    dt          = 0.001u"ps",
    temperature = 300.0u"K",
    friction    = 1.0u"ps^-1",
    n_steps     = 100,        # steps per WE segment
)
```

### 5AWL IDP fragment (`IDPFragmentConfig`)

A 166-atom solvation-stripped 10-residue fragment of the IDP 5AWL, using the
a99SB-disp force field. The progress coordinate is the full 18-element backbone
dihedral vector `[phi_1, ..., phi_9, psi_1, ..., psi_9]` (radians).

```julia
config = IDPFragmentConfig(
    dt          = 0.002u"ps",
    temperature = 300.0u"K",
    friction    = 1.0u"ps^-1",
    n_steps     = 500,
)
```

`dihedral_features(pcoord)` converts a raw dihedral vector to
`[sin θ₁, cos θ₁, …]` (periodicity-safe, suitable as neural-network input).

### Custom backends

Implement two methods and the rest of the interface is inherited:

```julia
struct MyConfig
    pdb_file::String; ff_files::Vector{String}
    dt; temperature; friction; n_steps::Int
end

DeepDriveWE.compute_pcoord(config::MyConfig, sys) = ...  # -> Vector{Float64}
```

---

## Binners

| Type | Dimensions | Key arguments |
|------|-----------|---------------|
| `RectilinearBinner` | 1-D | `bins::Vector`, `bin_target_count` |
| `RectilinearBinner2D` | 2-D | `bins_x`, `bins_y`, `bin_target_count`, `pcoord_idxs` |

---

## Resamplers and recyclers

| Type | Role |
|------|------|
| `HuberKimResampler` | Huber & Kim (1996) split/merge to a target walker count per bin |
| `SplitLowResampler` | Split the lightest walkers |
| `SplitHighResampler` | Split the heaviest walkers |
| `LowRecycler` | Restart walkers whose pcoord falls below a threshold |
| `HighRecycler` | Restart walkers whose pcoord rises above a threshold |

---

## Examples

### Ramachandran WE (`examples/ramachandran/`)

Plain WE and a plain-MD baseline on the alanine dipeptide `(phi, psi)` surface.
Includes `plot_ramachandran.jl` for a 2-D free-energy-surface comparison.

### IDP fragment WE (`examples/idp_fragment/`)

WE and a matched plain-MD baseline on the 5AWL 10-residue IDP fragment.
Records the full 18-dim backbone dihedral pcoord per segment.

Requires the `examples/idp_fragment/` environment:

```
$ cd examples/idp_fragment
$ julia --project=. run_we_driver.jl
$ julia --project=. run_md_driver.jl
```

### CVAE-driven WE (`examples/idp_cvae/`)

A variational autoencoder (Lux + Zygote) trained on WE-collected IDP fragment
conformations. The encoder produces a 2-D latent code that replaces the raw
dihedral pcoord for `RectilinearBinner2D` binning.

```
$ cd examples/idp_cvae
$ julia --project=. train_cvae.jl     # trains the CVAE, saves cvae.jld2
$ julia --project=. run_we_cvae.jl    # CVAE-guided WE campaign (100 iterations)
$ julia --project=. plot_latent.jl    # latent coverage + dihedral-correlation plots
```

Environment packages: `DeepDriveWE`, `Lux`, `Zygote`, `Optimisers`, `JLD2`,
`Plots`, `StatsBase`, `Unitful`.

### Differentiable potential learning (`examples/idp_diffsim/`)

**Phase 5, Track A**: a shared trunk (`TrunkHeads`) with a CV head and an
energy-correction head, trained through Molly.jl's Enzyme-differentiable
Langevin simulator. Implements the full pipeline:

| Script | What it does |
|--------|-------------|
| `spike_alanine.jl` | Spike 0: confirms finite Enzyme gradients through a 50-step NoUnits Langevin rollout |
| `model.jl` | `TrunkHeads` struct (hand-rolled dense layers, `Functors`-compatible), `NNCorrection2` general interaction |
| `stage1_pretrain.jl` | Force-matching pretrain against a99SB-disp on 200 MD-sampled configurations |
| `stage2_finetune.jl` | Enzyme reverse-mode fine-tuning through 100-step rollouts with CV-reconstruction + observable-matching loss |

> **Implementation note**: `Flux.Chain` as a `general_inter` triggers an
> Enzyme `"Type Module does not have a definite size"` compile error under
> Julia 1.12 / Enzyme 0.13 / Flux 0.16. `TrunkHeads` uses hand-rolled dense
> layers to avoid this; use the same pattern for any Enzyme-differentiable
> custom interaction.

```
$ cd examples/idp_diffsim
$ julia --project=. stage1_pretrain.jl   # ~3 min; saves output_stage1/trunkheads_pretrained.jld2
$ julia --project=. stage2_finetune.jl   # saves output_stage2/trunkheads_finetuned.jld2
```

Environment packages: `DeepDriveWE`, `Molly`, `Enzyme`, `Flux`, `NNlib`,
`Optimisers`, `Functors`, `AtomsCalculators`, `JLD2`, `Plots`, `Unitful`.

---

## Dependencies

### Core library (`Project.toml`)

| Package | Role |
|---------|------|
| [Molly.jl](https://github.com/JuliaMolSim/Molly.jl) `0.23` | Molecular dynamics (Langevin integrator, force fields, `torsion_angle`) |
| [JLD2.jl](https://github.com/JuliaIO/JLD2.jl) `0.4/0.5` | Restart and checkpoint serialization |
| [Unitful.jl](https://github.com/PainterQubits/Unitful.jl) `1` | Physical-unit safety for MD parameters |
| [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) `1.9` | Stack-allocated coordinate vectors |
| [JSON3.jl](https://github.com/quinnj/JSON3.jl) `1` | Binner serialization for checksum hashing |
| [StructTypes.jl](https://github.com/JuliaData/StructTypes.jl) `1` | JSON3 struct definitions |
| [SHA.jl](https://github.com/JuliaCrypto/SHA.jl) | Binner-topology change detection |
| [YAML.jl](https://github.com/JuliaData/YAML.jl) `0.4` | Configuration file support |

### Differentiable pipeline (`examples/idp_diffsim/`)

| Package | Role |
|---------|------|
| [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) `0.13` | Reverse-mode AD through `simulate!` |
| [AtomsCalculators.jl](https://github.com/JuliaMolSim/AtomsCalculators.jl) | `forces!` interface for custom `general_inter` types |
| [NNlib.jl](https://github.com/FluxML/NNlib.jl) | `relu`, `tanh` activations (used directly, not via `Flux.Chain`) |
| [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) | Adam optimizer; operates on `Functors`-tagged structs |
| [Functors.jl](https://github.com/FluxML/Functors.jl) | `@functor` for `TrunkHeads` ↔ `Optimisers.update!` |
| [Flux.jl](https://github.com/FluxML/Flux.jl) | Used in exploration scripts (`spike0_minimal.jl`); not compatible with Enzyme inside `simulate!` |
| [Plots.jl](https://github.com/JuliaPlots/Plots.jl) | Visualization |

### CVAE example (`examples/idp_cvae/`)

| Package | Role |
|---------|------|
| [Lux.jl](https://github.com/LuxDL/Lux.jl) | Functional-style NN for the CVAE encoder/decoder |
| [Zygote.jl](https://github.com/FluxML/Zygote.jl) | Reverse-mode AD for CVAE training (no `simulate!` in the loop) |
| [StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl) | Weighted histograms for latent-space coverage plots |

---

## Testing

```
$ julia --project=. test/runtests.jl
```

The test suite (180 tests) covers the WE data model, all binner/resampler/recycler
types, the alanine dipeptide and IDP fragment simulation backends, and the
end-to-end `run_we!` driver.

---

## Roadmap

- **Track B** – ACEpotentials.jl equivariant energy correction: fit a small
  ACE model on WE-sampled configurations, embed as a Molly `general_inter`,
  and probe Enzyme-through-ACE differentiability (B-full) or use as a fixed
  equivariant correction alongside the differentiable CV head (B-frozen).
- **IDP fragment port** – extend Track A/B to the 166-atom, 10-residue 5AWL
  fragment (self-attention trunk over residues).
- **Evaluation visualization** – compare free-energy surfaces and latent-space
  coverage across classical, CVAE-guided, and Track A/B WE runs.

---

## References

- Huber, G. A. & Kim, S. (1996). Weighted-ensemble Brownian dynamics simulations
  for protein association reactions. *Biophys. J.*, **70**(1), 97–110.
- Zwier, M. C. et al. (2015). WESTPA: an interoperable, highly scalable software
  package for weighted ensemble simulation and analysis. *J. Chem. Theory Comput.*,
  **11**(2), 800–809.
- Greener, J. G. & Jones, D. T. (2021). Differentiable molecular simulation can
  learn all the parameters in a coarse-grained force field for proteins.
  *PLOS ONE*, **16**(9), e0256990. *(GB99dms — direct prior art for
  Enzyme-differentiable Langevin simulation of disordered proteins.)*
- Eastman, P. et al. (2023). SPICE, a dataset of drug-like molecules and peptides
  for training machine learning potentials. *Sci. Data*, **10**, 11.

---

## License

MIT — see [LICENSE](LICENSE).

Developed at [Argonne National Laboratory](https://www.anl.gov).
