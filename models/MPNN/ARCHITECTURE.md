# CrystalMPNN: a chemically- and geometrically-motivated crystal graph network

**DRAFT for review.** How the MPNN pipeline turns a crystal structure into a
property prediction, and — the point of this document — *why each representation
and architecture choice is the shape it is*, traced back to the crystal
chemistry or geometry it encodes. Companion docs: [`DATALOADING.md`](DATALOADING.md)
(how data reaches the GPU), [`../../configs/README.md`](../../configs/README.md)
(every config knob), `claude.md` (research roadmap).

## Design philosophy

Modern universal potentials (MACE, NequIP, CHGNet) learn chemistry *from
scratch* — atomic numbers and positions in, everything else discovered from
millions of energy/force labels. That is the right trade when labels are
abundant. Our end target, superconducting T_c, is the opposite regime:
**data-starved (~6k labeled materials) and label-noisy**, where every bit of
chemistry the model does not have to rediscover is a bit of scarce signal it
can spend on the actual task.

So this architecture takes the complementary bet: **encode the descriptors a
solid-state chemist would reach for** — coordination environments, bond
valence, electronegativity contrast, polyhedral connectivity, bond angles —
**as explicit graph features and as inductive biases inside the message
passing itself.** The formation-energy work (MP_Energy, MPtrj) is the
benchmark regime where these choices are validated before they face T_c.

Everything below is rotation/translation **invariant** by construction
(distances, angles, weights — never raw coordinates), which is exactly right
for invariant scalar targets and deliberately insufficient for forces; the
equivariant extension is a separate roadmap phase.

---

## 1. The pipeline

```
CIF / pymatgen Structure
   │  crystal_graph_v4 (RPToleranceFactor): radical Voronoi + ECoN analysis
   ▼
full graph ──compact──► graph JSON          one per material/frame
   │                       │  main.py pack-dataset (optional, ~30x reads)
   │                       ▼
   │                  packed columnar store (memmap arrays + offsets)
   ▼
CIFDataV4 / PackedCIFDataV4 ──collate──► padded batch tensors
   ▼
CrystalMPNN ──► scalar (T_c, E_form, …) or 2-class logits (SC / non-SC)
```

Both dataset backends assemble samples through the same extraction/padding
code (`_extract_ragged` / `_assemble_sample` in `MPNNData.py`), so they are
bitwise-interchangeable; the model never knows which served it.

### 1.1 Graph construction (`crystal_graph_v4`)

The bonding graph is **not** a fixed-radius cutoff. Edges come from a
**two-pass self-consistent radical (radii-weighted) Voronoi tessellation**:

- *Why Voronoi rather than a distance cutoff:* a cutoff treats a 2.0 Å contact
  in an oxide the same as in an intermetallic and happily connects through
  atoms. Voronoi faces define "neighbor" the way coordination chemistry does —
  by shared contact geometry — and the face area gives a continuous
  significance weight for free.
- *Why radical (radii-weighted):* plain Voronoi bisects bonds at the midpoint,
  which misassigns faces when atom sizes differ (O²⁻ vs W⁶⁺). The radical
  construction bisects at radius-weighted positions. The radii themselves are
  **Shannon crystal radii chosen self-consistently**: pass 1 estimates each
  atom's coordination number, which selects the proper Shannon radius
  (Shannon radii are CN-dependent), and pass 2 recomputes the tessellation
  with those radii.
- Edge admission then applies chemical filters: minimum Voronoi face weight
  (0.005), a per-center distance-ratio guard, cation–anion role logic with
  empirically-tuned same-role/opposite-role bond-length caps, and a hard cap
  of 14 edges/atom.
- After the edge set is fixed, **Hoppe's effective coordination number
  (ECoN)** is computed per atom: each bond gets a weight that decays
  exponentially with its length relative to the atom's shortest bond. This is
  the chemist's answer to "how much does this bond count?" — a 2.0 Å and a
  2.6 Å Cu–O bond are both "bonds", but not equally.

The same machinery also emits **polyhedral edges**, **bond-angle triplets**,
and **dihedrals** (§2.3–2.4).

---

## 2. The representation: what the model sees and why

### 2.1 Node features (14, `NODE_FEA_LEN`)

| feature | chemical rationale |
|---|---|
| `Z` | element identity |
| `oxidation_state`, `ion_role` (+1/0/−1) | formal charge chemistry: who donates, who accepts; gates the cation–anion bond logic |
| `chi_pauling`, `chi_allen` | **two** electronegativity scales on purpose — Pauling's is thermochemical (bond-energy derived), Allen's is spectroscopic (configuration energy); they disagree most for exactly the heavy/transition elements common in superconductors |
| `ecn_value` | effective (Hoppe) coordination number — continuous, distortion-aware |
| `shannon_radius` | ionic size *in its observed coordination/charge state*, not a bare elemental radius |
| `cn_core` | integer count of core-shell bonds (ECoN weight ≥ 0.5) |
| `hist_corner/edge/face/other` | how this atom's coordination polyhedron shares with its neighbors — Pauling's third rule (face-sharing destabilizes, corner-sharing is benign) as four counts |
| `ionization_energy`, `electron_affinity` | free-atom electronic anchors (carried over from the original CGCNN featurization) |

All raw values; per-feature standardization (mean/std over the *training* split
only) is installed in the model as buffers, so the same normalization applies
at train/eval/inference and travels with the checkpoint.

### 2.2 Bonding-edge features (7, `NBR_FEA_LEN`)

| feature | rationale |
|---|---|
| `bond_length` | the basic geometric scalar |
| `bond_length / Σ(radii)` | the **dimensionless** version: 1.0 = "ideal" contact for this chemistry, so compression/strain is comparable across element pairs |
| `voronoi_weight_center`, `voronoi_weight_nbr` | shared-face area fraction from each endpoint's perspective — geometric contact significance |
| `ecn_weight_center`, `ecn_weight_nbr` | Hoppe bond weight from each endpoint's perspective — chemical bond significance |
| `delta_chi_pauling` | electronegativity contrast = ionicity of *this* bond |

Two non-obvious decisions:

- **Center-relative orientation.** A bond looks different from its two ends: a
  bond that is the strongest contact of a small cation can be a minor contact
  of a large anion (measured on MP_Energy: the two ECoN weights of an edge
  correlate only r≈0.53, the Voronoi weights r≈−0.10). The data layer
  therefore swaps the directional columns so that index 4 is always *the
  center atom's own* weight, for every directed use of the edge. (Legacy
  graphs orient via the builder's `source ≤ target` canonicalization — no
  rebuild needed.)
- **What was removed:** a binary `coord_sphere` (core/extended) flag was
  dropped after measuring |r| ≈ 0.90 against the ECoN columns — it is a
  threshold of information the model already receives continuously. Redundant
  inputs are overfitting surface, not information.

### 2.3 Polyhedral edges (7, `POLY_FEA_LEN`) — the second relation type

Second-neighbor connections between two atoms whose coordination polyhedra
share ≥1 bridging atom. This is **inter-polyhedral connectivity** — the
language of structural chemistry (corner-/edge-/face-sharing octahedra is how
one *describes* a perovskite) — and it is invisible to a bonding graph, which
must compose two hops to even notice it.

| feature | rationale |
|---|---|
| `shared_count` (1/2/3 = corner/edge/face) | Pauling's third rule again, now as a relation: face-sharing brings cations closer (destabilizing, but also electronically coupling) |
| `mean_angle_deg`, `std_angle_deg` | the bridge angle (e.g. the Cu–O–Cu superexchange angle in cuprates — directly tied to magnetic coupling) and its spread (distortion) |
| `mean_path_length`, `std_path_length` | through-bridge distances |
| `direct_distance` | cation–cation distance |
| `path_type` (+1/0/−1) | cation–anion–cation vs anion–cation–anion vs other |

Empirical validation: turning this channel on improves formation-energy test
MAE by ~6% relative (paired-significant) at matched aggregation/pooling.

### 2.4 Bond-angle triplets (3-body)

For every pair of bonds meeting at a center, the graph stores
`[center, edge_a, edge_b, cos θ]`. Angles are what distinguish a square-planar
from a tetrahedral 4-coordinate site — pure 2-body features cannot. Stored as
`cos θ` and expanded in a 12-Gaussian RBF over cos ∈ [−1, 1]: cos is linear in
the bond-vector dot product, and the basis is naturally densest where real
bond angles cluster (90°, 109.5°, 180°). Two consumption modes exist (§3.3).

---

## 3. The model: where chemistry enters the *architecture*

Backbone (per the benchmark configs): embed 14 → 64, three message-passing
layers, pool, MLP head. ~49k parameters poly-off, ~89–139k with the optional
channels — deliberately small relative to the data.

### 3.1 The message: atoms and bonds fuse by concatenation

For each directed edge (center *i* ← neighbor *j*):

```
m_ij = EdgeNet( [ h_i ‖ h_j ‖ e_ij ] )        EdgeNet: Linear(2d+7 → 64) → LN → softplus
                                                       → Linear(64 → d) → LN → softplus
```

The concatenation is positional: the first weight block reads the center, the
second the neighbor, the third the (center-oriented, standardized) physical
edge vector — so the message function is *directional*, as bonds are.
Standardization applies **only** to this MLP input; the aggregation below
reads the raw, physical weights.

### 3.2 Aggregation: chemistry decides how much each bond counts

Three interchangeable aggregations (`edge_aggregation`), which are the
experiment axis of the benchmark suites:

- **`ecn_weighted` (default).** A convex combination with **no learned
  parameters in the weighting**: `a_i = Σ_j (w_ij / Σw) · m_ij`, where `w_ij`
  is the center's own ECoN weight (bonds) or `shared_count` (poly edges). The
  inductive bias is explicit: *Hoppe's bond-valence answer to "how much does
  this neighbor matter" is taken as the attention distribution.* On 49k
  samples this physical prior beat or tied every learned alternative.
- **`attention`.** A learned scalar score per message — the ablation that asks
  whether the physical weighting is actually load-bearing.
- **`set_transformer`** ("idea A"). The neighbor shell is treated as a *set of
  tokens* `[h_j ‖ e_ij]` doing multi-head self-attention, and — the
  geometrically-motivated part — **the attention logit between neighbors j and
  k is biased by the bond angle ∠(j–i–k)**, via a per-head linear map of the
  angle's RBF expansion. This is the crystal analog of relative positional
  encoding: *the "relative position" of two neighbors of the same center IS
  the angle between their bonds.* A learned center query then reads the
  attended shell out into the atom's message. Heads can specialize (one on
  short strong bonds, one on angular geometry, one on ionicity contrast…).
  This injects 3-body geometry *inside* the attention mechanism rather than
  appending it as a feature. Data-hungry by construction — its verdict belongs
  to the 1.4M-frame MPtrj benchmark, not the 49k one (where it overfits).

The intensive (sum-to-1) weighting deliberately discards *total* coordination
strength, so an optional channel (`use_coord_magnitude`) re-injects
`log1p(Σ_j w_ij)` through a learned projection — under- vs over-coordination
is real chemistry (bond-valence sums) that a normalized average cannot see.

### 3.3 Where the angles go (two modes)

1. `use_bond_angles`: each bonding edge gets the **mean RBF(cos θ)** over the
   triplets it participates in at that center, concatenated to its 7 base
   features (edge dim 7 → 19). Cheap, but lossy: a perfect and a distorted
   octahedron can average to similar vectors.
2. `set_transformer`: the per-pair angle matrix biases attention directly
   (§3.2) — no averaging, every individual angle reaches the model.

### 3.4 Two relation types, fused

Bond and polyhedral messages come from **independent** message functions
(separate EdgeNets — chemically distinct relations should not share weights),
then combine in a single residual update:

```
h_i ← softplus( LayerNorm( bond_msg + poly_msg + h_i ) )          poly_fusion: "sum"
h_i ← softplus( LayerNorm( g_b⊙bond + g_p⊙poly + h_i ) )          poly_fusion: "gate"
```

The gate (`g_b, g_p = σ(Linear([bond ‖ poly]))`) lets the model weigh bonding
vs polyhedral context per atom and per channel; it is zero-initialized with
bias +2 so training *starts* at ≈0.88·(sum) — the additive baseline — and
learns selectivity rather than beginning half-attenuated.

### 3.5 Readout

Per-atom states pool into one crystal vector (`atom_pooling`): `mean`
(intensive — correct default for per-atom-normalized targets like eV/atom),
`mean_max` (the max channel surfaces a single decisive site, e.g. one
superconducting layer in a layered cell), per-atom `attention`, or `set2set`.
All scatter-based over a segment vector (no per-crystal loops). Then a small
MLP head; regression targets are z-normalized (optionally log1p) with the
inverse applied for reporting.

---

## 4. Honest limitations (and where the roadmap points)

- **No positions in the forward pass → no forces.** Energy is a function of
  precomputed invariants, so ∂E/∂x ≡ 0 through this model. The MPtrj
  energy work is the warm-up; force learning requires the equivariant phase
  (claude.md roadmap #5).
- **3-body is provably incomplete** (Pozdnyakov & Ceriotti 2020): distinct
  environments can share all distances and angles. Adequate for scalar
  targets in practice; dihedrals are stored for a 4-body extension.
- **Hand-crafted features inherit their assumptions**: ECoN/Shannon-radius
  logic presumes near-equilibrium ionic bonding and is noisiest for
  metallic/intermetallic systems and off-equilibrium frames — exactly where a
  from-scratch model has the advantage. This is a measured bet, not a free
  lunch.
- **Sum/gate fusion is the current ceiling on the poly channel** — the
  many-body upgrade (coordination polyhedra as hyperedges, set-attention at
  the polyhedron scale) is roadmap #7.

## 5. Empirical receipts

Choices above that were validated rather than assumed (paired bootstrap on a
shared test set, `scripts/compare_runs.py`): poly channel ≈ −6% test MAE
(significant); `coord_sphere` redundant at |r| ≈ 0.90 (removed); edge
orientation matters (src/tgt ECoN r ≈ 0.53); physical `ecn_weighted` ≥ learned
attention at 49k samples; set-transformer overfits at 49k (train 0.013 / val
0.058) — its real test, with the gate and coordination-magnitude channels, is
the running 6-arm MPtrj benchmark (`configs/mptrj_benchmark_suite/`).
