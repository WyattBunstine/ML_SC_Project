# GPSTransformer: a hybrid local/global crystal encoder

**DESIGN DRAFT + BUILD LOG.** This document specifies the full architecture;
Increment 1 — the local/global encoder + energy regression of §4–§6 — is
**implemented** (see the build log in §10), as are the multi-task heads (§9) and
position-differentiable conservative-autograd forces/stress (§7/§9) — built in the
multi-task increment (branch `gps-tier2`, 2026-06-18). Only the register/global-token
coupling (§5.1) and the ECoN-prior logits remain design ahead of implementation.
Companion docs: `../../claude.md`
(research roadmap — this is "our encoder" in Phase 2), `../MPNN/ARCHITECTURE.md`
(the chemically-motivated featurization and the validated message-passing pieces
this model reuses), `../head/` (the small T_c head this model feeds).

---

## 1. What this is, and why it exists

`GPSTransformer` is **our own crystal encoder** — the learned alternative to the
borrowed MACE-MP-0 backbone in the superconductor transfer-learning pipeline. It
maps a crystal structure to **contextualized per-atom embeddings** `(N_atoms,
D_enc)` and is interchangeable with MACE beneath the *same* frozen-encoder
contract the T_c head already consumes (see `../head/`).

The strategic position (from the roadmap):

- The end target, T_c, is data-starved (~5.8k labels). We therefore **pretrain
  the encoder on abundant labels**, freeze it (or lightly fine-tune), and train
  only a ~10k-parameter head on T_c. The encoder can be large *because it is
  pretrained and then frozen* — its parameters are not charged against the T_c
  label budget.
- **The differentiator vs. MACE is the pretraining target set, not just the
  architecture.** MACE is an energy/forces specialist. We pretrain on the
  **three physical channels that set T_c, per atom and jointly: forces (phonon
  leg), magnetic moment (spin leg), and electronic density of states near E_F
  (electronic leg)** — see §9. A force-only encoder (MACE) carries one leg; this
  carries all three, which is a pretraining MACE structurally cannot offer. The
  bet: an embedding supervised on all three channels spans more of the
  T_c-determining physics and transfers better — *especially* for the magnetic
  channel, which is entirely absent from MACE and is the dominant leg for the
  unconventional families where the frozen-MACE probe is worst (cuprate log-MAE
  0.92).
- **Invariant features, but position-differentiable.** Internal features stay
  invariant (distances/angles/weights — no irreps; correct for invariant scalar
  targets and keeps the validated attention/poly/physical machinery intact). But
  positions DO enter the forward pass so the energy is differentiable: forces
  come from `−∂E/∂x` via autograd (the SchNet/ALIGNN-FF trick — the gradient of
  an invariant scalar is automatically an equivariant vector). This is a
  deliberate change from an earlier "no positions, no forces" framing: forces are
  now a *core pretraining target*, and — critically — making every per-atom
  target position-differentiable means **coupling strengths are recoverable by
  autograd** (∂DOS/∂x ≈ the deformation potential ⟨I²⟩; ∂m/∂x the spin-lattice
  response), even without explicit coupling labels (§9). True e3nn/irrep
  equivariance remains out of scope unless force-accuracy SOTA becomes the goal.

### Why GPS (hybrid local + global)

The MPtrj benchmark established the single most important architectural fact we
have: the only aggregation that beat the physical-prior baseline was
**angle-biased local set attention** (`edge_aggregation: "set_transformer"`,
val MAE 0.0397 vs 0.0429), and the gain was *not* a parameter-count artifact.
GPSTransformer keeps that winning component as its **local channel** and adds a
**global channel** so crystal-wide context can shape the local readout — the
"GraphGPS" recipe (Rampášek et al. 2022): every block runs a local
message-passing/attention module and a global attention module in parallel and
fuses them. The motivation specific to us:

- Formation energy is *nearsighted* (why CHGNet/M3GNet do well with short
  cutoffs) — so on MPtrj the global channel should be roughly neutral; MPtrj is
  the **stability / training-dynamics check**, not where the global channel
  earns its keep.
- T_c is a crystal-*global* electronic property. The global channel is aimed at
  T_c, and its payoff is expected to show at fine-tuning, not on energy.
- Honest scope: with no inter-atomic geometry beyond bonds/polyhedra and no
  lattice vectors in the forward pass, the global channel captures global
  *chemical* context (the inventory and balance of environments), **not** global
  *geometric* structure (where environments sit, true long-range electrostatics).
  We scope its claim to "global chemical context," not long-range physics.

---

## 2. The contract this model must satisfy

Identical to the MACE plug (see `../head/embed_mace.py`):

- **Input:** one crystal (a cgv4 graph; at pretraining, a batch of them).
- **Output:** `(N_atoms, D_enc)` float tensor, **row i aligned to graph node i**
  (the same atom order the builder emits and the head's descriptor/label join
  assumes). Atom-order alignment is a hard correctness requirement and will carry
  the same per-structure assertion the MACE pass uses.
- **Frozen-transfer mode:** the per-atom embeddings are cached to disk in the
  exact `<id>.npy` layout `embed_mace.py` writes, so the head, the linear probe,
  the family-resolved metrics, and the `[MACE ‖ ours]` fusion arm all work with
  **zero head-side changes** — only the embedding *source* differs.
- **Pretraining mode:** the encoder is topped with a pooling + scalar head and
  trained end-to-end on formation energy; that head is discarded for transfer.

---

## 3. Inputs and features consumed

All from the existing `crystal_graph_v4` representation and packed store — **no
graph rebuild required** for v1 (everything below is already stored). Reuses the
`MPNN/MPNNData.py` extraction/collate path verbatim where possible.

| source | tensor (per crystal, padded in batch) | role |
|---|---|---|
| node features (14, `NODE_FEA_LEN`) | `(N, 14)` | atom token inputs (Z, oxidation, χ, ECoN, Shannon radius, cn_core, sharing-mode hist, …) |
| bond edges (7, `NBR_FEA_LEN`) | `(N, max_nbr, 7)` + idx | local channel relation 1 (center-oriented, standardized) |
| bond ECoN weights | `(N, max_nbr)` | physical attention prior (see §5.1) |
| bond-angle matrix `cos θ(j–i–k)` | `(N, max_nbr, max_nbr)` | local-channel attention bias (the validated angle bias) |
| poly edges (7, `POLY_FEA_LEN`) | `(N, max_poly, 7)` + idx | local channel relation 2 (inter-polyhedral) |
| `crystal_seg` (N,) + n_crystals | segment ids | within-crystal masking for the global channel (already emitted post-vectorized-readout) |

Standardization buffers (train-split mean/std) are installed in the model exactly
as in `CrystalMPNN`. Dihedrals are stored but **not consumed in v1** (they enter
only if/when we move featurization in-model in Phase 4; see §9).

---

## 4. Top-level structure

```
            cgv4 graph (nodes, bond edges, poly edges, angle bias, seg)
                                   │
              embed nodes 14 → d   │   (Linear + standardization buffers)
                                   ▼
                      h⁰  ∈ ℝ^{N×d}   atom tokens
                                   │
        ┌──────────────  × L GPS blocks  ──────────────┐
        │   h ← h + LocalAttn(LN h): bond(angle) + poly │   (local attn sublayer)
        │   h ← h + FFN(LN h)                           │   (local FFN sublayer)
        │   h ← h + GlobalAttn(LN h, crystal_seg)       │   (within-crystal attn)
        │   h ← h + FFN(LN h)                           │   (global FFN sublayer)
        │   [deferred] g ← register/global token → query│   (§5.1, §11 — not built)
        └───────────────────────────────────────────────┘
                                   │
                      h^L ∈ ℝ^{N×D_enc}   ── THE CONTRACT OUTPUT (per-atom embeddings)
                                   │
                ┌──────────────────┴───────────────────┐
   pretraining: pool → scalar energy head        transfer: cache h^L, freeze, feed head
                (discarded after pretraining)     (../head/ — linear probe / MLP / fusion)
```

`d = D_enc` (one width throughout; default target ~128–256, tuned at pretraining
scale where capacity is free). `L` ≈ 3–4 blocks.

---

## 5. The GPS block in detail

As built, each block is a **sequential** stack of pre-LayerNorm residual sublayers
(`h ← h + Sub(LN(h))`), in order: (1) a **local attention** sublayer — bond shell
attention (angle-biased) plus the poly shell attention, sharing one input LayerNorm
and summed into the residual; (2) a **local FFN**; (3) the optional **global**
within-crystal attention sublayer; (4) the **global FFN**. So the local channel is
a full transformer sublayer (attention + FFN) that mirrors the global channel. The
`local_transformer=False` flag (config `local_transformer`) instead collapses (1)+(2)
to the original Increment-1 fusion `h ← Softplus(LN(h + bond + poly))` with no local
FFN — kept as the validated-component ablation baseline. Fusion is sequential (local
then global), **not** the parallel GraphGPS variant (decision settled in §11).

### 5.1 Local channel (bond relation) — the validated component

This **is** `MPNN/MPNNModel.py::LocalSetTransformerAgg`, reused unchanged in
spirit: the neighbor shell of atom *i* is a set of tokens `[h_j ‖ e_ij]` doing
multi-head self-attention, with the **inter-neighbor bond angle ∠(j–i–k) added to
the attention logits** via a per-head RBF(cosθ) bias (relative positional
encoding for crystals). A center query reads the attended shell into the atom's
message.

Two upgrades over the standalone aggregator, both small:

1. **ECoN prior on the logits (optional, recommended).** Add `log w_ij` (the
   center's Hoppe ECoN weight) to the attention logits so the physical
   bond-valence prior *initializes* the attention distribution rather than being
   discarded — physics as a soft bias, not a hard constraint. This keeps the
   thing that made `ecn_weighted` strong at small scale available to the learned
   attention.
2. **Global-conditioned center query (the key GPS coupling).** The center query
   becomes a function of `[h_i ‖ g]`, where `g` is the crystal's global/register
   token from the previous block. This is the literal mechanism for "global
   context adjusts the local readout": what each atom looks for in its own shell
   is modulated by crystal-wide state.

### 5.2 Local channel (polyhedral relation)

The second relation type, kept as in the MPNN: an **independent** attention/
aggregation over each atom's polyhedral edges (separate weights — chemically
distinct relation), consuming the poly features (sharing mode, bridge-angle
mean/std, path type, …). Fused into the residual stream alongside the bond
message (sum or the existing learned gate, `poly_fusion`). No poly-pair geometry
bias in v1 (that needs a rebuild — Phase 5).

### 5.3 Global channel — within-crystal attention

Full multi-head self-attention over **all atom tokens of the same crystal**,
masked by `crystal_seg` so atoms never attend across crystals in a packed batch
(segmented/block-diagonal attention; reuses the seg tensor from the vectorized
readout). Unit cells are ~10–100 atoms, so this is cheap (≪ the bond/poly
attention cost). A small number of learnable **register tokens** (Darcet et al.
2023) are appended per crystal; they participate in the global attention and
their post-attention state becomes the crystal-level summary `g` that (a)
conditions the next block's local query (§5.1) and (b) is the natural readout
(§6).

Originally scoped with **no geometric positional bias** (permutation-invariant set
mixing over environment-aware tokens, global *chemical* context only). **Built
2026-06-17 (Tier 2): a PBC long-range distance bias** — the symmetric analog of the
local angle bias. Per crystal, min-image pairwise distances (`df = frac_i − frac_j;
df −= df.round(); dc = df·lattice; ‖dc‖`) are RBF-expanded and mapped to a per-head
additive bias on the global-attention logits, folded — with the padding mask — into
`nn.MultiheadAttention`'s float `attn_mask`. Computed once per forward (positions are
constant across blocks) and shared. Gated by `use_dist_bias` (default off; needs a
positioned pack — see §10); rotation/translation-invariant by construction; warns +
skips on a positionless (zero-lattice) batch. It also turns uniform all-to-all mixing
into locality-aware attention (an over-smoothing mitigation). Still expected ~neutral
on *energy* (nearsighted); its test is T_c.

### 5.4 FFN + norm

Position-wise `Linear(d→m·d)→act→Dropout→Linear(m·d→d)` with pre-LN residual. As
built there are **two per block** — one after the local attention sublayer and one
after the global (§5 intro) — each its own pre-LN residual. Expansion `m =
gps_ffn_mult` defaults to **2** (config `gps_ffn_mult`), not the 4 of a vanilla
transformer; activation is **softplus** (consistent with the MPNN; GELU remains a
tunable option). With `local_transformer=False` only the global FFN is present.

---

## 6. Readout & pretraining heads

- **Per-atom output `h^L` (N, D_enc):** the contract deliverable — cached for
  transfer, never pooled when used as a frozen encoder. This is the only thing
  transfer uses; every head below is **discarded after pretraining**.
- **Energy head — BUILT as a per-atom decomposition (default, 2026-06-17):**
  `E = mean_i head(h_i)` — a shared MLP maps each atom to a scalar energy, then
  averaged over the crystal (intensive, eV/atom). This is the standard MLIP
  "energy is a sum of local contributions" readout, and it keeps a high-contribution
  atom from being washed out by averaging embeddings before the nonlinear head
  (`per_atom_head=false` restores the older `mean`/`mean_max`-pool-then-head).
  Register-token / `g^L`-conditioned pooling remains deferred. **Forces = `−∂E/∂x`
  and stress = `∂E/∂strain` via autograd are implemented** (the positions-in-forward
  increment landed — §7/§9): the energy is differentiated through an in-forward
  Cartesian leaf (the differentiable column-replacement geometry).
- **Per-atom magnetic-moment head:** `h^L → scalar |m_i|` (collinear), the MPtrj
  magmom target (§9) — built.
- **Per-atom electronic head:** `h^L →` a per-atom Softplus DOS contribution, summed
  (`_segment_sum`) into the per-structure **total DOS** — the full 256-bin E_F-aligned
  spectrum over [-10,+5] eV, the MP-DOS target (§9). *(Built as the full spectrum, not
  the site `N_i(E_F)`/PDOS-window reduction once sketched in §9.4.)* **Bandgap** (a free
  per-frame scalar) is a built co-training auxiliary head.
- **Coupling outputs (derived, not separately supervised):** because every target
  above is a position-differentiable function, the transfer head can request
  `∂N_i(E_F)/∂x` (deformation-potential proxy) and `∂m_i/∂x` (spin-lattice
  response) by autograd — exposing electron-phonon / spin-phonon *coupling*
  features that no single target labels directly (§9, "ingredients vs.
  couplings").

---

## 7. Invariance & differentiability properties

- **Rotation & translation invariant features:** the network's internal features
  are built from distances/angles/weights, so scalar predictions (energy, |m|,
  N(E_F), T_c) are invariant. ✓
- **Position-differentiable:** raw positions (+ PBC images) DO enter the forward
  pass — but only *through* the invariant geometric quantities (bond lengths,
  angles, **ECoN weights in closed form**), which are themselves differentiable
  functions of position. So `−∂E/∂x` (forces) and the coupling derivatives
  (§6, §9) come from autograd, and they are correctly **equivariant** because the
  gradient of an invariant scalar is equivariant. No irreps needed.
- **Permutation equivariant** over atoms (per-atom outputs), **permutation
  invariant** crystal readout. ✓
- **What's still out of scope:** internal *tensor/irrep* equivariance
  (e3nn/MACE-style). We get equivariant forces via autograd, not via tensor
  features; this caps force *accuracy* below a high-L equivariant model but keeps
  the validated invariant machinery and avoids rebuilding MACE. The non-differentiable
  piece is the Voronoi *topology* (face existence) — it demotes to a fixed
  neighbor list per frame with smooth cutoff envelopes, the same epistemic status
  as any MLIP's neighbor list (§9 caveats).

---

## 8. Parameter budget & sizing

Two regimes, and they are governed by different budgets:

- **Encoder (this model):** pretrained then frozen → **not** charged against the
  ~5.8k T_c labels. Size it for the pretraining data (MPtrj, 1.4M frames): width
  128–256, L = 3–4, multi-head. Likely 0.3–2M params — fine, given the lesson
  from the scaled-set_transformer run is that *optimization* (warmup + grad clip,
  now in the MPNN trainer), not capacity, was the failure mode at this scale.
- **Head (downstream, `../head/`):** unchanged ~10k fresh-param budget; it sees
  only the frozen `(N, D_enc)` embeddings + the physical-descriptor bypass.

---

## 9. Multi-task per-atom pretraining (the core differentiator)

The encoder is pretrained to predict, **per atom and jointly with one shared
backbone**, the three physical channels that determine T_c. The premise: an
embedding forced to encode forces + magnetism + electronic-DOS-near-E_F spans the
phonon, spin, and electronic legs of superconductivity, so a light transfer head
can reach T_c. (Honest scope: this captures the *ingredients*; the *couplings*
between them are only partially supervised — see below.)

### 9.1 Targets and data sources (verified)

| target | physical leg | source | coverage / form |
|---|---|---|---|
| energy → **forces** (`−∂E/∂x`) | phonon (⟨ω²⟩/mass) | **MPtrj** | ~100% of frames; off-equilibrium; dense per-atom vector |
| **magnetic moment** `|m_i|` | spin (unconventional) | **MPtrj `magmom`** | **~14% of frames** (spin-polarized calcs only) — ragged; per-atom collinear scalar; off-equilibrium |
| **site DOS at E_F** `N_i(E_F)` | electronic (N(E_F) numerator of λ) | **Materials Project** projected DOS | equilibrium-only; separate dataset join |
| bandgap (optional aux) | electronic | MPtrj per-frame | free, off-equilibrium scalar |

Key data facts established 2026-06-15: MPtrj frames carry `force`, `stress`,
`magmom`, **and** `bandgap`. The extractor (`database/Extract_MPtrj.py`) once dropped
all but energy; that re-extraction is **DONE** — `packed_v4` now carries
force/magmom/stress + a bandgap index column (and exact per-edge `to_jimage`).
Forces and magmom are thus *off-equilibrium* (good — the
coupling derivatives ∂(·)/∂x are meaningful across configurations). DOS is the odd
one out: MP has it only for relaxed ground states, so the electronic channel is
**equilibrium-only** (see the coupling caveat in §9.4).

### 9.2 Curriculum (training order)

User proposal: warm up on magnetism, then introduce DOS. Refinement, given the
coverage table:
1. **Anchor on energy+forces+magmom together (all from MPtrj).** Forces are the
   dense (~100%), clean, geometry-defining signal — they should anchor the
   backbone, not come second; warming up on magmom *alone* would train on only
   ~14% of frames and waste the geometric signal. Magmom rides along from the
   start at near-zero marginal cost (same frames, masked where absent).
2. **Introduce MP DOS later** (the curriculum step that needs the second dataset +
   E_F alignment). This is where "warm up, then add DOS" is exactly right.
3. **(Phase 3) computed e-ph curriculum** (λ, ω_log from JARVIS-EPC / Marques /
   BETE-NET) as an optional bridge before T_c.

So: read the user's "warm up on magnetism" as "stage 1 = MPtrj (forces+magmom),
stage 2 = add MP DOS" — with forces explicitly in stage 1.

### 9.3 Coupling strengths via autograd (the gradient point)

We have **no direct labels** for the electron-phonon matrix elements ⟨I²⟩ that set
λ — they are *cross-derivatives* (∂electronic-structure/∂displacement), not any
single target. But because every per-atom head is a position-differentiable
function (§7), the transfer head can **compute** the coupling-relevant gradients by
autograd as derived features: `∂N_i(E_F)/∂x` (deformation-potential proxy),
`∂m_i/∂x` (spin-lattice response), and the force Jacobian / Hessian (force
constants). We expose these as optional inputs to the T_c head — encoding the
couplings the ingredients alone miss, without ever labeling them. (Cost: these are
Jacobian/VJP operations; restrict to on-site / nearest-neighbor blocks in practice.)

### 9.4 Honest caveats (principle-level)

- **Ingredients ≠ couplings.** Supervising the three channels at sampled configs
  gives the ingredients; T_c lives partly in their couplings. §9.3 *exposes*
  couplings via autograd but only the force channel is densely supervised
  off-equilibrium. Crucially, **DOS is equilibrium-only**, so ∂N(E_F)/∂x is an
  *unsupervised* model extrapolation — the electronic deformation potential is the
  weakest-supervised, most important quantity. (A future fix would need DOS on
  displaced frames, which MP lacks at scale.)
- **Magmom is a static proxy** for *dynamic* spin fluctuations and is least
  reliable (DFT+U/functional-dependent) for exactly the correlated systems
  (cuprates, heavy-fermion) that matter most — still strictly additive vs. a
  force-only encoder, just not the full story.
- **DOS target shaping (decision reversed at build time):** the implemented target is
  the **whole 256-bin DOS(E) spectrum** on a fixed E_F-aligned grid ([-10,+5] eV), not
  the N(E_F)/narrow-PDOS-window reduction once preferred here — it is what
  `Download_MP_dos.py` resamples and the per-atom Softplus head reconstructs.
- **Ragged multi-task labels + loss balancing:** train on the *union* of MPtrj
  (force+magmom) and MP-static (DOS), masking absent targets per sample; the
  per-target scale/noise differences make loss weighting (or gradient-surgery /
  uncertainty weighting) a real, load-bearing design choice.

### 9.5 The payoff: per-channel attribution

Separate pretraining heads let us **ablate which channel the T_c transfer relies
on, per family** — the falsifiable prediction being conventional families lean on
force+DOS, unconventional on magnetism. This turns the single
conventional/unconventional differential into a per-mechanism attribution: a
stronger scientific result than "encoder A beats B."

### 9.6 Other large DOS sources (beyond MP)

If MP's DOS coverage is thin: **JARVIS-DFT** (NIST, ~80k materials with DOS, and it
*also* carries DFT electron-phonon / superconductivity data — doubles as the Phase-3
e-ph source); **AFLOW** (largest, millions of entries, DOS via the AFLUX API for a
large fraction); **NOMAD** (heterogeneous aggregator) and **Materials Cloud / MC3D**
(curated subsets). OQMD is mostly thermodynamics, less DOS. (Sizes approximate, Jan
2026 knowledge — verify current coverage before committing.)

---

## 10. What is reused vs. new

**STRUCTURE (built 2026-06-16): GPS and MPNN are fully independent siblings.**
Shared, model-agnostic infrastructure was extracted into **`models/common/`**
(`data.py` — graph/packed loaders, collate, samplers, feature-stats, the angle-RBF
constants, `crystal_seg`; `pack.py`; `resmon.py`; `train.py` — the regression loop
+ `Normalizer`/`AverageMeter`/warmup-clip/checkpoint). Both `models/MPNN/` and
`models/GPSTransformer/` import from `common`; **neither imports the other.**

**Reused from `models/common` (shared, not duplicated):**
- `common/data.py` — graph/packed loaders, `_assemble_sample`, `collate_pool`,
  `build_angle_bias`, the packed store, `crystal_seg`, feature-stats, `ANGLE_RBF_*`.
- `common/train.py` — `run_regression` loop (SWA/warmup/clip/telemetry),
  `Normalizer`, `_to_input_var`, checkpointing.

**GPS-specific, implemented standalone in this folder (model.py):** the
angle-biased shell attention is **ported** (own `ShellAttention`, not imported
from MPNN) so GPS owns its model end-to-end; plus `WithinCrystalAttention`,
`FeedForward`, `GPSBlock`, `GPSCrystalNet`, the poly channel, pooling + MLP head.

**Built (this folder):**
- `model.py` — `GPSCrystalNet` (+ `ShellAttention`, `WithinCrystalAttention`,
  `FeedForward`, `GPSBlock`). Increment 1: local angle-biased shell attention +
  poly channel + interleaved within-crystal global attention, energy regression,
  invariant. Verified end-to-end (200-graph MP_Energy smoke, global on/off).
  **Update (2026-06-16): the local channel was promoted to a full pre-LN transformer
  sublayer** — clean residual shell attention (`h ← h + Attn(LN h)`) + its own FFN,
  mirroring the global channel. The `local_transformer` flag (default on; config
  `local_transformer`) reverts to the original fusion for ablation. Smoke-verified
  forward+backward across `local_transformer` × `gps_global` (adds ~265k params at
  d=128/L=4 from the per-block local FFNs).
  **Update (2026-06-17): Tier 2 / Increment 2 — distance bias, per-atom head,
  ablation ladder (branch `gps-tier2`).**
  - **`per_atom_head` (default)**: readout is now `E = mean_i head(h_i)` — each atom
    gets a scalar energy, then averaged (the standard MLIP energy-is-a-sum-of-local-
    contributions decomposition), so a high-contribution atom isn't washed out by
    averaging embeddings before the nonlinear head. `per_atom_head=false` restores
    pool-then-head (the only path that enables `mean_max`). `_segment_mean` accumulates
    in fp32 (bf16 autocast would drop low-order terms over many atoms).
  - **PBC distance bias** on the global attention (§5.3): `use_dist_bias` +
    `dist_cutoff`/`n_dist_rbf`. Needs a *positioned* pack (see below).
  - **Ablation gates** (one model, one factor per rung): `use_bond_edges`,
    `use_angle_bias` (gates the bond shell's angle term), `shell_aggregation ∈
    {attention, mean}` (a NEW non-attention mean baseline), and `n_conv=0` (raw atoms
    → head). Drives `configs/gps_ablation_suite/01–08` (raw → +bond → +poly →
    +attention → +angle → +local-transformer → +global → +distance-bias).
  - **Stability**: `gps_eform.json` at lr 0.003 diverged (loss spike → constant-
    predictor collapse); fixed to lr 3e-4 / warmup 3ep / grad-clip 0.5 + bf16 amp +
    size-grouped batching (`max_atoms_per_batch`) to bound the poly-shell O(N·M²) memory.
  - Guard: `scripts/smoke_dataset.py` (end-to-end dataset→dims→collate→forward, both
    backends × both models) — the per-sample tuple grew 6→8 to carry positions; run it
    before any data-layer change.
- `gps_main.py` — thin trainer entry: builds the model, drives `common`'s loaders
  + `run_regression`. CLI: `python main.py train-gps configs/gps/gps_eform.json`.
  **Update (2026-06-18): Increment 3 — multi-task physics pretraining (branch
  `gps-tier2`).** All co-trained at once over a masked union, then the encoder is
  frozen for the T_c probe:
  - **Conservative autograd forces/stress** — `−∂E/∂cart` and `∂E/∂strain` via a
    differentiable **column-replacement geometry** (recompute bond-length /
    length-over-ΣR / angle-cos from a Cartesian leaf using exact per-edge `to_jimage`;
    topology/Voronoi/chemistry held fixed). fp32 force path (bf16 ruins derivatives);
    the dist-bias is detached from the force graph to bound double-backward memory.
  - **Multi-task heads** live inside `GPSCrystalNet` (no separate wrapper): per-atom
    magmom, per-structure bandgap, per-structure total DOS (per-atom Softplus →
    `_segment_sum`, 256-bin E_F-aligned spectrum). `tasks=None` keeps the single-scalar
    path bit-identical.
  - **`run_multitask`** in `common/train.py` (additive; `run_regression` untouched):
    builds the `cart`/`strain` leaves per batch, std-normalized masked weighted loss,
    per-step NaN abort. `gps_main` branches on the config `tasks` list.
  - **Masked-union `ConcatMTDataset`** (`common/data.py`): `packed_v4` (MPtrj:
    energy/forces/stress/magmom/bandgap) ∪ the DOS pack (relaxed MP: dos), absent
    targets NaN-masked per sample. Material ids dedup across packs (no split leakage).
  - **Transfer**: `model.encode()` (per-atom `h` with `cart=None`, static, no autograd)
    + `models/head/embed_gps.py` / `main.py embed-gps` export per-structure `.npy` in
    the embed-mace layout. `GPSCrystalNet.from_args` is the single arch-spec source
    (gps_main + embed_gps build through it).
  - Configs: `configs/gps/gps_multitask.json` (full union) + the **signal** ablation
    ladder `configs/gps_mt_ablation_suite/01–04` (energy → +forces/stress →
    +magmom/bandgap → +DOS), distinct from the *architecture* ladder above.
  - Gates: `scripts/verify_autograd_forces.py`, `verify_multitask_train.py`,
    `verify_union_masking.py` (plus the existing `smoke_dataset.py`).

**Still to write (later increments), in build order:**
- **register tokens** + **global→local query conditioning** (the `g` coupling) in
  `GPSBlock` (§5.1, §11).
- **ECoN-prior logits** on the local attention (§5.1).

**Data work — DONE:**
- **positions (frac_coords + lattice), 2026-06-17.** `_compact_v4_graph` had been
  DROPPING them at build time. Regenerated WITHOUT a Voronoi rebuild:
  `deploy.sh augment-positions` backfills them from the source MPtrj structures
  (atom-order verified by Z) → re-pack. `use_dist_bias`/rung 08 train on `packed_v2`.
- **MPtrj physics + exact PBC images → `packed_v4`.** `Extract_MPtrj.py` now keeps
  `force`/`magmom`/`stress` (+ `bandgap` index); a full MPtrj REBUILD restores the
  builder's exact `edge["to_jimage"]` (`deploy.sh build-mptrj` → `augment-physics` →
  `pack-mptrj $SCRATCH_MPTRJ_PACK_V4`). The to_jimage *recompute* was proven
  ambiguous for multi-image bonds, hence the rebuild.
- **Materials-Project DOS pull → the DOS pack.** `database/Download_MP_dos.py` /
  `main.py fetch-dos` attaches the resampled **full** DOS spectrum (not site `N(E_F)`)
  by material id; `pack-dataset` → `database/datafiles/MP/dos_pack`
  (`has_dos`/`has_positions`/`has_to_jimage` true), shipped by `deploy.sh sync-dos-pack`.
  Coverage 31,403/49,280 relaxed-MP materials. (JARVIS-DFT/AFLOW remain §9.6 fallbacks.)

**Integration DONE (2026-06-16):** `scripts/deploy.sh` ships `models/common/` +
`models/GPSTransformer/`; `main.py train-gps` runs `gps_main.py`; GPS has a
dedicated trainer entry (not an MPNN dispatch) — the right call given the coming
multi-task / position-differentiable divergence.

---

## 11. Open design decisions (to settle before/while implementing)

1. **Block fusion order — SETTLED (2026-06-16): sequential.** Built as local→global,
   each a full pre-LN transformer sublayer (attention + FFN, §5), not the parallel
   GraphGPS variant. The local channel is itself a complete attention+FFN sublayer
   (`local_transformer`, default on). Revisit parallel only if one channel is found
   to dominate at pretraining scale.
2. **How `g` conditions the local query:** concat `[h_i ‖ g]` then project
   (cheap, explicit) vs. FiLM-style modulation. *Leaning concat-project for v1.*
3. **Register tokens:** how many (1 vs. 2–4) and whether the readout is the
   register only vs. register ‖ mean ‖ max.
4. **ECoN prior:** additive `log w` bias vs. learned scalar gate over it.
5. **Width / depth / heads** — swept at pretraining scale (capacity is free
   there); `atom_feat_len % heads == 0` constraint inherited from the local attn.
6. **Over-smoothing risk** from stacking global attention — mitigations: few
   blocks (L≤4), pre-LN residuals, register tokens instead of all-pairs hubs.
7. **Poly cost** at `max_poly=64`: the poly attention/bias is the memory hotspot
   (it already pushed data-wait 18s→43s/epoch in the benchmark); may cap or
   sparsify.

---

## 12. Success criteria

- **Pretraining (multi-task):** trains stably (warmup/clip), each channel beats its
  trivial baseline — energy/forces near the 126k set_transformer (≤~0.04 eV) or
  better, magmom and N(E_F) clearly above mean-predictors — *jointly*, without one
  task collapsing the others (the loss-balancing check).
- **Transfer (the real test):** under the *same* head as MACE, on 3DSC
  family-resolved T_c — competitive overall MAE, and specifically whether the
  multi-physics embedding closes the conventional/unconventional gap the frozen
  MACE (force-only) probe shows. The `[MACE ‖ ours]` fusion arm reveals
  complementary signal even if the head-to-head is a loss.
- **Mechanistic (the unique deliverable):** per-channel attribution (§9.5) — does
  T_c transfer lean on force+DOS for conventional families and on magmom for
  unconventional, as the physics predicts?

## 13. Cheap pre-tests before the build (de-risk first)

Both use data already in hand and need no new pretraining infra; run them before
committing to the multi-task build:
1. **DOS/magnetism as head inputs.** The 3DSC CSV already carries `dos_2`,
   `efermi_2`, `magmoms_2`, `total_magnetization_2`. Compute N(E_F)/magnetization
   summaries, add them to the existing T_c head (alongside MACE + phys bypass), and
   see if they help — and whether the gain concentrates by family as predicted. If
   yes, baking them into the *embedding* via multi-task pretraining is justified.
2. **MACE-only vs. phys-only vs. [MACE‖phys]** ablations on the current head — the
   complementarity test for explicit physics (from the prior discussion).
```
