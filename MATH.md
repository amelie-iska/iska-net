# Mathematical Note: UMA-Guided Sequence-Only Structure-Dynamics Prediction

This note formalizes how the sequence-first UGM curriculum uses UMA-style oracle feedback to train a graph-token model to predict output structure-dynamics *as graph-state records* from SELFIES and FASTA-style inputs only. In the first run, the model is not trained on PDB/mmCIF/SDF trajectory files, coordinates, distance maps, energies, force vectors, or MD trajectories. UMA enters as an external temperature-conditioned scoring oracle for sampled candidate graph states.

## 1. Objects

Let the four molecular input modalities be

$$
\mathcal M=\{\mathrm{SELFIES},\mathrm{protein},\mathrm{RNA},\mathrm{DNA}\}.
$$

An input example is a typed graph-record object

$$
X_G=(G_X,R_X,\Gamma_X,T),
\qquad T\in[300,400]\ \mathrm K,
$$

where `SELFIES`, protein FASTA, RNA sequence, DNA sequence, function descriptions, motifs, prompt tokens, and tool context are graph records. The sequence-only policy imposes

$$
X_G \cap \{\mathrm{coordinates},\mathrm{forces},\mathrm{energies},\mathrm{PDB},\mathrm{mmCIF},\mathrm{SDF},\mathrm{trajectory}\}=\varnothing.
$$

The model predicts an output graph state, not a structure file:

$$
p_\theta(Y_G\mid X_G,T).
$$

During the first curriculum, the relevant structure-dynamics output is a proxy graph

$$
Y_G^{\mathrm{dyn}}
=
\{r_k:\ r_k\in
\mathcal R_{\mathrm{proxy}}
\cup \mathcal R_{\mathrm{attn}}
\cup \mathcal R_{\mathrm{coupling}}
\cup \mathcal R_{\mathrm{motion}}
\cup \mathcal R_{\mathrm{oracle}}\},
$$

with record families such as

```text
SEQ_STRUCT_DYN_PROXY:temperature_conditioned
ATTN_BIN:sequence_to_motion:b48
TOKEN_COUPLING:uma:temperature_oracle:b57
UMA_INFLUENCE:uma:trajectory_physics:b52
TOKEN_MOTION:uma:refine:b45
UMA_TRAJ_BIN:refine:b45
UGM:oracle:uma_feedback
```

These records are the trainable interface by which the model learns how sequence and motif evidence should route, couple, and evolve under UMA feedback.
They are stage-specific: plain SELFIES/FASTA/RNA/DNA reconstruction and function-description rows may include temperature and function-description graph nodes, but they do not emit these fine-bin targets unless the UMA oracle structure-dynamics-proxy stage is active.

Function-description records are part of the conditioning graph. Sources such as NatureLM/SFM-style rows, UniGenX-style sequence/function metadata, ProTrek-style sequence-function pairs, UniProt annotations, GO terms, EC labels, and InterPro descriptions supply nodes

$$
F_G=\{\text{function claim},\text{evidence span},\text{ontology label},\text{annotation source}\}.
$$

The first-run model therefore learns

$$
p_\theta(Y_G^{\mathrm{dyn}}\mid X_{\mathrm{seq}},F_G,T),
$$

not merely \(p_\theta(Y_G^{\mathrm{dyn}}\mid X_{\mathrm{seq}},T)\). This matters because UMA-guided candidate graph states should be conditioned on the intended or observed biological function, not only on raw sequence tokens.

## 2. Graph-State Evolution

Reasoning is treated as graph-state evolution. At step \(t\), define

$$
S_t=(G_t,Z_t,C_t,T),
$$

where \(G_t\) is the partially decoded graph record set, \(Z_t\in\mathbb R^{n_t\times d}\) are hidden graph-token states, \(C_t\) is controller/tool state, and \(T\) is the continuous Kelvin condition. An action

$$
a_t\in\mathcal A(S_t)
$$

adds, edits, masks, verifies, or refines records:

$$
S_{t+1}=\mathcal T_\theta(S_t,a_t,\xi_t).
$$

For structure-dynamics proxy prediction, actions include:

$$
\begin{aligned}
a_t \in \{&
\texttt{add\_motif},
\texttt{add\_attention\_bin},
\texttt{add\_uma\_coupling},
\texttt{add\_token\_motion},\\
&\texttt{query\_UMA},
\texttt{repair\_candidate},
\texttt{branch\_thought},
\texttt{merge\_thought}\}.
\end{aligned}
$$

The hidden state update can be abstracted as

$$
Z_{t+1}
=F_\theta(Z_t,G_t,X_G,T)
+B_\theta(c_t,m_t,T),
$$

where \(c_t\) is a binned coupling record and \(m_t\) is a token-motion record. UMA does not directly supply gradients through coordinates; it supplies reward and feedback records that alter the learned distribution over \(c_t\), \(m_t\), and later graph states.

## 3. Temperature-Conditioned UMA Reward

Let a completed candidate proxy graph be \(Y\). UMA is treated as a black-box oracle

$$
\mathrm{UMA}: (X_G,Y,T)\mapsto u_T(Y;X_G)\in\mathbb R,
$$

where \(T\) is passed as a continuous input, not merely rounded to an anchor token. A general reward form is

$$
R_T(Y\mid X_G)
=
\exp\left(
\alpha\,u_T(Y;X_G)
-\lambda_v V(Y)
-\lambda_c C(Y)
-\lambda_\ell L(Y)
\right),
$$

where:

- \(V(Y)\) penalizes invalid graph records, parser failures, valence/string failures, impossible motif links, and tool-verifier failures.
- \(C(Y)\) penalizes inconsistency between sequence, motif, function-description, attention-bin, and motion records.
- \(L(Y)\) penalizes excessive graph size or tool calls.
- \(u_T\) is the temperature-conditioned UMA score.

If UMA exposes an energy-like quantity \(E_{\mathrm{UMA}}(Y,T)\), a Boltzmann-style reward can be used:

$$
R_T(Y\mid X_G)
=
\exp\left(
-\frac{E_{\mathrm{UMA}}(Y,T)}{k_B T}
-\lambda_v V(Y)
-\lambda_c C(Y)
-\lambda_\ell L(Y)
\right).
$$

The current codebase implements this as a FairChem/UMA oracle interface for production oracle stages. It does not train on a dataset of ground-truth energies or forces. A deterministic proxy backend exists only for unit tests and smoke runs that explicitly set `UGM_UMA_BACKEND=proxy`.

## 4. Attention-Bin Coupling

Let \(A_\theta^{(\ell,h)}(S_t)\in[0,1]^{n_t\times n_t}\) be a model attention matrix at layer \(\ell\), head \(h\), when exposed by the backend. The curriculum defines a target coupling prior

$$
P_T(i,j\mid S_t)
$$

over token pairs, derived from typed graph incidence, motif membership, temperature, verifier state, and UMA feedback. For example:

This section applies only to the UMA oracle structure-dynamics-proxy stage. It is not part of the ordinary sequence/function pretraining objective, where temperature conditioning can appear as a graph feature without creating 64-bin attention, coupling, influence, motion, or trajectory targets.

$$
P_T(i,j)\ \text{large if}\ 
i=\text{temperature token},\ j=\text{UMA oracle token},
$$

and

$$
P_T(i,j)\ \text{large if}\
i=\text{sequence motif or function-description token},\ j=\text{motion/refinement token}.
$$

The continuous prior is discretized into \(B=64\) ordinal bins

$$
b_T(i,j)=\mathrm{bin}\left(P_T(i,j)\right)
\in\{0,\ldots,63\}.
$$

These become graph records:

$$
\texttt{ATTN\_BIN:route:b00},\ldots,\texttt{ATTN\_BIN:route:b63}.
$$

The 64-way convention is deliberately analogous to distogram-style outputs in
modern structure prediction systems: the target is not a single scalar edge
strength but a categorical distribution over ordinal relation bins. AlphaFold 3
documents distogram outputs with shape \((n,n,64)\) in its public output
documentation (<https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md>);
UGM uses the same kind of fine-grained discretization for graph-state routing
records, not for supervised structure labels.

When real attention matrices are available, the auxiliary loss is

$$
\mathcal L_{\mathrm{attn}}
=
\sum_{\ell,h}
\lambda_{\ell,h}
\mathrm{KL}\!\left(P_T\ \middle\|\ A_\theta^{(\ell,h)}+\epsilon\right).
$$

When the backend does not expose attention weights, the same coupling information is still used through random-order autoregressive prediction of the binned records:

$$
\mathcal L_{\mathrm{bin}}
=
-\sum_{r\in \mathcal R_{\mathrm{attn}}\cup\mathcal R_{\mathrm{coupling}}}
\log p_\theta(r\mid X_G,G_{<r},T).
$$

This is why attention-bin training is compatible with a standard TokenGT block.

The model can also predict a full categorical distribution over the bins. For
route \(q\), let

$$
\rho_\theta^{q}(b\mid S_t,T)
=
\mathrm{softmax}(g_\theta^q(S_t,T))_b,
\qquad b\in\{0,\ldots,63\}.
$$

If the UMA oracle returns or induces a target bin distribution
\(\rho_{\mathrm{UMA}}^q\), for example by scoring several candidate graph-state
motions, the training loss can be

$$
\mathcal L_{\mathrm{bin64}}
=
\sum_q
\mathrm{CE}\left(\rho_{\mathrm{UMA}}^q,\rho_\theta^q\right)
\quad\text{or}\quad
\sum_q
\mathrm{KL}\left(\rho_{\mathrm{UMA}}^q\,\|\,\rho_\theta^q+\epsilon\right).
$$

Coarse labels such as low, medium, high, and critical may be retained as
diagnostic aliases, but they are not the primary training target.

## 4.1 Embedding-Distribution Geometry and Jensen-Shannon Distance

The hidden embedding vectors themselves also contain geometry. For visible
graph-token states \(z_i\in\mathbb R^d\), define probability vectors

$$
p_i=\mathrm{softmax}(z_i/\tau_h).
$$

For \(p_i,p_j\), define

$$
m_{ij}=\frac12(p_i+p_j),
$$

and

$$
\mathrm{JS}(p_i,p_j)
=
\frac12\KL(p_i\|m_{ij})
+
\frac12\KL(p_j\|m_{ij}).
$$

The Jensen-Shannon distance

$$
d_{\mathrm{JS}}(i,j)=\sqrt{\mathrm{JS}(p_i,p_j)}
$$

is bounded and symmetric. It gives a distributional geometry after softmax,
complementary to Euclidean hidden-state distance

$$
d_E(i,j)=\|z_i-z_j\|_2.
$$

The monitor therefore builds two distograms:

$$
D_E=\mathrm{hist}\{d_E(i,j):i<j\},
\qquad
D_{\mathrm{JS}}=\mathrm{hist}\{d_{\mathrm{JS}}(i,j):i<j\}.
$$

It also tracks the correlation

$$
\mathrm{corr}\left(d_E,d_{\mathrm{JS}}\right)
$$

to detect whether geometric separation in embedding space agrees with
distributional separation after softmax. An optional regularizer is

$$
\mathcal L_{\mathrm{JS}}
=
\left[\delta_{\mathrm{JS}}-\mathbb E_{i<j}d_{\mathrm{JS}}(i,j)\right]_+^2,
$$

which discourages hidden-state collapse in probability geometry. This is an
ablation switch, not a claim that Jensen-Shannon distance alone measures
physical structure.

## 4.2 Evolving Attention Maps as Fold-Contact Fields

The central mechanism for sequence-only structure-dynamics prediction is that
attention maps are treated as an evolving contact field, not merely as
explanatory heatmaps. Let \(A_t^{\ell,h}\in[0,1]^{n\times n}\) be the attention
matrix over sequence, motif, atom-slot, function, oracle, and motion tokens at
graph-state step \(t\). Define a symmetrized attention contact field

$$
C_t^{A}(i,j)
=
\sum_{\ell,h}w_{\ell h}
\frac12\left(A_t^{\ell,h}(i,j)+A_t^{\ell,h}(j,i)\right).
$$

Hidden-state geometry supplies two additional contact fields:

$$
C_t^{E}(i,j)=\exp\left(-\frac{\|z_i^t-z_j^t\|_2^2}{\sigma_E^2}\right),
\qquad
C_t^{JS}(i,j)=\exp\left(-\frac{d_{\mathrm{JS}}(p_i^t,p_j^t)^2}{\sigma_{JS}^2}\right).
$$

The model's fold-contact proxy is the convex mixture

$$
C_t(i,j)
=
\alpha_A C_t^A(i,j)
+\alpha_E C_t^E(i,j)
+\alpha_{JS} C_t^{JS}(i,j),
\qquad
\alpha_A+\alpha_E+\alpha_{JS}=1.
$$

Long-range sequence pairs with persistent high \(C_t(i,j)\) are candidate
folding contacts. The temporal change

$$
\Delta C_t(i,j)=C_{t+1}(i,j)-C_t(i,j)
$$

is a latent folding motion: increasing contact mass means the model is bringing
two token families together in graph-state space; decreasing mass means it is
separating them. In a protein row, \(i,j\) may be residue or motif tokens. In a
SELFIES row, they may be atom-slot or functional-group tokens. In RNA/DNA rows,
they may be base, loop, family, or pairing-hypothesis tokens.

The coordinate/frame decoder should then emit a candidate graph \(Y_t\) whose
rendered distances agree with the contact field. For generated coordinates
\(x_i^f\), define a soft rendered contact

$$
\widehat C_f(i,j)
=
\sigma\left(\frac{r_c-\|x_i^f-x_j^f\|_2}{s_c}\right),
$$

where \(r_c\) is a contact radius and \(s_c\) is a softness parameter. A
self-consistency term can be applied to generated candidates:

$$
\mathcal L_{\mathrm{contact\text{-}coord}}
=
\sum_{f,i<j}
\left(C_t(i,j)-\widehat C_f(i,j)\right)^2.
$$

This is not supervision from a ground-truth PDB file. It ties the model's own
evolving attention/embedding/JS contact hypothesis to the coordinates it emits.
UMA then scores the generated candidate graph at temperature \(T\). If
the candidate is physically poor, its GFlowNet reward drops, and the action
trajectory that produced the attention contacts and coordinates receives less
flow. If the candidate is chemically and physically plausible under UMA, the
attention-contact evolution that produced it is reinforced. Over training,
attention maps can therefore learn to encode fold-like contact formation from
sequence/function context, even though no training row supplies a target
structure.

## 5. UMA-Modulated Token Motion

Token-motion records summarize how latent graph states should evolve after oracle feedback. Let

$$
m_t=(a_t,b_t),
\qquad
a_t\in
\{\mathrm{explore},\mathrm{diversify},\mathrm{expand},
\mathrm{contract},\mathrm{refine},\mathrm{stabilize},
\mathrm{trajectory\_follow},\mathrm{oracle\_accept},\mathrm{oracle\_reject}\},
\qquad
b_t\in\{0,\ldots,63\}.
$$

A temperature-conditioned prior is

$$
\pi_0(m\mid T)
=
\begin{cases}
\text{higher mass on refine/stabilize}, & T\approx 300\ \mathrm K,\\
\text{higher mass on explore/diversify}, & T\approx 400\ \mathrm K.
\end{cases}
$$

UMA feedback updates this prior:

$$
\pi_{\theta}^{\mathrm{motion}}(m_t\mid S_t,X_G,T)
\propto
\pi_0(m_t\mid T)
\exp\left(\eta\,\widehat Q_\theta(S_t,m_t,T)\right),
$$

where \(\widehat Q_\theta\) is learned from verifier and UMA-scored rollouts. The hidden graph-token state follows a learned motion map:

$$
Z_{t+1}
=
Z_t+\Delta_\theta(Z_t,G_t,m_t,T).
$$

The motion map is not a molecular dynamics simulator. It is a latent graph-state transition mechanism trained to choose records and reasoning moves that UMA rewards.

The stronger trajectory proxy objective is:

$$
\mathcal R_{\mathrm{motion}}
=
\{
\texttt{TOKEN\_MOTION:uma:action:b00\ldots b63},
\texttt{UMA\_TRAJ\_BIN:action:b00\ldots b63}
\}.
$$

UMA scoring at continuous temperature \(T\) should influence both the selected
action \(a_t\) and the bin \(b_t\). A low-temperature oracle trace should push
probability mass toward high bins for stabilization, refinement, rejection of
invalid candidates, and trajectory-following moves. A high-temperature trace
should push more mass toward exploration, diversification, and candidate
acceptance when UMA scores remain plausible.

## 6. GFlowNet Training

A trajectory is

$$
\tau=(S_0,a_0,S_1,a_1,\ldots,S_T=Y).
$$

The forward policy samples graph-state actions:

$$
P_F(\tau\mid X_G,T)
=
\prod_{t=0}^{T-1}P_F(S_{t+1}\mid S_t,X_G,T).
$$

The trajectory-balance objective is

$$
\mathcal L_{\mathrm{TB}}
=
\left[
\log Z_\theta(X_G,T)
+\sum_{t=0}^{T-1}\log P_F(S_{t+1}\mid S_t,X_G,T)
-\log R_T(Y\mid X_G)
-\sum_{t=0}^{T-1}\log P_B(S_t\mid S_{t+1},X_G,T)
\right]^2.
$$

At optimum in a finite acyclic construction graph,

$$
P_F(Y\mid X_G,T)
=
\frac{R_T(Y\mid X_G)}{Z(X_G,T)}.
$$

Thus high-UMA-reward graph states are sampled more often, while diversity is preserved because GFlowNets sample proportional to reward rather than greedily maximizing it.

## 7. Random-Order Autoregression

For output records

$$
\mathcal R(Y)=\{r_1,\ldots,r_n\},
$$

random-order autoregression trains

$$
p_\theta(Y\mid X_G,T,\pi)
=
\prod_{t=1}^{n}
p_\theta(r_{\pi_t}\mid X_G,r_{\pi_{<t}},T,\pi_{\le t}).
$$

The relevant first-run records include sequence tokens, motifs, function descriptions, reasoning states, coupling bins, motion priors, and oracle feedback:

$$
\mathcal R(Y)
\subset
\mathcal R_{\mathrm{seq}}
\cup\mathcal R_{\mathrm{motif}}
\cup\mathcal R_{\mathrm{reason}}
\cup\mathcal R_{\mathrm{attn}}
\cup\mathcal R_{\mathrm{coupling}}
\cup\mathcal R_{\mathrm{motion}}
\cup\mathcal R_{\mathrm{oracle}}.
$$

Structured reveal orders place oracle-enabling records early:

$$
\mathrm{seq/motif}
\rightarrow
\mathrm{attention/coupling}
\rightarrow
\mathrm{UMA\ influence}
\rightarrow
\mathrm{motion/trajectory\ bins}
\rightarrow
\mathrm{oracle}
\rightarrow
\mathrm{function/reasoning\ explanation}.
$$

This causes the model to learn that sequence, motif, and function-description evidence should first create a candidate graph-state route, then route through UMA scoring, then emit a temperature-conditioned motion/refinement hypothesis.

## 8. What “Predict Structure-Dynamics” Means Here

In the first run, structure-dynamics prediction means learning a conditional distribution over oracle-scored graph-state records:

$$
p_\theta(
\text{proxy dynamics records},
\text{coupling records},
\text{motion records},
\text{atom/coordinate/frame records},
\text{function/reasoning records}
\mid
\text{SELFIES/FASTA},T
).
$$

The intended output is still an actual structure-dynamics hypothesis: a typed all-atom graph with atom identities, bonds, and coordinate/frame records. The first-run restriction is only about the supervision source. The model is not trained by copying ground-truth PDB/mmCIF/SDF files, minimizing supervised RMSD to deposited structures, or imitating MD frames. Instead, it samples candidate structure-dynamics graph states from SELFIES/FASTA/RNA/DNA plus function context and receives temperature-conditioned UMA/verifier/GFlowNet feedback that rewards physically plausible candidates. Direct supervised coordinate, force-label, or MD-frame datasets can be added only in a later explicit policy phase; oracle-guided coordinate/frame prediction is part of the current target behavior, while generated-token PDB rendering is optional and not required for this pass.

The learning signal is:

$$
\nabla_\theta \mathbb E_{\tau\sim P_\theta}
\left[\log R_T(Y_\tau\mid X_G)\right],
$$

implemented through random-order record likelihood, verifier losses, and GFlowNet trajectory balance. In score-function form, the basic direction is

$$
\nabla_\theta J(\theta)
=
\mathbb E_{\tau\sim P_\theta}
\left[
\left(\log R_T(Y_\tau\mid X_G)-b(X_G,T)\right)
\nabla_\theta \log P_\theta(\tau\mid X_G,T)
\right],
$$

where \(b\) is a baseline for variance reduction. GFlowNet training replaces this greedy reinforcement picture with reward-proportional terminal sampling, which better supports diverse candidate graph states.

## 9. Consequence

UMA influences the model through three coupled paths:

1. **Coupling path:** In the UMA proxy stage, UMA-conditioned records teach which token families should exchange information, represented by 64-bin `ATTN_BIN`, `TOKEN_COUPLING`, and `UMA_INFLUENCE` records.
2. **Motion path:** In the same stage, UMA-conditioned rewards teach which latent graph-state moves are useful at a given temperature, represented by 64-bin `TOKEN_MOTION` and `UMA_TRAJ_BIN` records.
3. **Sampling path:** GFlowNet trajectory balance changes the terminal distribution so high-reward sequence-grounded graph states are sampled more often without collapsing to a single candidate.

The result is a sequence-only system that can learn to propose and refine structure-dynamics *hypotheses as graph states* from SELFIES and FASTA inputs, with UMA providing temperature-conditioned external pressure toward physically or chemically plausible candidates.

## 10. Token-Budgeted Source Selection

The expanded full-run corpus is selected under a hard untruncated graph-token
budget

$$
\sum_{d\in\mathcal D_{\mathrm{selected}}}
\sum_{x\in d}
\left(
|S_X(x)|+1+2|Y(x)|
\right)
\le 5\times 10^9,
$$

where \(S_X(x)\) is the serialized source graph-token set and \(Y(x)\) is the
random-order target record set. The factor \(2|Y|\) comes from the training
format: each supervised target is paired with a position/query token and a
revealed target token.

The selection objective is not simply maximum token count. Each candidate
dataset \(d\) receives a utility estimate

$$
U(d)=
w_g G(d)+w_s S(d)+w_f F(d)+w_v V(d)+w_\ell L(d)
-w_c C(d)-w_r R(d),
$$

where:

- \(G(d)\) measures explicit graph-state supervision.
- \(S(d)\) measures sequence/SELFIES/FASTA/RNA/DNA coverage.
- \(F(d)\) measures function-description or annotation grounding.
- \(V(d)\) measures verifier/tool/selection signal.
- \(L(d)\) measures general-language quality.
- \(C(d)\) penalizes compute and context cost.
- \(R(d)\) penalizes licensing, leakage, and schema risk.

The resulting ranked active additions are:

$$
\begin{aligned}
&\text{GraphWalks}
\succ \text{GraphInstruct}
\succ \text{PubChem10M SELFIES}
\succ \text{UniProt function text}\\
&\succ \text{Rfam}
\succ \text{RNAcentral}
\succ \text{DNA coding regions}
\succ \text{OpenMathReasoning TIR}\\
&\succ \text{OpenMathReasoning GenSelect}
\succ \text{DCLM 1B slice}.
\end{aligned}
$$

If the budget guard fails after graphification, lower-ranked high-volume
sources are capped first. GraphWalks stays active unless the run is explicitly
changed, because it directly supervises the graph-state evolution behavior that
ordinary text corpora do not provide.

The current manifest implements this as row caps:

$$
\begin{array}{c|c}
\text{source} & \text{default cap}\\
\hline
\text{PubChem10M SELFIES} & 8{,}000{,}000\\
\text{Rfam} & 1{,}000{,}000\\
\text{RNAcentral} & 500{,}000\\
\text{DNA coding regions} & 500{,}000\\
\text{OpenMathReasoning TIR} & 1{,}300{,}000\\
\text{OpenMathReasoning GenSelect} & 300{,}000
\end{array}
$$

Dataset Viewer sample rows put the expanded default near \(4.9\times 10^9\)
untruncated graph-sequence tokens when added to the old baseline. The exact
value is computed after graphification; the count guard, not the estimate, is
the acceptance criterion.
