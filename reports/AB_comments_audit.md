# A.B. comments — detailed audit against the current manuscript

Status key: **✓ fully addressed** · **◐ addressed as scoped (acknowledged extension)** ·
**△ newly strengthened this pass**. Locations refer to the current 29-page build.

| # | Comment | Status | Where / how |
|---|---------|--------|-------------|
| 1 | Clarify what is genuinely new | ✓ | *Contributions* (Intro): the novelty is the **coupling**, not the ingredients — confidence from the *same* decision process, regulating *both* private and social learning, credibility an *emergent, propagating* network property (not a fixed edge weight). Reinforced by *What the model changes relative to adjacent traditions* and the central hypothesis now stated in the opening paragraph. |
| 2 | Neuroscience as inspiration, not justification | ✓ | *Why confidence is socially transmitted* is built around A.B.'s three questions verbatim (why transmitted socially / which computations are shared / relation to teaching signals), citing Joiner et al. 2017, Lockwood–Klein–Flügge 2021, Niv 2019, Behrens et al. 2018, Rao–Six–Cortese–Banerjee 2025, and stating explicitly "we do *not* claim a neural implementation." Reinforced by *A neuroeconomic reading* + Fig. (neuromap) with a "functional analogy, not implementation" banner. |
| 3 | Confidence treated as perfectly observable → should be latent/inferred | △ | *Information structure* now reads confidence as "communicated **or behaviourally inferred** subjective reliability, not objective accuracy." **New this pass:** testable prediction (v) makes latent-confidence inference (reaction time, vacillation, reputation) an explicit, falsifiable lab prediction, and *A neuroeconomic reading* adds a hierarchical **latent-variable estimand** (infer each sender's trial-wise confidence, test whether the receiver weights the *inferred* confidence, not accuracy). The generative latent-confidence model remains flagged as the next modelling step. |
| 4 | Two social channels may not be independent | ◐ | Kept separate by design (Contribution ii, in prose/equations/code/ablations); *Limitations* explicitly notes anticipatory exposure may reweight later feedback, so the channels can interact. Appropriate for a baseline that isolates each. |
| 5 | Static community structure → adaptive/reputation | ◐ | *Limitations*: fixed weights vs. adaptive, reputation-driven trust flagged as a richer model. (Robustness R5 already perturbs the graph — weighted SBM, noisy W — but not adaptivity.) |
| 6 | Missing latent task states (belief-state / cognitive maps) | ◐ | *Cognitive interpretation*: the model tracks action values, not latent task states; relating credibility transmission to belief-state / cognitive-map inference (Niv 2019; Behrens et al. 2018) is named as a theoretical extension. Implementing it would change the model, so it is scoped out. |
| 7 | Confidence only affects learning → should affect exploration | ◐ | *Limitations*: confidence should also modulate exploration (low → explore, high → exploit). Acknowledged; partial coverage already exists because confidence enters choice through the augmented value / anticipatory field, but not exploration per se. |
| 8 | Missing normative benchmark | ✓ | *A normative benchmark*: vs. the non-social learner (λ=η=0) and the omniscient optimum (zero regret) — social learning cuts regret ~16× in the corrective regime (7,400→450) and inflates it ~2.5× in the distortive one (4,800→12,000). A precise beneficial-vs-pathological statement. |
| 9 | Give parameters cognitive meaning | ✓ | *Cognitive interpretation* + default-parameter table: β value sensitivity, σ internal decision noise, a speed–accuracy threshold, κ₁/κ₂ confidence weights, α reward learning rate, λ susceptibility to social influence, η retrospective credit assignment. |
| 10 | Emphasise testable predictions | ✓ / △ | *Testable predictions* now lists **five** (was four), each tied to a figure/manipulation, plus *A neuroeconomic reading* → a model-based design (Fig. experiment) with regressors and falsifiable signatures. Prediction (v) + the latent estimand are new this pass. |
| minor | Intro too long before the hypothesis | ✓ | Central hypothesis stated at the end of the first paragraph, pointing to the mechanism figure. |
| minor | Need a schematic information-flow figure | ✓ | Fig. 1 (mechanism). |
| minor | Reads like a formal spec; add intuition | ✓ | Proofs relocated to Appendix A; interpretation paragraphs added; the dynamical-systems view (bifurcation + two phase portraits) gives visual intuition. |
| minor | Organise the Monte-Carlo section around fewer hypotheses | ✓ | Results grouped into four regime claims + the phase diagram; the testable-predictions block ties analyses to hypotheses. |
| — | Move proofs to appendix | ✓ | Appendix A ("Proofs"). |

**Bottom line.** Comments 1, 2, 8, 9, 10 are fully addressed; 4, 5, 6, 7 are addressed as
deliberately scoped extensions (implementing them would enlarge the model the
comments themselves warn against enlarging); comment 3 was the one genuine
half-gap — acknowledged but not operationalised — and is now converted into an
explicit prediction and estimand. No comment is unaddressed.

---

# The "latent" story: what the paper can predict for a lab

The core move A.B. asked for is to stop treating a sender's confidence as directly
readable and instead treat it as a **latent variable the receiver infers**. The
model already generates confidence from the decision process,
`C = σ(κ₁|v|/a − κ₂·log(1+τ/τ₀))`, so it makes sharp, quantitative predictions
about which *observable cues* should drive credibility, and how to recover the
latent quantity statistically. Concretely:

## Manipulations (all orthogonalise confidence from accuracy)
1. **Reaction-time cue.** Because `C` decreases with decision time τ, a receiver
   using RT as a confidence proxy should follow fast senders more. Manipulate the
   sender's RT (speed pressure; or show/hide the sender's RT) at matched evidence
   and accuracy → following should fall monotonically with sender RT.
2. **Misleading-but-strong evidence.** Confidence tracks evidence *magnitude*, not
   correctness (U-shaped in the value contrast; see the decision-confidence
   figure). A "confident-error" condition (evidence strong but pointing to the
   wrong arm) yields fast, high-confidence, wrong senders who are over-followed →
   distortive transmission. This is the cell a purely accuracy-driven account
   cannot produce.
3. **Vacillation / hesitation cue.** Changes-of-mind and mouse-trajectory
   curvature signal low confidence; senders who visibly vacillate should be
   discounted, controlling for accuracy.
4. **Reputation reset.** If receivers infer reliability from history, resetting or
   spoofing a sender's past accuracy should shift following independently of
   current accuracy.
5. **Cue masking.** Hiding RT / adding response jitter / hiding reports should
   attenuate transmission even at fixed accuracy — a direct test that credibility
   runs through the inferred-confidence channel rather than through accuracy.

## Model-based (hierarchical latent-variable) analysis — the "latent style"
- Fit a generative model in which each sender's trial-wise latent confidence
  `Ĉ_{j,t}` is a function of RT, explicit report and running reliability, and the
  receiver's update is `η·g(Ĉ)·δ_soc`.
- **Test:** the receiver's social prediction-error weight loads on the *inferred*
  latent confidence `Ĉ`, not on observed accuracy — and, at matched `Ĉ`, adding
  accuracy explains no further variance. Failure of this ordering falsifies the
  credibility channel.
- **Model-based regressors** supplied by the theory for behaviour or neural
  signal: decision confidence `C_{i,t}`, private PE `δ_{i,t}`, and the
  credibility-weighted social PE.

## Predicted signatures (falsifiable)
- Influence ∝ sender confidence, *controlling for* correctness.
- Early over-confidence ⇒ persistent herding once λ exceeds the local
  amplification threshold λ⋆ (the early-lead lock-in).
- Higher decision noise / degraded evidence ⇒ weaker contagion.
- Manipulating confidence independently of accuracy changes network dynamics
  without changing individual accuracy.

## Design at a glance
A dyadic or small-network group-decision task with a 2×2 core (induced confidence
× accuracy) realising the confident-error cell, RT and choice-trajectory logging
for latent-confidence estimation, and optional model-based fMRI using the three
regressors above. This is exactly the protocol sketched in the experiment-design
figure, now with the latent-inference channel made explicit.

*Note (sensitive-topic framing not applicable).* These are standard
group-decision / value-learning paradigms; no clinical population is required.
