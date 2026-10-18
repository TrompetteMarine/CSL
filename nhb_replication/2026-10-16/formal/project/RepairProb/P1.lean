import RepairTasks
import Mathlib.Probability.Notation
import Mathlib.MeasureTheory.Function.ConditionalExpectation.Real
import Mathlib.MeasureTheory.Function.ConditionalExpectation.PullOut

/-! P1: heterogeneous conditional signed mean and the fixed-private-score field
Jacobian.  Nothing here moves a community average through the nonlinear
readout `G`; the private gaps `q_i` stay individual. -/

namespace CSLRepair

open MeasureTheory

lemma sigmoid_pos (x : ℝ) : 0 < sigmoid x := by
  unfold sigmoid; positivity

lemma sigmoid_lt_one (x : ℝ) : sigmoid x < 1 := by
  unfold sigmoid
  rw [div_lt_one (by positivity)]
  linarith [Real.exp_pos (-x)]

lemma continuous_sigmoid : Continuous sigmoid := by
  unfold sigmoid
  exact continuous_const.div (continuous_const.add (Real.continuous_exp.comp continuous_neg))
    (fun x => by positivity)

/-- One agent: if the action indicator has conditional mean `p` given `m`, and
the decisiveness `s(p)` is `m`-measurable and bounded, the conditional mean of
the signed decisiveness `s(p)(2·1_A - 1)` is `s(p)(2p - 1)`. -/
lemma condexp_signed_decisiveness {Ω : Type*} {m mΩ : MeasurableSpace Ω} (hm : m ≤ mΩ)
    (P : Measure Ω) [IsProbabilityMeasure P] (A : Set Ω) (hA : MeasurableSet A)
    (p d : Ω → ℝ) (hdm : Measurable[m] d) (hd : ∀ ω, |d ω| ≤ 1)
    (hcond : P[A.indicator (fun _ => (1 : ℝ)) | m] =ᵐ[P] p) :
    P[fun ω => d ω * (2 * A.indicator (fun _ => (1 : ℝ)) ω - 1) | m] =ᵐ[P]
      fun ω => d ω * (2 * p ω - 1) := by
  set I : Ω → ℝ := A.indicator (fun _ => (1 : ℝ)) with hI
  have hIi : Integrable I P := (integrable_const _).indicator hA
  have hgi : Integrable (fun ω => 2 * I ω - 1) P := (hIi.const_mul 2).sub (integrable_const _)
  have hdsm : StronglyMeasurable[m] d := hdm.stronglyMeasurable
  have hprod : Integrable (d * fun ω => 2 * I ω - 1) P :=
    Integrable.bdd_mul hgi (hdsm.mono hm).aestronglyMeasurable
      (c := 1) (Filter.Eventually.of_forall (fun ω => by simpa using hd ω))
  have h1 := condExp_mul_of_stronglyMeasurable_left hdsm hprod hgi
  have h2 : P[fun ω => 2 * I ω - 1 | m] =ᵐ[P] fun ω => 2 * p ω - 1 := by
    have hs := condExp_sub (hIi.const_mul 2) (integrable_const (1 : ℝ)) m (μ := P)
    have hc := condExp_smul (μ := P) (2 : ℝ) I m
    filter_upwards [hs, hc, hcond] with ω hs hc hp
    have e1 : (fun ω => 2 * I ω - 1) = (fun ω => 2 * I ω) - fun _ => (1 : ℝ) := rfl
    have e2 : (fun ω => 2 * I ω) = (2 : ℝ) • I := rfl
    rw [e1, hs, Pi.sub_apply, e2, hc, condExp_const hm]
    simp [hp]
  filter_upwards [h1, h2] with ω h1 h2
  change P[d * fun ω => 2 * I ω - 1 | m] ω = _
  rw [h1, Pi.mul_apply, h2]

/-- Choice gap `Δ_i = q_i + λ (K ψ)_c` of agent `i = ⟨c, j⟩`. -/
noncomputable def choiceGap {C : Type*} [Fintype C] {n : C → ℕ} (K : C → C → ℝ) (lam : ℝ)
    (q : (Σ c, Fin (n c)) → ℝ) (ψ : C → ℝ) (i : Σ c, Fin (n c)) : ℝ :=
  q i + lam * act K ψ i.1

/-- P1 (conditional signed mean).  Pre-trial information is a sub-σ-algebra
`m`.  The private gaps `q_i` and the lagged community masses `ψ` are
`m`-measurable, the score map is measurable and maps `[0,1]` into `[0,1]`, and
each action event `A_i` has conditional probability
`p_i = σ(h Δ_i)` given `m`.  Then the conditional mean of the next community
signed mass is the community average of the individual readouts
`G(q_i + λ (K ψ)_c)`, with `G = weightedReadout score h`.  No independence
between agents is needed, and no equality of private gaps is assumed. -/
theorem conditional_signed_mean {C : Type*} [Fintype C] (n : C → ℕ)
    {Ω : Type*} {m mΩ : MeasurableSpace Ω} (hm : m ≤ mΩ)
    (P : Measure Ω) [IsProbabilityMeasure P]
    (K : C → C → ℝ) (lam h : ℝ) (score : ℝ → ℝ) (hsm : Measurable score)
    (hs01 : ∀ p, 0 ≤ p → p ≤ 1 → 0 ≤ score p ∧ score p ≤ 1)
    (q : (Σ c, Fin (n c)) → Ω → ℝ) (hq : ∀ i, Measurable[m] (q i))
    (ψ : C → Ω → ℝ) (hψ : ∀ c, Measurable[m] (ψ c))
    (A : (Σ c, Fin (n c)) → Set Ω) (hA : ∀ i, MeasurableSet (A i))
    (hcond : ∀ i, P[(A i).indicator (fun _ => (1 : ℝ)) | m] =ᵐ[P]
      fun ω => sigmoid (h * choiceGap K lam (fun j => q j ω) (fun d => ψ d ω) i))
    (c : C) :
    P[fun ω => (1 / (n c : ℝ)) * ∑ j : Fin (n c),
        score (sigmoid (h * choiceGap K lam (fun j => q j ω) (fun d => ψ d ω) ⟨c, j⟩)) *
          (2 * (A ⟨c, j⟩).indicator (fun _ => (1 : ℝ)) ω - 1) | m] =ᵐ[P]
      fun ω => (1 / (n c : ℝ)) * ∑ j : Fin (n c),
        weightedReadout score h (choiceGap K lam (fun j => q j ω) (fun d => ψ d ω) ⟨c, j⟩) := by
  set pr : (Σ c, Fin (n c)) → Ω → ℝ :=
    fun i ω => sigmoid (h * choiceGap K lam (fun j => q j ω) (fun d => ψ d ω) i) with hpr
  have hprm : ∀ i, Measurable[m] (pr i) := by
    intro i
    simp only [hpr, choiceGap, act]
    exact continuous_sigmoid.measurable.comp (by fun_prop)
  have hd : ∀ i ω, |score (pr i ω)| ≤ 1 := by
    intro i ω
    have := hs01 (pr i ω) (sigmoid_pos _).le (sigmoid_lt_one _).le
    rw [abs_le]; constructor <;> linarith [this.1, this.2]
  have hterm : ∀ j : Fin (n c), P[fun ω => score (pr ⟨c, j⟩ ω) *
      (2 * (A ⟨c, j⟩).indicator (fun _ => (1 : ℝ)) ω - 1) | m] =ᵐ[P]
      fun ω => score (pr ⟨c, j⟩ ω) * (2 * pr ⟨c, j⟩ ω - 1) :=
    fun j => condexp_signed_decisiveness hm P (A ⟨c, j⟩) (hA _) (pr ⟨c, j⟩)
      (fun ω => score (pr ⟨c, j⟩ ω)) (hsm.comp (hprm _)) (fun ω => hd _ ω) (hcond _)
  have hint : ∀ j ∈ (Finset.univ : Finset (Fin (n c))), Integrable (fun ω => score (pr ⟨c, j⟩ ω) *
      (2 * (A ⟨c, j⟩).indicator (fun _ => (1 : ℝ)) ω - 1)) P := by
    intro j _
    have hgi : Integrable (fun ω => 2 * (A ⟨c, j⟩).indicator (fun _ => (1 : ℝ)) ω - 1) P :=
      (((integrable_const _).indicator (hA _)).const_mul 2).sub (integrable_const _)
    exact Integrable.bdd_mul hgi ((hsm.comp (hprm _)).mono hm le_rfl).aestronglyMeasurable
      (c := 1) (Filter.Eventually.of_forall (fun ω => by simpa using hd _ ω))
  have hsum := condExp_finset_sum hint m (μ := P)
  have hsmul := condExp_smul (μ := P) (1 / (n c : ℝ))
    (∑ j : Fin (n c), fun ω => score (pr ⟨c, j⟩ ω) *
      (2 * (A ⟨c, j⟩).indicator (fun _ => (1 : ℝ)) ω - 1)) m
  have hall : ∀ᵐ ω ∂P, ∀ j : Fin (n c), P[fun ω => score (pr ⟨c, j⟩ ω) *
      (2 * (A ⟨c, j⟩).indicator (fun _ => (1 : ℝ)) ω - 1) | m] ω =
      score (pr ⟨c, j⟩ ω) * (2 * pr ⟨c, j⟩ ω - 1) := ae_all_iff.mpr hterm
  have e : (fun ω => (1 / (n c : ℝ)) * ∑ j : Fin (n c),
        score (pr ⟨c, j⟩ ω) * (2 * (A ⟨c, j⟩).indicator (fun _ => (1 : ℝ)) ω - 1)) =
      (1 / (n c : ℝ)) • (∑ j : Fin (n c), fun ω => score (pr ⟨c, j⟩ ω) *
        (2 * (A ⟨c, j⟩).indicator (fun _ => (1 : ℝ)) ω - 1)) := by
    funext ω; simp [Finset.sum_apply]
  change P[fun ω => (1 / (n c : ℝ)) * ∑ j : Fin (n c),
        score (pr ⟨c, j⟩ ω) * (2 * (A ⟨c, j⟩).indicator (fun _ => (1 : ℝ)) ω - 1) | m] =ᵐ[P] _
  rw [e]
  filter_upwards [hsmul, hsum, hall] with ω h1 h2 h3
  rw [h1, Pi.smul_apply, h2, Finset.sum_apply, smul_eq_mul]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  rw [h3 j]
  rfl

/-- The fixed-private-score field Jacobian
`J = λ · diag((1/n_c) Σ_{i ∈ c} G'_i) · K`, as a continuous linear map. -/
noncomputable def fieldJacobian {C : Type*} [Fintype C] (n : C → ℕ) (K : C → C → ℝ)
    (lam : ℝ) (Gd : (Σ c, Fin (n c)) → ℝ) : (C → ℝ) →L[ℝ] (C → ℝ) :=
  ContinuousLinearMap.pi fun c =>
    (lam * ((1 / (n c : ℝ)) * ∑ j : Fin (n c), Gd ⟨c, j⟩)) •
      ∑ d, K c d • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : C => ℝ) d

/-- Entries of the field Jacobian: `J_{cd} = λ ((1/n_c) Σ_{i∈c} G'_i) K_{cd}`. -/
theorem fieldJacobian_apply {C : Type*} [Fintype C] (n : C → ℕ) (K : C → C → ℝ)
    (lam : ℝ) (Gd : (Σ c, Fin (n c)) → ℝ) (v : C → ℝ) (c : C) :
    fieldJacobian n K lam Gd v c =
      lam * ((1 / (n c : ℝ)) * ∑ j : Fin (n c), Gd ⟨c, j⟩) * act K v c := by
  simp [fieldJacobian, act, Finset.sum_apply]

/-- P1 (field Jacobian).  Hold every private gap `q_i` fixed and view the
conditional mean signed mass as a function of the lagged field `ψ`.  At any
point where every individual readout derivative exists — the derivative of
`G` at agent `i`'s own choice gap is `Gd i` — the map is Fréchet
differentiable with derivative `λ · diag(community mean of Gd) · K`.
Different agents may sit at different points of `G`; nothing is linearised
at a common or mean gap. -/
theorem field_jacobian {C : Type*} [Fintype C] (n : C → ℕ) (K : C → C → ℝ) (lam : ℝ)
    (G : ℝ → ℝ) (q : (Σ c, Fin (n c)) → ℝ) (ψ : C → ℝ) (Gd : (Σ c, Fin (n c)) → ℝ)
    (hG : ∀ i, HasDerivAt G (Gd i) (choiceGap K lam q ψ i)) :
    HasFDerivAt (fun z : C → ℝ => fun c =>
        (1 / (n c : ℝ)) * ∑ j : Fin (n c), G (choiceGap K lam q z ⟨c, j⟩))
      (fieldJacobian n K lam Gd) ψ := by
  rw [hasFDerivAt_pi']
  intro c
  set L : (C → ℝ) →L[ℝ] ℝ :=
    lam • ∑ d, K c d • ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : C => ℝ) d with hL
  have hinner : ∀ j : Fin (n c),
      HasFDerivAt (fun z : C → ℝ => choiceGap K lam q z ⟨c, j⟩) L ψ := by
    intro j
    have hlin : (fun z : C → ℝ => choiceGap K lam q z ⟨c, j⟩) =
        fun z => q ⟨c, j⟩ + L z := by
      funext z
      simp [hL, choiceGap, act]
    rw [hlin]
    exact (L.hasFDerivAt).const_add _
  have hsum := HasFDerivAt.sum (u := Finset.univ)
    (fun j _ => (hG ⟨c, j⟩).comp_hasFDerivAt ψ (hinner j))
  have hfin := hsum.const_mul (1 / (n c : ℝ))
  convert hfin using 1
  · funext z; simp [Finset.sum_apply]
  · ext v
    simp only [ContinuousLinearMap.comp_apply, ContinuousLinearMap.proj_apply,
      fieldJacobian_apply, ContinuousLinearMap.smul_apply, ContinuousLinearMap.sum_apply,
      smul_eq_mul, hL]
    simp [act, Finset.mul_sum, Finset.sum_mul]
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by ring

end CSLRepair
