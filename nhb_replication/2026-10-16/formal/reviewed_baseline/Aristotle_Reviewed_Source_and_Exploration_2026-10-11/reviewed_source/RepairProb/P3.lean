import RepairProb.P1
import Mathlib.Probability.Martingale.BorelCantelli

/-! P3: uniform exploration floor, finite action-avoidance bound, and almost-sure
repeated sampling of both actions under indefinite continuation.

This is *not* a statement about efficient learning: the avoidance bound
`(1 - ε)^H` can be extremely close to one for long stretches when `ε` is small. -/

namespace CSLRepair

open MeasureTheory Filter
open scoped Topology

lemma one_sub_sigmoid (x : ℝ) : 1 - sigmoid x = sigmoid (-x) := by
  unfold sigmoid
  have h1 : 0 < 1 + Real.exp (-x) := by positivity
  have h2 : 0 < 1 + Real.exp x := by positivity
  rw [neg_neg]
  field_simp
  rw [show Real.exp x = Real.exp (-x) * Real.exp x * Real.exp x by
    rw [← Real.exp_add]; simp]
  rw [← Real.exp_add]; simp
  ring_nf
  rw [show Real.exp (-x) * Real.exp x = 1 by rw [← Real.exp_add]; simp]
  ring_nf

lemma sigmoid_monotone : Monotone sigmoid := by
  intro x y hxy
  unfold sigmoid
  apply one_div_le_one_div_of_le (by positivity)
  have : Real.exp (-y) ≤ Real.exp (-x) := Real.exp_le_exp.mpr (by linarith)
  linarith

/-- The exploration floor `ε* = σ(-h(1+λ))`. -/
noncomputable def explorationFloor (h lam : ℝ) : ℝ := sigmoid (-(h * (1 + lam)))

lemma explorationFloor_pos (h lam : ℝ) : 0 < explorationFloor h lam := sigmoid_pos _

lemma explorationFloor_lt_one (h lam : ℝ) : explorationFloor h lam < 1 := sigmoid_lt_one _

/-- P3 (bounded gap): scores in `[0,1]` and a lagged signed mass in `[-1,1]^M`
under a row-stochastic field give `|Δ| ≤ 1 + λ`. -/
theorem choice_gap_bound {C : Type*} [Fintype C] (K : C → C → ℝ) (hK : RowStochastic K)
    (lam : ℝ) (hlam : 0 ≤ lam) (Q1 Q0 : ℝ) (hQ1 : 0 ≤ Q1 ∧ Q1 ≤ 1) (hQ0 : 0 ≤ Q0 ∧ Q0 ≤ 1)
    (ψ : C → ℝ) (hψ : ∀ d, |ψ d| ≤ 1) (c : C) :
    |(Q1 - Q0) + lam * act K ψ c| ≤ 1 + lam := by
  have hf := field_sup_bound K hK ψ 1 hψ c
  have hq : |Q1 - Q0| ≤ 1 := by
    rw [abs_le]; constructor <;> linarith [hQ1.1, hQ1.2, hQ0.1, hQ0.2]
  calc |(Q1 - Q0) + lam * act K ψ c| ≤ |Q1 - Q0| + |lam * act K ψ c| := abs_add_le _ _
    _ = |Q1 - Q0| + lam * |act K ψ c| := by rw [abs_mul, abs_of_nonneg hlam]
    _ ≤ 1 + lam * 1 := by gcongr
    _ = 1 + lam := by ring

/-- P3 (exploration floor): under the same bounds, both conditional action
probabilities `σ(hΔ)` and `1 - σ(hΔ)` are at least `ε* = σ(-h(1+λ)) > 0`. -/
theorem exploration_floor {C : Type*} [Fintype C] (K : C → C → ℝ) (hK : RowStochastic K)
    (h lam : ℝ) (hh : 0 < h) (hlam : 0 ≤ lam) (Q1 Q0 : ℝ)
    (hQ1 : 0 ≤ Q1 ∧ Q1 ≤ 1) (hQ0 : 0 ≤ Q0 ∧ Q0 ≤ 1)
    (ψ : C → ℝ) (hψ : ∀ d, |ψ d| ≤ 1) (c : C) :
    explorationFloor h lam ≤ sigmoid (h * ((Q1 - Q0) + lam * act K ψ c)) ∧
      explorationFloor h lam ≤ 1 - sigmoid (h * ((Q1 - Q0) + lam * act K ψ c)) := by
  have hb := abs_le.mp (choice_gap_bound K hK lam hlam Q1 Q0 hQ1 hQ0 ψ hψ c)
  constructor
  · apply sigmoid_monotone
    nlinarith [hb.1]
  · rw [one_sub_sigmoid]
    apply sigmoid_monotone
    nlinarith [hb.2]

/-- Conditional probability of a complement. -/
lemma condExp_indicator_compl {Ω : Type*} {m mΩ : MeasurableSpace Ω} (hm : m ≤ mΩ)
    (P : Measure Ω) [IsProbabilityMeasure P] (A : Set Ω) (hA : MeasurableSet A) :
    P[Aᶜ.indicator (fun _ => (1 : ℝ)) | m] =ᵐ[P]
      fun ω => 1 - P[A.indicator (fun _ => (1 : ℝ)) | m] ω := by
  have e : Aᶜ.indicator (fun _ => (1 : ℝ)) = (fun _ => (1 : ℝ)) - A.indicator (fun _ => 1) :=
    Set.indicator_compl A _
  rw [e]
  filter_upwards [condExp_sub (integrable_const (1 : ℝ))
    ((integrable_const (1 : ℝ)).indicator hA) m (μ := P)] with ω hω
  rw [hω, Pi.sub_apply, condExp_const hm]

/-- Generic sequential-conditioning bound: if each event `E t` is decided by
trial `t` (measurable for `ℱ (t+1)`) and has conditional probability at least
`ε` given the pre-trial information `ℱ t`, then avoiding `E` for `H`
consecutive trials has probability at most `(1-ε)^H`. -/
theorem avoidance_bound {Ω : Type*} {mΩ : MeasurableSpace Ω} (ℱ : Filtration ℕ mΩ)
    (P : Measure Ω) [IsProbabilityMeasure P] (E : ℕ → Set Ω)
    (hE : ∀ t, MeasurableSet[ℱ (t + 1)] (E t)) (ε : ℝ) (hε1 : ε ≤ 1)
    (hε : ∀ t, ∀ᵐ ω ∂P, ε ≤ P[(E t).indicator (fun _ => (1 : ℝ)) | ℱ t] ω)
    (t0 H : ℕ) :
    P.real (⋂ t ∈ Finset.Ico t0 (t0 + H), (E t)ᶜ) ≤ (1 - ε) ^ H := by
  induction H with
  | zero => simp
  | succ H ih =>
    set S := ⋂ t ∈ Finset.Ico t0 (t0 + H), (E t)ᶜ with hS
    have hSm : MeasurableSet[ℱ (t0 + H)] S := by
      refine Finset.measurableSet_biInter _ (fun t ht => ?_)
      have ht' := (Finset.mem_Ico.mp ht).2
      exact (ℱ.mono (by omega : t + 1 ≤ t0 + H) _ (hE t)).compl
    have hSm' : MeasurableSet S := ℱ.le _ _ hSm
    have hEm : MeasurableSet (E (t0 + H)) := ℱ.le _ _ (hE _)
    have hsplit : (⋂ t ∈ Finset.Ico t0 (t0 + (H + 1)), (E t)ᶜ) = S \ E (t0 + H) := by
      ext ω
      simp only [hS, Set.mem_iInter, Set.mem_compl_iff, Set.mem_diff, Finset.mem_Ico]
      constructor
      · intro h
        exact ⟨fun t ht => h t ⟨ht.1, by omega⟩, h _ ⟨by omega, by omega⟩⟩
      · rintro ⟨h1, h2⟩ t ht
        rcases lt_or_eq_of_le (Nat.lt_succ_iff.mp (by omega : t < t0 + H + 1)) with h | h
        · exact h1 t ⟨ht.1, h⟩
        · rw [h]; exact h2
    have hint : ∫ ω in S, P[(E (t0 + H)).indicator (fun _ => (1 : ℝ)) | ℱ (t0 + H)] ω ∂P =
        P.real (E (t0 + H) ∩ S) := by
      rw [setIntegral_condExp (ℱ.le _) ((integrable_const _).indicator hEm) hSm,
        integral_indicator hEm]
      simp [Measure.real, Measure.restrict_apply hEm]
    have hlow : ε * P.real S ≤
        ∫ ω in S, P[(E (t0 + H)).indicator (fun _ => (1 : ℝ)) | ℱ (t0 + H)] ω ∂P := by
      have := setIntegral_mono_ae_restrict (μ := P) (s := S)
        (f := fun _ => ε) (integrableOn_const (by simp))
        (integrable_condExp.integrableOn) (ae_restrict_of_ae (hε (t0 + H)))
      simpa [setIntegral_const, mul_comm] using this
    have hadd := measureReal_diff_add_inter (μ := P) (s := S) hEm
    rw [hsplit, pow_succ]
    have hcomm : P.real (E (t0 + H) ∩ S) = P.real (S ∩ E (t0 + H)) := by rw [Set.inter_comm]
    have hS0 : 0 ≤ P.real S := measureReal_nonneg
    have h1e : 0 ≤ 1 - ε := by linarith
    nlinarith [mul_le_mul_of_nonneg_right ih h1e]

/-- Generic almost-sure repeated occurrence: under the same uniform conditional
lower bound `ε > 0`, the event `E t` occurs for infinitely many `t` almost surely
(Lévy's conditional Borel–Cantelli lemma). -/
theorem frequently_of_condexp_floor {Ω : Type*} {mΩ : MeasurableSpace Ω}
    (ℱ : Filtration ℕ mΩ) (P : Measure Ω) [IsProbabilityMeasure P] (E : ℕ → Set Ω)
    (hE : ∀ t, MeasurableSet[ℱ (t + 1)] (E t)) (ε : ℝ) (hεpos : 0 < ε)
    (hε : ∀ t, ∀ᵐ ω ∂P, ε ≤ P[(E t).indicator (fun _ => (1 : ℝ)) | ℱ t] ω) :
    ∀ᵐ ω ∂P, ∃ᶠ t in atTop, ω ∈ E t := by
  let s : ℕ → Set Ω := fun n => Nat.casesOn n ∅ E
  have hs : ∀ n, MeasurableSet[ℱ n] (s n) := by
    intro n
    cases n with
    | zero => exact @MeasurableSet.empty Ω (ℱ 0)
    | succ k => exact hE k
  have hlevy := ae_mem_limsup_atTop_iff P hs
  have hall : ∀ᵐ ω ∂P, ∀ t, ε ≤ P[(E t).indicator (fun _ => (1 : ℝ)) | ℱ t] ω :=
    ae_all_iff.mpr hε
  filter_upwards [hlevy, hall] with ω hω hb
  have hlim : ω ∈ limsup s atTop := by
    rw [hω]
    refine tendsto_atTop_mono (fun n => ?_)
      ((tendsto_natCast_atTop_atTop (R := ℝ)).atTop_mul_const hεpos)
    calc (n : ℝ) * ε = ∑ _k ∈ Finset.range n, ε := by simp
      _ ≤ _ := Finset.sum_le_sum (fun k _ => hb k)
  rw [← limsup_nat_add s 1] at hlim
  exact mem_limsup_iff_frequently_mem.mp hlim

/-- P3 (model level).  One agent in community `c` is tracked over trials
`t = 0, 1, 2, …`.  `ℱ t` is the pre-trial information of trial `t`; the
action-1 event `A t` is decided during trial `t`, so it is `ℱ (t+1)`-measurable.
The pre-trial scores satisfy `Q1 t, Q0 t ∈ [0,1]`, the lagged community signed
masses satisfy `|ψ t d| ≤ 1`, `K` is row stochastic, `h > 0`, `λ ≥ 0`, and the
conditional probability of action 1 given `ℱ t` is `σ(h Δ_t)` with
`Δ_t = Q1 t - Q0 t + λ (K ψ_t)_c`.  Then, with `ε* = σ(-h(1+λ))`:
(1) both conditional action probabilities are a.s. at least `ε*`;
(2) for every start `t0` and horizon `H`, avoiding action 1 (resp. action 0) on
the `H` trials `t0, …, t0+H-1` has probability at most `(1-ε*)^H`;
(3) almost surely each action is selected infinitely often. -/
theorem exploration_and_repeated_sampling {C : Type*} [Fintype C]
    {Ω : Type*} {mΩ : MeasurableSpace Ω} (ℱ : Filtration ℕ mΩ)
    (P : Measure Ω) [IsProbabilityMeasure P]
    (K : C → C → ℝ) (hK : RowStochastic K) (h lam : ℝ) (hh : 0 < h) (hlam : 0 ≤ lam)
    (c : C) (Q1 Q0 : ℕ → Ω → ℝ) (ψ : ℕ → C → Ω → ℝ)
    (hQ1 : ∀ t ω, 0 ≤ Q1 t ω ∧ Q1 t ω ≤ 1) (hQ0 : ∀ t ω, 0 ≤ Q0 t ω ∧ Q0 t ω ≤ 1)
    (hψ : ∀ t d ω, |ψ t d ω| ≤ 1)
    (A : ℕ → Set Ω) (hA : ∀ t, MeasurableSet[ℱ (t + 1)] (A t))
    (hchoice : ∀ t, P[(A t).indicator (fun _ => (1 : ℝ)) | ℱ t] =ᵐ[P]
      fun ω => sigmoid (h * ((Q1 t ω - Q0 t ω) + lam * act K (fun d => ψ t d ω) c))) :
    (∀ t, ∀ᵐ ω ∂P, explorationFloor h lam ≤ P[(A t).indicator (fun _ => (1 : ℝ)) | ℱ t] ω ∧
        explorationFloor h lam ≤ P[(A t)ᶜ.indicator (fun _ => (1 : ℝ)) | ℱ t] ω) ∧
    (∀ t0 H, P.real (⋂ t ∈ Finset.Ico t0 (t0 + H), (A t)ᶜ) ≤
        (1 - explorationFloor h lam) ^ H) ∧
    (∀ t0 H, P.real (⋂ t ∈ Finset.Ico t0 (t0 + H), A t) ≤
        (1 - explorationFloor h lam) ^ H) ∧
    (∀ᵐ ω ∂P, ∃ᶠ t in atTop, ω ∈ A t) ∧
    (∀ᵐ ω ∂P, ∃ᶠ t in atTop, ω ∉ A t) := by
  have hfloor : ∀ t, ∀ᵐ ω ∂P,
      explorationFloor h lam ≤ P[(A t).indicator (fun _ => (1 : ℝ)) | ℱ t] ω ∧
      explorationFloor h lam ≤ P[(A t)ᶜ.indicator (fun _ => (1 : ℝ)) | ℱ t] ω := by
    intro t
    filter_upwards [hchoice t, condExp_indicator_compl (ℱ.le t) P (A t) (ℱ.le _ _ (hA t))]
      with ω h1 h2
    have hb := exploration_floor K hK h lam hh hlam (Q1 t ω) (Q0 t ω) (hQ1 t ω) (hQ0 t ω)
      (fun d => ψ t d ω) (fun d => hψ t d ω) c
    rw [h2, h1]
    exact hb
  have hAc : ∀ t, MeasurableSet[ℱ (t + 1)] (A t)ᶜ := fun t => (hA t).compl
  have h1 : ∀ t, ∀ᵐ ω ∂P,
      explorationFloor h lam ≤ P[(A t).indicator (fun _ => (1 : ℝ)) | ℱ t] ω :=
    fun t => (hfloor t).mono fun _ h => h.1
  have h0 : ∀ t, ∀ᵐ ω ∂P,
      explorationFloor h lam ≤ P[(A t)ᶜ.indicator (fun _ => (1 : ℝ)) | ℱ t] ω :=
    fun t => (hfloor t).mono fun _ h => h.2
  have hle := (explorationFloor_lt_one h lam).le
  have hpos := explorationFloor_pos h lam
  refine ⟨hfloor, fun t0 H => avoidance_bound ℱ P A hA _ hle h1 t0 H, fun t0 H => ?_,
    frequently_of_condexp_floor ℱ P A hA _ hpos h1,
    frequently_of_condexp_floor ℱ P (fun t => (A t)ᶜ) hAc _ hpos h0⟩
  simpa using avoidance_bound ℱ P (fun t => (A t)ᶜ) hAc _ hle h0 t0 H

end CSLRepair
