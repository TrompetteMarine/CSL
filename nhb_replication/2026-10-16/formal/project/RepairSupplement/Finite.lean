import RepairProb

/-! Additional exact finite-model statements for the October 2026 revision.
These declarations do not assert a population-limit or invariant-law theorem.
The literal realised update is checked before induction in time. -/

namespace CSLRepair

open scoped BigOperators

theorem private_gain_bounds (d amin amax : ℝ)
    (hd0 : 0 ≤ d) (hd1 : d ≤ 1) (ha : amin ≤ amax) :
    amin ≤ alphaPlus d amin amax ∧ alphaPlus d amin amax ≤ amax ∧
    amin ≤ alphaMinus d amin amax ∧ alphaMinus d amin amax ≤ amax := by
  unfold alphaPlus alphaMinus
  refine ⟨?_,?_,?_,?_⟩ <;> nlinarith [mul_nonneg (sub_nonneg.mpr ha) hd0,
    mul_nonneg (sub_nonneg.mpr ha) (sub_nonneg.mpr hd1)]

theorem preference_score_bounds (p : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    (1/2:ℝ) ≤ max p (1-p) ∧ max p (1-p) ≤ 1 := by
  constructor
  · have h1 := le_max_left p (1-p)
    have h2 := le_max_right p (1-p)
    linarith
  · exact max_le hp1 (by linarith)

theorem logistic_decisiveness_bounds (x : ℝ) :
    (1/2:ℝ) ≤ max (sigmoid x) (1-sigmoid x) ∧
      max (sigmoid x) (1-sigmoid x) ≤ 1 :=
  preference_score_bounds (sigmoid x) (sigmoid_pos x).le (sigmoid_lt_one x).le

theorem realised_gain_bounds (q R d amin amax : ℝ)
    (hd0 : 0 ≤ d) (hd1 : d ≤ 1) (ha : amin ≤ amax) :
    amin ≤ (if 0 ≤ R-q then alphaPlus d amin amax else alphaMinus d amin amax) ∧
      (if 0 ≤ R-q then alphaPlus d amin amax else alphaMinus d amin amax) ≤ amax := by
  have hb := private_gain_bounds d amin amax hd0 hd1 ha
  split_ifs
  · exact ⟨hb.1,hb.2.1⟩
  · exact ⟨hb.2.2.1,hb.2.2.2⟩


theorem private_target_den_pos (μ d amin amax : ℝ)
    (hμ0 : 0 ≤ μ) (hμ1 : μ ≤ 1) (hd0 : 0 ≤ d) (hd1 : d ≤ 1)
    (hmin : 0 < amin) (ha : amin ≤ amax) :
    0 < μ*alphaPlus d amin amax+(1-μ)*alphaMinus d amin amax := by
  have hb := private_gain_bounds d amin amax hd0 hd1 ha
  nlinarith [mul_nonneg hμ0 (sub_nonneg.mpr hb.1),
    mul_nonneg (sub_nonneg.mpr hμ1) (sub_nonneg.mpr hb.2.2.1)]

theorem private_target_bounds (μ d amin amax : ℝ)
    (hμ0 : 0 ≤ μ) (hμ1 : μ ≤ 1) (hd0 : 0 ≤ d) (hd1 : d ≤ 1)
    (hmin : 0 < amin) (ha : amin ≤ amax) :
    0 ≤ driftTarget μ d amin amax ∧ driftTarget μ d amin amax ≤ 1 := by
  have hb := private_gain_bounds d amin amax hd0 hd1 ha
  have hp : 0 ≤ alphaPlus d amin amax := le_trans hmin.le hb.1
  have hm : 0 ≤ alphaMinus d amin amax := le_trans hmin.le hb.2.2.1
  have hden := private_target_den_pos μ d amin amax hμ0 hμ1 hd0 hd1 hmin ha
  unfold driftTarget
  constructor
  · exact div_nonneg (mul_nonneg hμ0 hp) hden.le
  · apply (div_le_one hden).mpr
    nlinarith [mul_nonneg (sub_nonneg.mpr hμ1) hm]


noncomputable def realisedUpdate (q R α b v w r ε : ℝ) : ℝ :=
  q + α * (R - q) + b * v / (w + ε) * (r - w * q)

/-- Bounds for the literal peer-plus-private update, with no clipping. -/
theorem realised_update_bounds (q R α amax b v w r ε : ℝ)
    (hq : 0 ≤ q ∧ q ≤ 1) (hR : 0 ≤ R ∧ R ≤ 1)
    (hα0 : 0 ≤ α) (hα1 : α ≤ amax) (hb : 0 ≤ b)
    (hv0 : 0 ≤ v) (hvw : v ≤ w) (hw1 : w ≤ 1)
    (hr0 : 0 ≤ r) (hrw : r ≤ w) (hε : 0 < ε)
    (hbudget : amax + b ≤ 1) :
    0 ≤ realisedUpdate q R α b v w r ε ∧
      realisedUpdate q R α b v w r ε ≤ 1 := by
  have hg := peer_gain_bounds b v w ε hb hv0 hvw hε
  have hw0 : 0 ≤ w := le_trans hv0 hvw
  have hgain := gain_budget α amax b (b*v/(w+ε)) w hα1 hg.1 hg.2 hw0 hw1 hbudget
  have hc := convex_update_bounds q r R w α (b*v/(w+ε))
    hq.1 hq.2 hR.1 hR.2 hα0 hg.1 hr0 hrw hgain
  have he : realisedUpdate q R α b v w r ε =
      (1-α-(b*v/(w+ε))*w)*q + α*R + (b*v/(w+ε))*r := by
    unfold realisedUpdate
    ring
  rwa [he]

/-- Any selected-arm mass made with nonnegative exposure of total at most one
has the constraints used in `realised_update_bounds`. This also covers exact
self exclusion, since the exposure total is then at most one. -/
theorem peer_mass_constraints {I : Type*} [Fintype I]
    (W a R d : I → ℝ) (hW : ∀ i, 0 ≤ W i) (hWsum : ∑ i, W i ≤ 1)
    (ha : ∀ i, 0 ≤ a i ∧ a i ≤ 1)
    (hR : ∀ i, 0 ≤ R i ∧ R i ≤ 1)
    (hd : ∀ i, 0 ≤ d i ∧ d i ≤ 1) :
    0 ≤ (∑ i, W i*a i) ∧ (∑ i, W i*a i) ≤ 1 ∧
    0 ≤ (∑ i, W i*a i*R i) ∧ (∑ i, W i*a i*R i) ≤ (∑ i, W i*a i) ∧
    0 ≤ (∑ i, W i*a i*d i) ∧ (∑ i, W i*a i*d i) ≤ (∑ i, W i*a i) := by
  have hwa : ∀ i, 0 ≤ W i*a i := fun i => mul_nonneg (hW i) (ha i).1
  have hw0 : 0 ≤ ∑ i, W i*a i := Finset.sum_nonneg (fun i _ => hwa i)
  have hw1 : (∑ i, W i*a i) ≤ 1 := by
    calc _ ≤ ∑ i, W i := Finset.sum_le_sum (fun i _ => by nlinarith [(ha i).2, hW i])
         _ ≤ 1 := hWsum
  refine ⟨hw0, hw1, ?_, ?_, ?_, ?_⟩
  · exact Finset.sum_nonneg (fun i _ => mul_nonneg (hwa i) (hR i).1)
  · exact Finset.sum_le_sum (fun i _ => by nlinarith [(hR i).2, hwa i])
  · exact Finset.sum_nonneg (fun i _ => mul_nonneg (hwa i) (hd i).1)
  · exact Finset.sum_le_sum (fun i _ => by nlinarith [(hd i).2, hwa i])

theorem signed_mass_bound (n : ℕ) (hn : 0 < n)
    (d y : Fin n → ℝ) (hd : ∀ i, |d i| ≤ 1) (hy : ∀ i, |y i| ≤ 1) :
    |(∑ i, d i*y i) / (n : ℝ)| ≤ 1 := by
  have hsum : |∑ i, d i*y i| ≤ (n : ℝ) := by
    calc _ ≤ ∑ i, |d i*y i| := Finset.abs_sum_le_sum_abs _ _
         _ ≤ ∑ _i : Fin n, (1:ℝ) := Finset.sum_le_sum (fun i _ => by
           rw [abs_mul]; exact mul_le_one₀ (hd i) (abs_nonneg _) (hy i))
         _ = _ := by simp
  have hnR : (0:ℝ) ≤ n := Nat.cast_nonneg n
  rw [abs_div, abs_of_nonneg hnR]
  exact (div_le_one (by exact_mod_cast hn)).mpr hsum

/-- Deterministic induction closes the bounded-state hypotheses of the
previous exploration theorem for every realised sequence satisfying the
model's coefficient constraints. It requires no independence assumption. -/
theorem scores_invariant (q : ℕ → ℝ) (R α v w r : ℕ → ℝ)
    (amax b ε : ℝ) (hq0 : 0 ≤ q 0 ∧ q 0 ≤ 1)
    (hR : ∀ t, 0 ≤ R t ∧ R t ≤ 1)
    (hα : ∀ t, 0 ≤ α t ∧ α t ≤ amax) (hb : 0 ≤ b)
    (hv : ∀ t, 0 ≤ v t ∧ v t ≤ w t) (hw : ∀ t, w t ≤ 1)
    (hr : ∀ t, 0 ≤ r t ∧ r t ≤ w t) (hε : 0 < ε)
    (hbudget : amax+b ≤ 1)
    (hstep : ∀ t, q (t+1) = realisedUpdate (q t) (R t) (α t) b (v t) (w t) (r t) ε) :
    ∀ t, 0 ≤ q t ∧ q t ≤ 1 := by
  intro t
  induction t with
  | zero => exact hq0
  | succ t ih =>
    rw [hstep t]
    exact realised_update_bounds _ _ _ _ _ _ _ _ _ ih (hR t)
      (hα t).1 (hα t).2 hb (hv t).1 (hv t).2 (hw t) (hr t).1 (hr t).2 hε hbudget

theorem act_scale {C : Type*} [Fintype C] (B : C → C → ℝ)
    (ψ : C → ℝ) (scale : ℝ) (c : C) :
    act B (fun d => scale*ψ d) c = scale*act B ψ c := by
  simp only [act, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro d _
  ring

theorem controlled_field_rescaling {C : Type*} [Fintype C]
    (B : C → C → ℝ) (ψ : C → ℝ) (scale lam : ℝ) (c : C) :
    lam*act B (fun d => scale*ψ d) c = (scale*lam)*act B ψ c := by
  rw [act_scale]
  ring

theorem emitted_signal_rescaling (n : ℕ) (sign : Fin n → ℝ) (score scale : ℝ) :
    (∑ i, (scale*score)*sign i)/(n:ℝ) = scale*((∑ i, score*sign i)/(n:ℝ)) := by
  simp_rw [mul_assoc]
  rw [← Finset.mul_sum]
  ring


noncomputable def twoField (Γ x y : ℝ) : ℝ × ℝ :=
  ((1-Γ)*x+Γ*y, Γ*x+(1-Γ)*y)

theorem two_community_common (Γ x : ℝ) : twoField Γ x x = (x,x) := by
  ext <;> simp only [twoField] <;> ring

theorem two_community_relative (Γ x : ℝ) :
    twoField Γ x (-x) = ((1-2*Γ)*x, -(1-2*Γ)*x) := by
  ext <;> simp only [twoField] <;> ring

theorem two_community_difference (Γ x y : ℝ) :
    (twoField Γ x y).1-(twoField Γ x y).2 = (1-2*Γ)*(x-y) := by
  simp only [twoField]
  ring

theorem default_frozen_multipliers :
    (1/2:ℝ)*12/2*(3/5) = 9/5 ∧
    (1/2:ℝ)*12/2*(3/5)*(1-2*(3/20)) = 63/50 := by norm_num

/-- The default relative multiplier exceeds one: attenuation relative to the
common mode must not be described as contraction of the adaptive process. -/
theorem default_relative_amplifies : (1:ℝ) < (1/2:ℝ)*12/2*(3/5)*(1-2*(3/20)) := by
  norm_num

theorem sigmoid_strictMono : StrictMono sigmoid := by
  intro x y hxy
  unfold sigmoid
  apply one_div_lt_one_div_of_lt (by positivity)
  have he : Real.exp (-y) < Real.exp (-x) := Real.exp_lt_exp.mpr (by linarith)
  linarith

theorem withdrawal_probability (h g lam H : ℝ) (hh : 0 ≤ h) (hfield : lam*H ≤ 0) :
    sigmoid (h*(g+lam*H)) ≤ sigmoid (h*g) := by
  apply sigmoid_monotone
  nlinarith

theorem withdrawal_probability_strict (h g lam H : ℝ) (hh : 0 < h)
    (hfield : lam*H < 0) :
    sigmoid (h*(g+lam*H)) < sigmoid (h*g) := by
  apply sigmoid_strictMono
  nlinarith

theorem average_subset_bound {I : Type*} [Fintype I] [DecidableEq I]
    (S : Finset I) (p : I → ℝ) (u : ℝ)
    (hpS : ∀ i ∈ S, p i ≤ u) (hp : ∀ i, p i ≤ 1) :
    (∑ i, p i) ≤ (S.card : ℝ)*u + ((Fintype.card I : ℝ)-(S.card : ℝ)) := by
  classical
  have hh : ∀ i : I, p i ≤ if i ∈ S then u else 1 := by
    intro i
    split_ifs with hi
    · exact hpS i hi
    · exact hp i
  calc _ ≤ ∑ i : I, (if i ∈ S then u else 1) := Finset.sum_le_sum (fun i _ => hh i)
       _ = _ := by
         rw [← Finset.sum_add_sum_compl S]
         rw [Finset.sum_ite_of_true (fun i hi => hi),
           Finset.sum_ite_of_false (fun i hi => Finset.mem_compl.mp hi)]
         simp [Finset.card_compl, Nat.cast_sub (Finset.card_le_univ S)]

/-- Finite-population version of the expression bound, with the exact logistic
probabilities and the exact fraction of agents in the specified subset. -/
theorem population_expression_bound {I : Type*} [Fintype I] [DecidableEq I]
    (S : Finset I) (g H : I → ℝ) (h lam gbar a : ℝ)
    (hI : 0 < Fintype.card I) (hh : 0 ≤ h)
    (hS : ∀ i ∈ S, g i ≤ gbar ∧ lam*H i ≤ -a) :
    (∑ i, sigmoid (h*(g i+lam*H i))) / (Fintype.card I : ℝ) ≤
      ((S.card : ℝ)/(Fintype.card I : ℝ))*sigmoid (h*(gbar-a)) +
        (1-(S.card : ℝ)/(Fintype.card I : ℝ)) := by
  have hbound := average_subset_bound S
    (fun i => sigmoid (h*(g i+lam*H i))) (sigmoid (h*(gbar-a)))
    (fun i hi => by apply sigmoid_monotone; nlinarith [(hS i hi).1,(hS i hi).2])
    (fun i => (sigmoid_lt_one _).le)
  have hn : (0:ℝ) < Fintype.card I := by exact_mod_cast hI
  have hdiv := div_le_div_of_nonneg_right hbound (le_of_lt hn)
  convert hdiv using 1
  field_simp


/-- Algebraic conditional-variance identity for two Bernoulli reward branches.
Adding a common peer increment to both branches leaves this quantity unchanged. -/
theorem bernoulli_variance_identity (μ x0 x1 : ℝ) :
    μ*(x1-(μ*x1+(1-μ)*x0))^2 +
      (1-μ)*(x0-(μ*x1+(1-μ)*x0))^2 = μ*(1-μ)*(x1-x0)^2 := by ring

theorem reward_branch_separation (q ap am amin : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hap : amin ≤ ap) (ham : amin ≤ am) :
    amin ≤ ap*(1-q)+am*q := by
  nlinarith [mul_nonneg (sub_nonneg.mpr hap) (sub_nonneg.mpr hq1),
    mul_nonneg (sub_nonneg.mpr ham) hq0]

theorem reward_variance_lower_bound (μ q ap am amin base : ℝ)
    (hμ0 : 0 ≤ μ) (hμ1 : μ ≤ 1) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (ha0 : 0 ≤ amin) (hap : amin ≤ ap) (ham : amin ≤ am) :
    μ*(1-μ)*amin^2 ≤ μ*(1-μ)*
      ((base+ap*(1-q))-(base-am*q))^2 := by
  have hsep := reward_branch_separation q ap am amin hq0 hq1 hap ham
  have hsq : amin^2 ≤ ((base+ap*(1-q))-(base-am*q))^2 := by nlinarith
  exact mul_le_mul_of_nonneg_left hsq (mul_nonneg hμ0 (sub_nonneg.mpr hμ1))

end CSLRepair
