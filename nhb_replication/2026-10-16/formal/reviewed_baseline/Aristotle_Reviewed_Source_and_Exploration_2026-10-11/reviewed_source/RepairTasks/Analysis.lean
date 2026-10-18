import RepairCore
import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.Analysis.Calculus.Deriv.Inv

/-! Analytic repair lemmas, checked locally under the pinned Lean/mathlib.
    Aristotle is asked to independently verify these exact declarations. -/

namespace CSLRepair

open scoped Topology BigOperators
open Filter

noncomputable def sigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

noncomputable def weightedReadout (score : ℝ → ℝ) (h : ℝ) (x : ℝ) : ℝ :=
  score (sigmoid (h * x)) * (2 * sigmoid (h * x) - 1)

/-- A1: Continuity of the score is enough. Do not add differentiability of
    score: that would exclude the intended preference map at p=1/2. -/
theorem readout_derivative (score : ℝ → ℝ) (h : ℝ)
    (hs : ContinuousAt score (1/2)) :
    HasDerivAt (weightedReadout score h) (score (1/2) * h / 2) 0 := by
  have hh : HasDerivAt (fun x : ℝ => h*x) h 0 := by
    simpa using (hasDerivAt_id (0 : ℝ)).const_mul h
  have he := (Real.hasDerivAt_exp (-(h*(0:ℝ)))).comp 0 hh.neg
  have hd := (hasDerivAt_const (0:ℝ) (1:ℝ)).add he
  have hsigma : HasDerivAt (fun x : ℝ => sigmoid (h*x)) (h/4) 0 := by
    convert (hasDerivAt_const (0:ℝ) (1:ℝ)).div hd
      (by change (1 + Real.exp (-(h*(0:ℝ)))) ≠ 0; positivity) using 1 <;>
      simp [sigmoid] <;> ring
  have hg : HasDerivAt (fun x : ℝ => 2*sigmoid (h*x)-1) (h/2) 0 := by
    convert (hsigma.const_mul 2).sub_const 1 using 1 <;> ring
  have hcont : ContinuousAt (fun x : ℝ => score (sigmoid (h*x))) 0 := by
    have hz : sigmoid (h*(0:ℝ)) = 1/2 := by norm_num [sigmoid]
    have hss : ContinuousAt score (sigmoid (h*(0:ℝ))) := by rw [hz]; exact hs
    exact ContinuousAt.comp (f := fun x : ℝ => sigmoid (h*x)) (x := 0)
      hss hsigma.continuousAt
  apply hasDerivAt_iff_tendsto_slope.mpr
  have ht := (hcont.tendsto.mono_left nhdsWithin_le_nhds).mul hg.tendsto_slope
  convert ht using 1
  · ext x
    simp [weightedReadout, slope_def_field, sigmoid]
    ring
  · norm_num [sigmoid]
    ring

/-- A2: Zero score floor eliminates the finite linear threshold. -/
theorem zero_floor_derivative (score : ℝ → ℝ) (h : ℝ)
    (hs : ContinuousAt score (1/2)) (hzero : score (1/2) = 0) :
    HasDerivAt (weightedReadout score h) 0 0 := by
  have hd := readout_derivative score h hs
  rw [hzero, zero_mul, zero_div] at hd
  exact hd

/-- A3: Pairwise row-distance bound, usable for directed networks. The maximum
    over c,d yields the Dobrushin oscillation contraction bound. -/
theorem pairwise_oscillation_bound {C : Type*} [Fintype C] [DecidableEq C]
    (B : C → C → ℝ) (hB : RowStochastic B) (z : C → ℝ)
    (L U : ℝ) (hLU : L ≤ U) (hz : ∀ j, L ≤ z j ∧ z j ≤ U) (c d : C) :
    |act B z c - act B z d| ≤
      ((∑ j, |B c j - B d j|) / 2) * (U - L) := by
  have hmid : ∀ j, |z j - (U+L)/2| ≤ (U-L)/2 := by
    intro j
    rw [abs_le]
    constructor <;> linarith [(hz j).1, (hz j).2]
  calc
    |act B z c - act B z d| = |∑ j, (B c j-B d j)*(z j-(U+L)/2)| := by
      congr 1
      simp only [act, sub_mul, mul_sub, Finset.sum_sub_distrib,
        ← Finset.sum_mul, hB.2 c, hB.2 d]
      ring
    _ ≤ ∑ j, |(B c j-B d j)*(z j-(U+L)/2)| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ j, |B c j-B d j| * |z j-(U+L)/2| := by simp only [abs_mul]
    _ ≤ ∑ j, |B c j-B d j| * ((U-L)/2) := by
      apply Finset.sum_le_sum
      intro j hj
      exact mul_le_mul_of_nonneg_left (hmid j) (abs_nonneg _)
    _ = ((∑ j, |B c j-B d j|)/2)*(U-L) := by
      rw [← Finset.sum_mul]
      ring

/-- A4: Stronger finite self-exclusion ratio bound. Here u is self weight,
    w is full selected-arm mass, and c is full selected-arm score mass. -/
theorem self_exclusion_ratio_bound (u w c d ε : ℝ)
    (hu : 0 ≤ u) (huw : u ≤ w) (hd0 : 0 ≤ d) (hd1 : d ≤ 1)
    (hc0 : u*d ≤ c) (hc1 : c ≤ (w-u) + u*d) (hε : 0 < ε) :
    |c / (w+ε) - (c-u*d) / (w-u+ε)| ≤ u / (w+ε) := by
  have hw : 0 ≤ w := le_trans hu huw
  have hden : 0 < w+ε := by positivity
  have hden2 : 0 < w-u+ε := by linarith
  have he0 : 0 ≤ (c-u*d)/(w-u+ε) := by
    exact div_nonneg (sub_nonneg.mpr hc0) (le_of_lt hden2)
  have he1 : (c-u*d)/(w-u+ε) ≤ 1 := by
    apply (div_le_iff₀ hden2).2
    linarith
  have hdiff : |d-(c-u*d)/(w-u+ε)| ≤ 1 := by
    rw [abs_le]
    constructor <;> linarith
  have heq : c/(w+ε)-(c-u*d)/(w-u+ε) =
      u/(w+ε)*(d-(c-u*d)/(w-u+ε)) := by
    field_simp
    <;> ring
  rw [heq, abs_mul, abs_of_nonneg (div_nonneg hu (le_of_lt hden))]
  simpa using mul_le_mul_of_nonneg_left hdiff (div_nonneg hu (le_of_lt hden))

/-- A5: Block approximation error requires a bound on within-block deviations;
    equality of community row totals alone does not supply such a bound. -/
theorem field_deviation_bound {I : Type*} [Fintype I]
    (W V f : I → ℝ) (r : ℝ) (hr : 0 ≤ r) (hf : ∀ i, |f i| ≤ r) :
    |∑ i, (W i - V i) * f i| ≤ r * (∑ i, |W i - V i|) := by
  calc
    |∑ i, (W i-V i)*f i| ≤ ∑ i, |(W i-V i)*f i| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i, |W i-V i| * |f i| := by simp only [abs_mul]
    _ ≤ ∑ i, |W i-V i| * r := by
      apply Finset.sum_le_sum
      intro i hi
      exact mul_le_mul_of_nonneg_left (hf i) (abs_nonneg _)
    _ = r*(∑ i, |W i-V i|) := by rw [← Finset.sum_mul]; ring

/-- A6: A stationary network weight gives the autonomous macro coordinate. -/
theorem stationary_macro_coordinate {C : Type*} [Fintype C] [DecidableEq C]
    (B : C → C → ℝ) (π z : C → ℝ)
    (hπ : ∀ d, ∑ c, π c * B c d = π d) :
    (∑ c, π c * act B z c) = ∑ d, π d * z d := by
  simp only [act]
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro d hd
  simp_rw [← mul_assoc]
  rw [← Finset.sum_mul, hπ d]

end CSLRepair
