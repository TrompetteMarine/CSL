import RepairSupplement.Finite

namespace CSLRepair

noncomputable def peerIncrement (b v w r q ε : ℝ) : ℝ :=
  b*v*(r-w*q)/(w+ε)

theorem peer_partial_v (b v w r q ε : ℝ) (_hden : w+ε ≠ 0) :
    HasDerivAt (fun x => peerIncrement b x w r q ε) (b*(r-w*q)/(w+ε)) v := by
  unfold peerIncrement
  convert (((hasDerivAt_id v).const_mul b).mul_const (r-w*q)).div_const (w+ε) using 1; ring

theorem peer_partial_r (b v w r q ε : ℝ) (_hden : w+ε ≠ 0) :
    HasDerivAt (fun x => peerIncrement b v w x q ε) (b*v/(w+ε)) r := by
  unfold peerIncrement
  convert (((hasDerivAt_id r).sub_const (w*q)).const_mul (b*v)).div_const (w+ε) using 1; ring

theorem peer_partial_q (b v w r q ε : ℝ) (_hden : w+ε ≠ 0) :
    HasDerivAt (fun x => peerIncrement b v w r x ε) (-b*v*w/(w+ε)) q := by
  unfold peerIncrement
  convert (((hasDerivAt_const q r).sub ((hasDerivAt_id q).const_mul w)).const_mul (b*v)).div_const (w+ε) using 1; ring

theorem peer_partial_w (b v w r q ε : ℝ) (hden : w+ε ≠ 0) :
    HasDerivAt (fun x => peerIncrement b v x r q ε)
      (-b*v*(q*ε+r)/(w+ε)^2) w := by
  unfold peerIncrement
  convert (((hasDerivAt_const w r).sub ((hasDerivAt_id w).mul_const q)).const_mul (b*v)).div
    ((hasDerivAt_id w).add_const ε) hden using 1; simp only [id_eq, Pi.sub_apply]; ring

/-- All four actual coordinate derivatives of the regularised peer increment
are bounded by b on the admissible convex domain. -/
theorem peer_partial_bounds (b v w r q ε : ℝ) (hb : 0 ≤ b)
    (hv0 : 0 ≤ v) (hvw : v ≤ w) (hw1 : w ≤ 1)
    (hr0 : 0 ≤ r) (hrw : r ≤ w) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hε : 0 < ε) :
    |b*(r-w*q)/(w+ε)| ≤ b ∧ |b*v/(w+ε)| ≤ b ∧
    |-b*v*w/(w+ε)| ≤ b ∧ |-b*v*(q*ε+r)/(w+ε)^2| ≤ b := by
  have hw0 : 0 ≤ w := le_trans hv0 hvw
  have hd : 0 < w+ε := by positivity
  have hrq : |r-w*q| ≤ w := by
    rw [abs_le]
    have hwq0 := mul_nonneg hw0 hq0
    have hwq1 := mul_le_mul_of_nonneg_left hq1 hw0
    constructor <;> nlinarith
  have hg := peer_gain_bounds b v w ε hb hv0 hvw hε
  have hqer0 : 0 ≤ q*ε+r := by positivity
  have hqer1 : q*ε+r ≤ w+ε := by nlinarith
  have hvden : v ≤ w+ε := by linarith
  have hprod : v*(q*ε+r) ≤ (w+ε)^2 := by
    nlinarith [mul_le_mul hvden hqer1 hqer0 (le_of_lt hd)]
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [abs_div, abs_mul, abs_of_nonneg hb, abs_of_pos hd]
    apply (div_le_iff₀ hd).mpr
    nlinarith [mul_le_mul_of_nonneg_left hrq hb]
  · rw [abs_of_nonneg hg.1]
    exact hg.2
  · have he : -b*v*w/(w+ε) = -(b*v/(w+ε)*w) := by ring
    rw [he, abs_neg, abs_of_nonneg (mul_nonneg hg.1 hw0)]
    nlinarith [mul_le_mul_of_nonneg_left hw1 hg.1]
  · have he : -b*v*(q*ε+r)/(w+ε)^2 = -(b*v*(q*ε+r)/(w+ε)^2) := by ring
    rw [he, abs_neg, abs_of_nonneg (by positivity : 0 ≤ b*v*(q*ε+r)/(w+ε)^2)]
    apply (div_le_iff₀ (sq_pos_of_pos hd)).mpr
    nlinarith [mul_le_mul_of_nonneg_left hprod hb]

/-- Exact self exclusion changes the complete peer increment by at most 2bu,
and therefore by the more conservative 3bu used in the coupling estimate.
The score and reward remaining after exclusion are constrained explicitly. -/
theorem peer_self_exclusion_bound (b u v w r d R q ε : ℝ)
    (hb : 0 ≤ b) (hu : 0 ≤ u) (huw : u ≤ w)
    (hd0 : 0 ≤ d) (hd1 : d ≤ 1) (hR0 : 0 ≤ R) (hR1 : R ≤ 1)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hv0 : u*d ≤ v) (hv1 : v ≤ w-u+u*d)
    (hr0 : u*R ≤ r) (hr1 : r ≤ w-u+u*R) (hε : 0 < ε) :
    |peerIncrement b v w r q ε -
      peerIncrement b (v-u*d) (w-u) (r-u*R) q ε| ≤ 2*b*u := by
  have hw0 : 0 ≤ w := le_trans hu huw
  have hden : 0 < w+ε := by positivity
  have hden' : 0 < w-u+ε := by linarith
  have hratio := self_exclusion_ratio_bound u w v d ε hu huw hd0 hd1 hv0 hv1 hε
  have hratio0 : 0 ≤ (v-u*d)/(w-u+ε) :=
    div_nonneg (sub_nonneg.mpr hv0) (le_of_lt hden')
  have hratio1 : (v-u*d)/(w-u+ε) ≤ 1 := by
    apply (div_le_iff₀ hden').mpr
    linarith
  have hrw0 : 0 ≤ r := le_trans (mul_nonneg hu hR0) hr0
  have hrw1 : r ≤ w := by nlinarith [mul_nonneg hu (sub_nonneg.mpr hR1)]
  have hrq : |r-w*q| ≤ w := by
    rw [abs_le]
    constructor <;> nlinarith [mul_nonneg hw0 hq0,
      mul_le_mul_of_nonneg_left hq1 hw0]
  have hRq : |R-q| ≤ 1 := by
    rw [abs_le]
    constructor <;> linarith
  have he : peerIncrement b v w r q ε -
      peerIncrement b (v-u*d) (w-u) (r-u*R) q ε =
      b*(v/(w+ε)-(v-u*d)/(w-u+ε))*(r-w*q) +
        b*((v-u*d)/(w-u+ε))*u*(R-q) := by
    unfold peerIncrement
    ring
  rw [he]
  calc _ ≤ |b*(v/(w+ε)-(v-u*d)/(w-u+ε))*(r-w*q)| +
      |b*((v-u*d)/(w-u+ε))*u*(R-q)| := abs_add_le _ _
       _ = b * |v/(w+ε)-(v-u*d)/(w-u+ε)| * |r-w*q| +
         b*((v-u*d)/(w-u+ε))*u * |R-q| := by
           simp only [abs_mul, abs_of_nonneg hb, abs_of_nonneg hu,
             abs_of_nonneg hratio0]
       _ ≤ b*(u/(w+ε))*w + b*1*u*1 := by
         gcongr
       _ ≤ 2*b*u := by
         have hfrac : u/(w+ε)*w ≤ u := by
           rw [div_mul_eq_mul_div]
           apply (div_le_iff₀ hden).mpr
           nlinarith [mul_nonneg hu (le_of_lt hε)]
         nlinarith [mul_le_mul_of_nonneg_left hfrac hb]

theorem peer_self_exclusion_bound_three (b u v w r d R q ε : ℝ)
    (hb : 0 ≤ b) (hu : 0 ≤ u) (huw : u ≤ w)
    (hd0 : 0 ≤ d) (hd1 : d ≤ 1) (hR0 : 0 ≤ R) (hR1 : R ≤ 1)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hv0 : u*d ≤ v) (hv1 : v ≤ w-u+u*d)
    (hr0 : u*R ≤ r) (hr1 : r ≤ w-u+u*R) (hε : 0 < ε) :
    |peerIncrement b v w r q ε -
      peerIncrement b (v-u*d) (w-u) (r-u*R) q ε| ≤ 3*b*u := by
  have h := peer_self_exclusion_bound b u v w r d R q ε hb hu huw
    hd0 hd1 hR0 hR1 hq0 hq1 hv0 hv1 hr0 hr1 hε
  nlinarith [mul_nonneg hb hu]

end CSLRepair
