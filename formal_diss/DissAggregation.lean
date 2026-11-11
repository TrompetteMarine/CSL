/- KGi-DISS-I039 — aggregation layer of the dissonance-gated learning model.
   Lean 4.28.0, core library only (no mathlib, no lake project). Statements over `Rat`
   (the model's rational parameter space); they hold over any linear ordered field.
   Scope: ALGEBRA AND ORDER only — nothing stochastic, nothing behavioural is certified.

   Kernel (as implemented in the task generator, with phi_minus = 1):
     q' = q + kappa * gain c l * delta                         gain c l = 1 - c (1 - l)
     D  = max (a - r) 0 + phip * max (r - a) 0
     l' = clip01 (l - nuF * max (D - tau) 0 + (if D ≤ tau then nuR else 0))
     a' = clip01 ((1 - rho) a + rho ((1 - chi) r + chi ref))                              -/

/-- Learning gain as a function of legitimacy `l` and coupling `c` (the paper is `c = 1`). -/
def gain (c l : Rat) : Rat := 1 - c * (1 - l)

/-- T6 (freeze): at full coupling a fully delegitimised agent does not move, whatever
    the evidence. -/
theorem T6_freeze (kappa q delta c l : Rat) (hc : c = 1) (hl : l = 0) :
    q + kappa * gain c l * delta = q := by
  subst hc; subst hl
  simp only [gain]
  grind

/-- T7 (gain ratio): the gain is 1 at full legitimacy and 1 - c at zero legitimacy, so
    the coupling is one minus the ratio of the two. -/
theorem T7_gain_ratio (c : Rat) : gain c 1 = 1 ∧ gain c 0 = 1 - c := by
  constructor <;> (simp only [gain]; grind)

/-- T8 (aggregation identity, two types with weights p and 1 - p): the mean update is the
    product of the means plus a covariance term. Mean sufficiency is the vanishing of
    that term. -/
theorem T8_aggregation_identity (p g1 g2 d1 d2 : Rat) :
    p * (g1 * d1) + (1 - p) * (g2 * d2)
      = (p * g1 + (1 - p) * g2) * (p * d1 + (1 - p) * d2)
        + p * (1 - p) * ((g1 - g2) * (d1 - d2)) := by
  grind

theorem sq_nonneg_rat (z : Rat) : 0 ≤ z * z := by
  rcases (Rat.le_total : 0 ≤ z ∨ z ≤ 0) with h | h
  · exact Rat.mul_nonneg h h
  · have h1 : 0 ≤ -z := by grind
    have h2 : 0 ≤ (-z) * (-z) := Rat.mul_nonneg h1 h1
    grind

/-- T9 (dispersion identity): second moment minus squared mean, two types. -/
theorem T9_dispersion (p a b : Rat) :
    p * (a * a) + (1 - p) * (b * b) - (p * a + (1 - p) * b) * (p * a + (1 - p) * b)
      = p * (1 - p) * ((a - b) * (a - b)) := by
  grind

/-- T9b: the dispersion term is non-negative for a genuine mixture. -/
theorem T9b_dispersion_nonneg (p a b : Rat) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    (p * a + (1 - p) * b) * (p * a + (1 - p) * b) ≤ p * (a * a) + (1 - p) * (b * b) := by
  have h1 : (0 : Rat) ≤ 1 - p := by grind
  have h := Rat.mul_nonneg (Rat.mul_nonneg hp0 h1) (sq_nonneg_rat (a - b))
  have hid := T9_dispersion p a b
  grind

/-- T9c (witness against mean sufficiency): for a proper mixture of two different gains
    the inequality is strict, so two populations with the same mean gain differ in their
    second-step aggregate response. -/
theorem T9c_dispersion_strict (p a b : Rat) (hp0 : 0 < p) (hp1 : p < 1) (hab : a ≠ b) :
    (p * a + (1 - p) * b) * (p * a + (1 - p) * b) < p * (a * a) + (1 - p) * (b * b) := by
  have h1 : (0 : Rat) < 1 - p := by grind
  have hsq : 0 < (a - b) * (a - b) := by
    rcases (Rat.le_total : a ≤ b ∨ b ≤ a) with h | h
    · have hlt : 0 < b - a := by grind
      have := Rat.mul_pos hlt hlt
      grind
    · have hlt : 0 < a - b := by grind
      exact Rat.mul_pos hlt hlt
  have h := Rat.mul_pos (Rat.mul_pos hp0 h1) hsq
  have hid := T9_dispersion p a b
  grind

/-- T10 (two evidence steps): the aggregate second-step response w (1 - w), averaged over
    two types, falls short of the representative agent's by exactly the dispersion. -/
theorem T10_two_step (p a b : Rat) :
    p * (a * (1 - a)) + (1 - p) * (b * (1 - b))
      = (p * a + (1 - p) * b) * (1 - (p * a + (1 - p) * b))
        - p * (1 - p) * ((a - b) * (a - b)) := by
  grind

/-- Self-contained power (as in DissCore). -/
def cpow (c : Rat) : Nat → Rat
  | 0 => 1
  | n + 1 => c * cpow c n

theorem cpow_nonneg (x : Rat) (hx : 0 ≤ x) : ∀ n, 0 ≤ cpow x n := by
  intro n
  induction n with
  | zero => show (0 : Rat) ≤ 1; grind
  | succ n ih => exact Rat.mul_nonneg hx ih

theorem cpow_mono (x y : Rat) (hx : 0 ≤ x) (hxy : x ≤ y) : ∀ n, cpow x n ≤ cpow y n := by
  intro n
  induction n with
  | zero => show (1 : Rat) ≤ 1; grind
  | succ n ih =>
      have hy : 0 ≤ y := by grind
      have h1 : x * cpow x n ≤ y * cpow x n :=
        Rat.mul_le_mul_of_nonneg_right hxy (cpow_nonneg x hx n)
      have h2 : y * cpow x n ≤ y * cpow y n := Rat.mul_le_mul_of_nonneg_left ih hy
      show x * cpow x n ≤ y * cpow y n
      grind

/-- T11 (horizon inequality, two types): with residual factors x = 1 - w1, y = 1 - w2 in
    [0, ∞), the aggregate residual after t evidence events, p x^t + (1 - p) y^t, is at
    least the representative agent's, (p x + (1 - p) y)^t, at EVERY horizon t. -/
theorem T11_horizon (p x y : Rat) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hx : 0 ≤ x) (hy : 0 ≤ y) :
    ∀ t, cpow (p * x + (1 - p) * y) t ≤ p * cpow x t + (1 - p) * cpow y t := by
  have h1p : (0 : Rat) ≤ 1 - p := by grind
  have hm : 0 ≤ p * x + (1 - p) * y := by
    have a1 := Rat.mul_nonneg hp0 hx
    have a2 := Rat.mul_nonneg h1p hy
    grind
  intro t
  induction t with
  | zero => simp [cpow]; grind
  | succ n ih =>
      have hstep : (p * x + (1 - p) * y) * cpow (p * x + (1 - p) * y) n
          ≤ (p * x + (1 - p) * y) * (p * cpow x n + (1 - p) * cpow y n) :=
        Rat.mul_le_mul_of_nonneg_left ih hm
      have hcov : 0 ≤ p * (1 - p) * ((x - y) * (cpow x n - cpow y n)) := by
        have hpp := Rat.mul_nonneg hp0 h1p
        rcases (Rat.le_total : x ≤ y ∨ y ≤ x) with h | h
        · have hc := cpow_mono x y hx h n
          have a1 : 0 ≤ y - x := by grind
          have a2 : 0 ≤ cpow y n - cpow x n := by grind
          have a3 := Rat.mul_nonneg hpp (Rat.mul_nonneg a1 a2)
          grind
        · have hc := cpow_mono y x hy h n
          have a1 : 0 ≤ x - y := by grind
          have a2 : 0 ≤ cpow x n - cpow y n := by grind
          exact Rat.mul_nonneg hpp (Rat.mul_nonneg a1 a2)
      have hid := T8_aggregation_identity p x y (cpow x n) (cpow y n)
      show (p * x + (1 - p) * y) * cpow (p * x + (1 - p) * y) n
            ≤ p * (x * cpow x n) + (1 - p) * (y * cpow y n)
      grind

/-- T12 (two-type identification): with a share pi at gain 1 - c and the rest at gain 1,
    the mean ratio psi = 1 - c pi and the variance V = c² pi (1 - pi) determine c and pi. -/
theorem T12_two_type_identification (c pi psi V : Rat)
    (hpsi : psi = 1 - c * pi) (hV : V = c * c * pi * (1 - pi)) :
    c * (1 - psi) = V + (1 - psi) * (1 - psi)
      ∧ pi * (V + (1 - psi) * (1 - psi)) = (1 - psi) * (1 - psi) := by
  subst hpsi; subst hV
  constructor <;> grind

def clip01 (x : Rat) : Rat := min 1 (max 0 x)

theorem clip01_mono (x y : Rat) (h : x ≤ y) : clip01 x ≤ clip01 y := by
  simp only [clip01]; grind

theorem clip01_range (x : Rat) : 0 ≤ clip01 x ∧ clip01 x ≤ 1 := by
  simp only [clip01]; constructor <;> grind

/-- Dissonance with phi_minus = 1 and surplus weight `phip`. -/
def diss (phip a r : Rat) : Rat := max (a - r) 0 + phip * max (r - a) 0

def legStep (nuF nuR tau phip l a r : Rat) : Rat :=
  clip01 (l - nuF * max (diss phip a r - tau) 0 + (if diss phip a r ≤ tau then nuR else 0))

def aspStep (rho chi a r ref : Rat) : Rat :=
  clip01 ((1 - rho) * a + rho * ((1 - chi) * r + chi * ref))

theorem asp_mono (rho chi a a' r ref ref' : Rat) (hr0 : 0 ≤ rho) (hr1 : rho ≤ 1)
    (hc0 : 0 ≤ chi) (ha : a ≤ a') (href : ref ≤ ref') :
    aspStep rho chi a r ref ≤ aspStep rho chi a' r ref' := by
  apply clip01_mono
  have h1 : (0 : Rat) ≤ 1 - rho := by grind
  have e1 : (1 - rho) * a ≤ (1 - rho) * a' := Rat.mul_le_mul_of_nonneg_left ha h1
  have e2 : chi * ref ≤ chi * ref' := Rat.mul_le_mul_of_nonneg_left href hc0
  have e3 : rho * (chi * ref) ≤ rho * (chi * ref') := Rat.mul_le_mul_of_nonneg_left e2 hr0
  grind

/-- Legitimacy step on a loss (r = 0): higher aspiration and lower legitimacy give lower
    legitimacy. -/
theorem leg_mono_loss (nuF nuR tau phip l l' a a' : Rat) (hF : 0 ≤ nuF) (hR : 0 ≤ nuR)
    (ha0 : 0 ≤ a) (ha : a ≤ a') (hl : l' ≤ l) :
    legStep nuF nuR tau phip l' a' 0 ≤ legStep nuF nuR tau phip l a 0 := by
  have ha0' : 0 ≤ a' := by grind
  have d1 : diss phip a 0 = a := by simp only [diss]; grind
  have d2 : diss phip a' 0 = a' := by simp only [diss]; grind
  simp only [legStep, d1, d2]
  apply clip01_mono
  have m : max (a - tau) 0 ≤ max (a' - tau) 0 := by grind
  have e : nuF * max (a - tau) 0 ≤ nuF * max (a' - tau) 0 := Rat.mul_le_mul_of_nonneg_left m hF
  by_cases h1 : a' ≤ tau
  · have h2 : a ≤ tau := by grind
    simp [h1, h2]; grind
  · by_cases h2 : a ≤ tau
    · simp [h1, h2]; grind
    · simp [h1, h2]; grind

/-- Legitimacy step on a win (r = 1) with aspirations in [0,1] and phip ≤ tau: the win
    never generates dissonance above tolerance, so the step does not depend on the
    aspiration. -/
theorem leg_win_eq (nuF nuR tau phip l a : Rat) (hp0 : 0 ≤ phip) (hpt : phip ≤ tau)
    (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    legStep nuF nuR tau phip l a 1 = clip01 (l + nuR) := by
  have hmax1 : max (a - 1) 0 = 0 := by grind
  have hmax2 : max (1 - a) 0 = 1 - a := by grind
  have hle : phip * (1 - a) ≤ phip * 1 := Rat.mul_le_mul_of_nonneg_left (by grind) hp0
  have hd : diss phip a 1 ≤ tau := by simp only [diss, hmax1, hmax2]; grind
  have hz : max (diss phip a 1 - tau) 0 = 0 := by grind
  simp only [legStep, hd, hz, if_true]
  congr 1
  grind

theorem leg_mono_win (nuF nuR tau phip l l' a a' : Rat) (hp0 : 0 ≤ phip) (hpt : phip ≤ tau)
    (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (ha0' : 0 ≤ a') (ha1' : a' ≤ 1) (hl : l' ≤ l) :
    legStep nuF nuR tau phip l' a' 1 ≤ legStep nuF nuR tau phip l a 1 := by
  rw [leg_win_eq nuF nuR tau phip l' a' hp0 hpt ha0' ha1',
      leg_win_eq nuF nuR tau phip l a hp0 hpt ha0 ha1]
  apply clip01_mono; grind

/-- One trial: state (a, l); outcome r; reference ref. Legitimacy uses the pre-update
    aspiration, as in the generator. -/
def step (nuF nuR tau phip rho chi : Rat) (s : Rat × Rat) (r ref : Rat) : Rat × Rat :=
  (aspStep rho chi s.1 r ref, legStep nuF nuR tau phip s.2 s.1 r)

/-- A trial record: (win, ref under the low board, ref under the high board). -/
def refOf (hi : Bool) (x : Bool × Rat × Rat) : Rat := if hi then x.2.2 else x.2.1

def run (nuF nuR tau phip rho chi : Rat) (hi : Bool) : Rat × Rat → List (Bool × Rat × Rat) → Rat × Rat
  | s, [] => s
  | s, x :: tl =>
      run nuF nuR tau phip rho chi hi
        (step nuF nuR tau phip rho chi s (if x.1 then 1 else 0) (refOf hi x)) tl

/-- T13 (reference monotonicity): same outcomes, references ordered trial by trial, initial
    aspirations ordered in [0,1] and legitimacies ordered the other way. Then after any
    number of trials the higher-reference path has the higher aspiration and the lower
    legitimacy. Requires phip ≤ tau. -/
theorem T13_reference_monotone (nuF nuR tau phip rho chi : Rat)
    (hF : 0 ≤ nuF) (hR : 0 ≤ nuR) (hp0 : 0 ≤ phip) (hpt : phip ≤ tau)
    (hr0 : 0 ≤ rho) (hr1 : rho ≤ 1) (hc0 : 0 ≤ chi) :
    ∀ (xs : List (Bool × Rat × Rat)), (∀ x ∈ xs, x.2.1 ≤ x.2.2) →
      ∀ (s s' : Rat × Rat), 0 ≤ s.1 → s.1 ≤ s'.1 → s'.1 ≤ 1 → s'.2 ≤ s.2 →
        (run nuF nuR tau phip rho chi false s xs).1 ≤ (run nuF nuR tau phip rho chi true s' xs).1
        ∧ (run nuF nuR tau phip rho chi true s' xs).2 ≤ (run nuF nuR tau phip rho chi false s xs).2 := by
  intro xs
  induction xs with
  | nil =>
      intro _ s s' _ h1 _ h3
      exact ⟨h1, h3⟩
  | cons x tl ih =>
      intro hall s s' h0 h1 h2 h3
      have hx : x.2.1 ≤ x.2.2 := hall x (by simp)
      have htl : ∀ y ∈ tl, y.2.1 ≤ y.2.2 := fun y hy => hall y (by simp [hy])
      have hs1 : s.1 ≤ 1 := by grind
      have hs0' : 0 ≤ s'.1 := by grind
      simp only [run]
      apply ih htl
      · exact (clip01_range _).1
      · show aspStep rho chi s.1 _ (refOf false x) ≤ aspStep rho chi s'.1 _ (refOf true x)
        apply asp_mono rho chi s.1 s'.1 _ _ _ hr0 hr1 hc0 h1
        simp [refOf]; exact hx
      · exact (clip01_range _).2
      · show legStep nuF nuR tau phip s'.2 s'.1 _ ≤ legStep nuF nuR tau phip s.2 s.1 _
        cases hw : x.1 with
        | true =>
            simp
            exact leg_mono_win nuF nuR tau phip s.2 s'.2 s.1 s'.1 hp0 hpt h0 hs1 hs0' h2 h3
        | false =>
            simp
            exact leg_mono_loss nuF nuR tau phip s.2 s'.2 s.1 s'.1 hF hR h0 h1 h3

/-- T14 (product bound): under a linear rule the mean ratio psi = gain c lH / gain c lL
    cannot fall below 1 - c, so psi alone bounds the coupling from below only. -/
theorem T14_product_bound (c lH lL : Rat) (hc0 : 0 ≤ c) (hc1 : c ≤ 1) (hH : 0 ≤ lH)
    (hL : lL ≤ 1) : (1 - c) * gain c lL ≤ gain c lH := by
  have h1 : (0 : Rat) ≤ 1 - c := by grind
  have h2 : (0 : Rat) ≤ 1 - lL := by grind
  have a1 := Rat.mul_nonneg (Rat.mul_nonneg h1 hc0) h2
  have a2 := Rat.mul_nonneg hc0 hH
  simp only [gain]
  grind

/-- T15 (association, two types): two quantities that are ordered the same way across the
    types have a non-negative covariance. With f = E[T | type] and g = E[F | type] this is
    the sign of the within-arm association between revision and reported fairness. -/
theorem T15_association (p f1 f2 g1 g2 : Rat) (hp0 : 0 ≤ p) (hp1 : p ≤ 1)
    (hf : f1 ≤ f2) (hg : g1 ≤ g2) :
    (p * f1 + (1 - p) * f2) * (p * g1 + (1 - p) * g2) ≤ p * (f1 * g1) + (1 - p) * (f2 * g2) := by
  have h1 : (0 : Rat) ≤ 1 - p := by grind
  have a1 : 0 ≤ f2 - f1 := by grind
  have a2 : 0 ≤ g2 - g1 := by grind
  have h := Rat.mul_nonneg (Rat.mul_nonneg hp0 h1) (Rat.mul_nonneg a1 a2)
  have hid := T8_aggregation_identity p f1 f2 g1 g2
  grind

#print axioms T6_freeze
#print axioms T7_gain_ratio
#print axioms T8_aggregation_identity
#print axioms T9_dispersion
#print axioms T9b_dispersion_nonneg
#print axioms T9c_dispersion_strict
#print axioms T10_two_step
#print axioms T11_horizon
#print axioms T12_two_type_identification
#print axioms T13_reference_monotone
#print axioms T14_product_bound
#print axioms T15_association
