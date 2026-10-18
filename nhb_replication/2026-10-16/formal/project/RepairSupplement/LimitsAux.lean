import Mathlib.Data.Real.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Ring.GeomSum
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Lean.Elab.Tactic.Omega

/-!
# Deterministic infrastructure for the finite-horizon population limits

The recurrence theorem starts from an explicit one-step error inequality. It
does not assert that a stochastic coupling satisfies that inequality. Likewise,
the locality theorem starts from an explicit two-hop dependency property. It
does not establish local weak convergence of a graph sequence or an empirical
law of large numbers. These boundaries are part of the formal statements.
-/

namespace CSLRepair

open scoped BigOperators

/-- Finite-horizon discrete Gronwall bound, including the initial discrepancy.
There is no assumption that `K < 1`, and no uniform-in-time conclusion. -/
theorem finite_horizon_error_bound (E : ℕ → ℝ) (K δ : ℝ)
    (hK : 0 ≤ K) (hstep : ∀ t, E (t+1) ≤ K * E t + δ) (T : ℕ) :
    E T ≤ K^T * E 0 + δ * ∑ j ∈ Finset.range T, K^j := by
  induction T with
  | zero => simp
  | succ T ih =>
    calc
      E (T+1) ≤ K * E T + δ := hstep T
      _ ≤ K * (K^T * E 0 + δ * ∑ j ∈ Finset.range T, K^j) + δ :=
        by linarith [mul_le_mul_of_nonneg_left ih hK]
      _ = K^(T+1) * E 0 + δ * ∑ j ∈ Finset.range (T+1), K^j := by
        rw [geom_sum_succ, pow_succ]
        ring

/-- A nonnegative two-coordinate recursion implies a scalar recursion using
any upper bound on the two column sums of its propagation matrix. -/
theorem coupled_error_step (X Z : ℕ → ℝ) (a b c d K εX εZ : ℝ)
    (hX : ∀ t, 0 ≤ X t) (hZ : ∀ t, 0 ≤ Z t)
    (hcol₁ : a+c ≤ K) (hcol₂ : b+d ≤ K)
    (hstepX : ∀ t, X (t+1) ≤ a*X t+b*Z t+εX)
    (hstepZ : ∀ t, Z (t+1) ≤ c*X t+d*Z t+εZ) (t : ℕ) :
    X (t+1)+Z (t+1) ≤ K*(X t+Z t)+(εX+εZ) := by
  have h₁ := mul_le_mul_of_nonneg_right hcol₁ (hX t)
  have h₂ := mul_le_mul_of_nonneg_right hcol₂ (hZ t)
  linarith [hstepX t, hstepZ t]

/-- Explicit finite-horizon estimate for the two error coordinates. The
constants in the assumed one-step bounds must be supplied by the coupling
argument for the actual process. -/
theorem coupled_finite_horizon_error_bound (X Z : ℕ → ℝ)
    (a b c d K εX εZ : ℝ) (hK : 0 ≤ K)
    (hX : ∀ t, 0 ≤ X t) (hZ : ∀ t, 0 ≤ Z t)
    (hcol₁ : a+c ≤ K) (hcol₂ : b+d ≤ K)
    (hstepX : ∀ t, X (t+1) ≤ a*X t+b*Z t+εX)
    (hstepZ : ∀ t, Z (t+1) ≤ c*X t+d*Z t+εZ) (T : ℕ) :
    X T+Z T ≤ K^T*(X 0+Z 0)+(εX+εZ)*∑ j ∈ Finset.range T, K^j := by
  apply finite_horizon_error_bound (fun t => X t+Z t) K (εX+εZ) hK
  exact coupled_error_step X Z a b c d K εX εZ hX hZ hcol₁ hcol₂ hstepX hstepZ

/-- A stochastic row puts at most unit squared mass on its support. -/
theorem stochastic_row_square_bound {V : Type*} [Fintype V]
    (w : V → ℝ) (hw : ∀ v, 0 ≤ w v) (hsum : ∑ v, w v = 1) :
    ∑ v, (w v)^2 ≤ 1 := by
  have hunit : ∀ v, w v ≤ 1 := by
    intro v
    calc
      w v ≤ ∑ u, w u := Finset.single_le_sum (fun u _ => hw u) (Finset.mem_univ v)
      _ = 1 := hsum
  calc
    ∑ v, (w v)^2 ≤ ∑ v, w v := by
      apply Finset.sum_le_sum
      intro v _
      nlinarith [hw v, hunit v]
    _ = 1 := hsum

/-- Dimension-free squared-weight bound used when all inverse block sizes
are at most `δ`. Here `u` is the inverse-size array; no sampling claim is made. -/
theorem weighted_inverse_size_bound {V : Type*} [Fintype V]
    (w u : V → ℝ) (δ : ℝ) (hw : ∀ v, 0 ≤ w v)
    (hsum : ∑ v, w v = 1) (hu : ∀ v, 0 ≤ u v ∧ u v ≤ δ) :
    ∑ v, (w v)^2 * u v ≤ δ := by
  have hunit : ∀ v, w v ≤ 1 := by
    intro v
    calc
      w v ≤ ∑ u, w u := Finset.single_le_sum (fun u _ => hw u) (Finset.mem_univ v)
      _ = 1 := hsum
  calc
    ∑ v, (w v)^2 * u v ≤ ∑ v, w v * δ := by
      apply Finset.sum_le_sum
      intro v _
      have hsq : (w v)^2 ≤ w v := by nlinarith [hw v, hunit v]
      calc
        (w v)^2 * u v ≤ w v * u v := mul_le_mul_of_nonneg_right hsq (hu v).1
        _ ≤ w v * δ := mul_le_mul_of_nonneg_left (hu v).2 (hw v)
    _ = δ := by rw [← Finset.sum_mul, hsum, one_mul]

section Locality

variable {V S I : Type*}

/-- A path of at most `r` graph edges. Equality steps allow a shorter path to
be padded. No connectedness, finiteness or symmetry assumption is needed. -/
def Within (Adj : V → V → Prop) : ℕ → V → V → Prop
  | 0, u, v => u = v
  | r+1, u, v => ∃ w, (u = w ∨ Adj u w) ∧ Within Adj r w v

theorem within_refl (Adj : V → V → Prop) (r : ℕ) (u : V) : Within Adj r u u := by
  induction r with
  | zero => rfl
  | succ r ih => exact ⟨u, Or.inl rfl, ih⟩

theorem within_comp (Adj : V → V → Prop) {r s : ℕ} {u v w : V}
    (huv : Within Adj r u v) (hvw : Within Adj s v w) :
    Within Adj (r+s) u w := by
  induction r generalizing u with
  | zero =>
    change u = v at huv
    subst u
    simpa using hvw
  | succ r ih =>
    rcases huv with ⟨x, hux, hxv⟩
    rw [Nat.succ_add]
    exact ⟨x, hux, ih hxv⟩

theorem within_mono (Adj : V → V → Prop) {r s : ℕ} {u v : V}
    (hrs : r ≤ s) (h : Within Adj r u v) : Within Adj s u v := by
  have he : r + (s-r) = s := Nat.add_sub_of_le hrs
  rw [← he]
  exact within_comp Adj h (within_refl Adj (s-r) v)

/-- Synchronous evolution with a fresh vertex-indexed innovation array at
each step. The state space may itself be an entire finite community block. -/
def localEvolve (step : (V → S) → (V → I) → V → S)
    (initial : V → S) (innovations : ℕ → V → I) : ℕ → V → S
  | 0 => initial
  | t+1 => step (localEvolve step initial innovations t) (innovations t)

/-- The precise hypothesis required by the finite-cone theorem: an update at
`u` can inspect states and fresh innovations only within two graph hops. -/
def TwoHopLocal (Adj : V → V → Prop)
    (step : (V → S) → (V → I) → V → S) : Prop :=
  ∀ (x y : V → S) (ξ ζ : V → I) (u : V),
    (∀ v, Within Adj 2 u v → x v = y v) →
    (∀ v, Within Adj 2 u v → ξ v = ζ v) →
    step x ξ u = step y ζ u

/-- A local stage may have different input, innovation and output types. -/
def OneHopLocal {A B C : Type*} (Adj : V → V → Prop)
    (stage : (V → A) → (V → B) → V → C) : Prop :=
  ∀ (x y : V → A) (ξ ζ : V → B) (u : V),
    (∀ v, Within Adj 1 u v → x v = y v) →
    (∀ v, Within Adj 1 u v → ξ v = ζ v) →
    stage x ξ u = stage y ζ u

/-- The model's radius-two structure: current emissions depend on one-hop
lagged states, then score updates inspect one-hop current emissions. This
compositional fact explains why a one-hop transition theorem cannot simply
be applied without accounting for the emission phase. -/
theorem one_hop_stages_two_hop {L : Type*} (Adj : V → V → Prop)
    (emit : (V → S) → (V → I) → V → L)
    (update : (V → S) → (V → L) → V → S)
    (hemit : OneHopLocal Adj emit) (hupdate : OneHopLocal Adj update) :
    TwoHopLocal Adj (fun x ξ => update x (emit x ξ)) := by
  intro x y ξ ζ root hx hξ
  apply hupdate
  · intro v hrv
    exact hx v (within_mono Adj (by decide : 1 ≤ 2) hrv)
  · intro v hrv
    apply hemit
    · intro w hvw
      exact hx w (within_comp Adj hrv hvw)
    · intro w hvw
      exact hξ w (within_comp Adj hrv hvw)

/-- Equality of initial states and innovations in the radius-`2T` cone
implies equality of the state at the root after `T` synchronous steps. This
certifies a dependency statement, not convergence in probability or law. -/
theorem two_hop_finite_cone (Adj : V → V → Prop)
    (step : (V → S) → (V → I) → V → S) (hstep : TwoHopLocal Adj step)
    (x y : V → S) (ξ ζ : ℕ → V → I) (T : ℕ) (root : V)
    (hinitial : ∀ v, Within Adj (2*T) root v → x v = y v)
    (hinnov : ∀ t, t < T → ∀ v, Within Adj (2*T) root v → ξ t v = ζ t v) :
    localEvolve step x ξ T root = localEvolve step y ζ T root := by
  induction T generalizing root with
  | zero => exact hinitial root (within_refl Adj 0 root)
  | succ T ih =>
    change step (localEvolve step x ξ T) (ξ T) root =
      step (localEvolve step y ζ T) (ζ T) root
    apply hstep
    · intro v hrv
      apply ih v
      · intro w hvw
        apply hinitial w
        have hc := within_comp Adj hrv hvw
        convert hc using 1
        omega
      · intro t ht w hvw
        apply hinnov t (by omega) w
        have hc := within_comp Adj hrv hvw
        convert hc using 1
        omega
    · intro v hrv
      apply hinnov T (by omega) v
      exact within_mono Adj (by omega) hrv

/-- The entire observed root path through `T`, rather than only its endpoint,
is determined by the same radius-`2T` initial/innovation cone. -/
theorem two_hop_path_finite_cone (Adj : V → V → Prop)
    (step : (V → S) → (V → I) → V → S) (hstep : TwoHopLocal Adj step)
    (x y : V → S) (ξ ζ : ℕ → V → I) (T : ℕ) (root : V)
    (hinitial : ∀ v, Within Adj (2*T) root v → x v = y v)
    (hinnov : ∀ t, t < T → ∀ v, Within Adj (2*T) root v → ξ t v = ζ t v) :
    ∀ t, t ≤ T → localEvolve step x ξ t root = localEvolve step y ζ t root := by
  intro t ht
  apply two_hop_finite_cone Adj step hstep x y ξ ζ t root
  · intro v hv
    exact hinitial v (within_mono Adj (by omega) hv)
  · intro s hs v hv
    exact hinnov s (by omega) v (within_mono Adj (by omega) hv)

end Locality

end CSLRepair
