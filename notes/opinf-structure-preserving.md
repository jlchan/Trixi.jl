# Structure-Preserving and Dissipative Operator Inference

Notes on energy-preserving OpInf (EP-OpInf) and related formulations that impose dissipative or definiteness structure on inferred operators.

**Viewing tip:** Open this file in a Markdown previewer with math support (e.g. VS Code/Cursor preview, GitHub, or a viewer with KaTeX/MathJax). Inline math uses `$...$`; display math uses `$$...$$`.

---

## 1. What EP-OpInf preserves

**References:**
- Koike & Qian, *Energy-Preserving Reduced Operator Inference for Efficient Design and Control* ([arXiv:2401.02889](https://arxiv.org/abs/2401.02889))
- Kimisis et al., *On the representation of energy-preserving quadratic operators with application to Operator Inference* ([AML 2025](https://doi.org/10.1016/j.aml.2025.109761))

Consider a semi-discrete quadratic system

$$
\dot{x} = A x + H(x \otimes x),
$$

where $x \in \mathbb{R}^n$ and $\otimes$ denotes the Kronecker product.

The **energy-preserving** property enforced by EP-OpInf is

$$
x^\top H(x \otimes x) = 0 \quad \text{for all } x,
$$

or equivalently

$$
\langle x,\, H(x \otimes x) \rangle = 0.
$$

For kinetic energy $E = \tfrac{1}{2}\|x\|^2$, this means the quadratic term does **not** change total energy. It only redistributes energy among degrees of freedom.

### Implementation variants

| Method | How energy preservation is enforced |
|--------|-------------------------------------|
| **EP-OpInf** (Koike–Qian) | Equality constraints in a constrained least-squares / optimization solve |
| **Seq_OpInf_EP** (Kimisis et al.) | Skew-symmetric sub-block parameterization of $\hat{H}$, enforced via sequential closed-form LS solves |

Kimisis et al. prove that every energy-preserving quadratic operator admits an equivalent representation with **skew-symmetric sub-matrices**.

### Important nuance

These papers target systems where:

- **Convection / advection** is energy-preserving and lives in $H$.
- **Dissipation** (viscosity, diffusion, etc.) lives in the **linear operator** $A$.

**Example (viscous Burgers):**

$$
\partial_t u = -\partial_x\left(\tfrac{1}{2}u^2\right) + \nu \partial_{xx} u.
$$

After discretization:

- The quadratic convection term is energy-preserving (in $H$).
- The viscous term $-\nu \partial_{xx}$ sits in $A$.

**EP-OpInf does not impose SPD or negative-definite structure on $A$.**

Kimisis et al. note that unconstrained OpInf can incorrectly attribute dissipation to $\hat{H}$ rather than $\hat{A}$, even when trajectory error looks acceptable. EP-OpInf fixes the physics of the conservative/dissipative split.

---

## 2. OpInf variants with dissipative or definiteness structure

Several lines of work impose **symmetry and (positive/negative) definiteness** on inferred operators. These typically constrain **linear** or **second-order mechanical** terms, not the quadratic $H$.

### 2.1 SPIR-OpInf (symmetric negative definite linear operator)

**Reference:** Sawant, Kramer & Peherstorfer, *Physics-informed regularization and structure preservation for learning stable reduced models from data with operator inference* (CMAME 2023)

SPIR-OpInf solves a constrained optimization problem:

$$
\min_{\hat{A},\,\hat{B},\,\hat{F},\,\hat{G}}
\; J + \lambda\left(\|\hat{F}\|_F^2 + \|\hat{G}\|_F^2\right)
\quad \text{subject to} \quad
\hat{A} + \epsilon I \preceq 0.
$$

So the inferred linear operator is **symmetric negative definite** (dissipative in the usual quadratic-energy sense). The paper notes that other structures (e.g. skew-symmetry) can be imposed analogously.

**Closest generic OpInf answer to “dissipative structure on inferred operators,” though it targets negative definiteness of the linear part, not SPD.**

### 2.2 Lagrangian OpInf (LOpInf)

**Reference:** Sharma & Kramer, *Preserving Lagrangian structure in data-driven reduced-order modeling of large-scale dynamical systems* (2022)

For mechanical systems, LOpInf embeds Lagrangian structure into OpInf. In the extended **LOpInf-SpML** framework, reduced operators are learned with semidefinite constraints:

$$
\min \|\cdots\|_F
\quad \text{subject to} \quad
\hat{K} = \hat{K}^\top \succ 0,
\qquad
\hat{C} = \hat{C}^\top \succ 0,
$$

for reduced **stiffness** $\hat{K}$ and **damping** $\hat{C}$.

### 2.3 Constrained mechanical OpInf (cOpInf)

**Reference:** Rodopoulos et al., *An operator inference oriented approach for linear mechanical systems* (2023, [arXiv:2210.07710](https://arxiv.org/abs/2210.07710))

For second-order mechanical systems, cOpInf enforces on inferred mass, stiffness, and damping matrices:

$$
M \succ 0, \qquad K \succ 0, \qquad E \succeq 0,
$$

using semidefinite programming when additional force data are available. This is an explicit **SPD/PSD-constrained** OpInf formulation.

### 2.4 Entropy-stable / entropy-conserving OpInf (conservation laws)

**Reference:** Sawant, *Learning structured and stable reduced models from data with operator inference* (PhD thesis, NYU)

For nonlinear conservation laws:

| Method | Constraint |
|--------|------------|
| **EC-OpInf** | Entropy conservation on inferred flux operators |
| **ES-OpInf** | Entropy dissipation: $dS/dt \le 0$ via constrained OpInf |

This is dissipative structure in **entropy variables**, not SPD of an elliptic operator directly, but directly relevant to conservation-law ROMs. Constraints use convexity of the entropy ($\partial^2 S / \partial q^2 \succ 0$).

### 2.5 Related but not strictly OpInf

| Method | Role |
|--------|------|
| **Goyal et al. (2023)** | Sparse regression with soft/hard constraints on linear stability and quadratic energy preservation (precursor to Kimisis et al.) |
| **H-OpInf** | Hamiltonian / symplectic structure — **conservative**, not dissipative |
| **PIR-OpInf** | Physics-informed regularization for Lyapunov stability; penalties and Hurwitz post-processing rather than hard SPD constraints |

---

## 3. Mapping to gradient and elliptic operators

| Structure | Typical OpInf enforcement | Relevant work |
|-----------|---------------------------|---------------|
| Energy-preserving quadratic convection | $x^\top H(x \otimes x) = 0$ | EP-OpInf, Seq_OpInf_EP |
| Dissipative linear diffusion / viscosity | $\hat{A} \preceq 0$ or $-\hat{G}^\top \hat{K} \hat{G}$ | SPIR-OpInf |
| SPD stiffness / damping | $\hat{K} \succ 0$, $\hat{C} \succ 0$ | LOpInf, cOpInf |
| Entropy dissipation | $dS/dt \le 0$ on flux operators | ES-OpInf |
| SPD elliptic operator in entropy variables | Not yet a standard standalone OpInf package; natural via $-\hat{G}^\top \hat{K} \hat{G}$ with $\hat{K} \succeq 0$ | Related ROM theory |

---

## 4. Bottom line

1. **EP-OpInf / Kimisis et al.** preserve that the **quadratic operator does not change energy** ($x^\top H(x \otimes x) = 0$).

2. **Dissipative structure is handled elsewhere in OpInf**, mainly by constraining **linear** or **second-order mechanical** operators:
   - **SPIR-OpInf:** symmetric negative definite $\hat{A}$
   - **LOpInf / cOpInf:** SPD $\hat{K}$, PSD/SPD $\hat{C}$
   - **ES-OpInf:** entropy dissipation for conservation laws

3. For a viscous Burgers / KSE-type split, the natural combined picture is:
   - **EP-OpInf on $\hat{H}$** for conservative convection
   - **SPIR-OpInf** or a factored form on $\hat{A}$ or $-\hat{G}^\top \hat{K}\hat{G}$ for dissipation

   rather than expecting EP-OpInf alone to enforce SPD dissipation.

---

## 5. Suggested combined formulation (sketch)

For systems with both conservative quadratic transport and dissipative elliptic regularization in entropy variables $w$:

$$
\dot{a} = \hat{A} a + \hat{H}(a \otimes a) - \hat{G}^\top \hat{K}(a)\, \hat{G}\, a,
$$

one could combine:

1. **Seq_OpInf_EP** (or EP-OpInf) on $\hat{H}$ with $a^\top \hat{H}(a \otimes a) = 0$.
2. **SPIR-OpInf-style SDP** on the dissipative part, e.g. parameterize $-\hat{G}^\top \hat{K}\hat{G}$ with $\hat{K}(a) = B(a)^\top B(a) + \epsilon I \succeq 0$.

This is a research direction rather than an established packaged method.
