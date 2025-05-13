# Table of Contents

- [Table of Contents](#table-of-contents)
- [Summary: From Classical Mechanics to the Schrödinger Equation](#summary-from-classical-mechanics-to-the-schrödinger-equation)
  - [1. Classical Hamiltonian](#1-classical-hamiltonian)
  - [2. Hamilton's Equations of Motion](#2-hamiltons-equations-of-motion)
  - [3. From the Hamiltonian to the Schrödinger Equation](#3-from-the-hamiltonian-to-the-schrödinger-equation)
  - [4. Probabilistic Interpretation](#4-probabilistic-interpretation)
  - [5. Final Summary](#5-final-summary)
- [2. The Measurement Problem and Stationary States](#2-the-measurement-problem-and-stationary-states)
    - [Measurement and Wavefunction Collapse](#measurement-and-wavefunction-collapse)
    - [Double-Slit Experiment](#double-slit-experiment)
    - [Heisenberg Uncertainty Principle](#heisenberg-uncertainty-principle)
    - [Expectation Values](#expectation-values)
    - [Variance of an Operator](#variance-of-an-operator)
    - [Solving Schrödinger’s Equation via Separation of Variables](#solving-schrödingers-equation-via-separation-of-variables)
    - [🔁 Back to the original Schrödinger equation](#-back-to-the-original-schrödinger-equation)
    - [⚖️ Equating both sides](#️-equating-both-sides)
    - [🎯 This gives the **time-independent Schrödinger equation**:](#-this-gives-the-time-independent-schrödinger-equation)
    - [Stationary States](#stationary-states)
    - [🎯 Implications](#-implications)

# Summary: From Classical Mechanics to the Schrödinger Equation

## 1. Classical Hamiltonian

$$ 
\textbf{Hamiltonian:} \quad H(q, p) = \frac{p^2}{2m} + V(q)
$$

$$
\text{Where:} \quad T = \frac{p^2}{2m} \quad \text{(kinetic energy)}, \quad V(q) \quad \text{(potential energy)}
$$

---

## 2. Hamilton's Equations of Motion

The equations that describe the temporal evolution of a system are:

$$
\frac{dq}{dt} = \frac{\partial H}{\partial p}, \quad \frac{dp}{dt} = -\frac{\partial H}{\partial q}
$$

$$
\text{The first gives the velocity:} \quad \frac{dq}{dt} = \frac{p}{m}
$$

$$
\text{The second gives the force:} \quad \frac{dp}{dt} = F = -\frac{dV}{dq}
$$

---

## 3. From the Hamiltonian to the Schrödinger Equation

In quantum mechanics, we cannot know position and momentum exactly. Thus, we introduce the **wave function** \( \Psi(x, t) \).

We "quantize" the Hamiltonian by substituting:

$$
\hat{q} = x, \quad \hat{p} = -i \hbar \frac{d}{dx}
$$

Therefore, the quantum Hamiltonian is:

$$
\hat{H} = -\frac{\hbar^2}{2m} \frac{d^2}{dx^2} + V(x)
$$

The time-dependent Schrödinger equation is:

$$
i \hbar \frac{\partial \Psi}{\partial t} = \hat{H} \Psi
$$

$$
i \hbar \frac{\partial \Psi}{\partial t} = -\frac{\hbar^2}{2m} \frac{\partial^2 \Psi}{\partial x^2} + V(x) \Psi
$$

---

## 4. Probabilistic Interpretation

$$
|\Psi(x,t)|^2 = \text{probability density}
$$

Probability of finding the particle between a and b:

$$
\int_a^b |\Psi(x,t)|^2 dx
$$

Normalization:

$$
\int_{-\infty}^{+\infty} |\Psi(x,t)|^2 dx = 1
$$

If 
$$
\Psi(x)
$$

is not normalized:

$$
|N| = \frac{1}{\sqrt{\int |\Psi(x)|^2 dx}}
$$

---

## 5. Final Summary

$$
\begin{array}{|c|c|}
\hline
\textbf{Classical} & \textbf{Quantum} \\
\hline
q \text{ (position)} & x \text{ (multiplication operator)} \\
\hline
p \text{ (momentum)} & -i \hbar \frac{d}{dx} \text{ (derivative)} \\
\hline
H = T + V & \hat{H} = -\frac{\hbar^2}{2m} \frac{d^2}{dx^2} + V(x) \\
\hline
\frac{dq}{dt}, \frac{dp}{dt} & i \hbar \frac{\partial \Psi}{\partial t} = \hat{H} \Psi \\
\hline
\end{array}
$$

# 2. The Measurement Problem and Stationary States

The measurement problem in quantum mechanics arises when we try to understand how the act of measurement affects the state of a quantum system. Before measurement, a system is described by a wave function that represents a superposition of all possible states. Upon measurement, the wave function collapses to a specific state, leading to the question of how and why this collapse occurs.

Stationary states are particular solutions to the Schrödinger equation that do not change in time, except for a phase factor. These states are characterized by having a definite energy and are crucial for understanding the behavior of quantum systems. In stationary states, the probability density remains constant over time, allowing for a stable description of the system's properties.

### Measurement and Wavefunction Collapse

Electrons were measured as particles by Thomson and Millikan. A widely accepted explanation today is that **a measurement collapses the wavefunction**:

$$
\text{If I measure the position, } \Psi(x,t) \rightarrow \delta(x - x_0)
$$

This means: *“Now I know for sure that the particle is at \( x_0 \)”*

The Dirac delta is zero everywhere, except at \( x = x_0 \), where it is infinite, but in such a way that:

$$
\int_{-\infty}^{\infty} \delta(x - x_0) dx = 1
$$

The symbol \( \delta(x - x_0) \) is called the **Dirac delta function**. It is not a regular function, but a "generalized function" or "distribution" used in theoretical physics and mathematics.

Conceptually:
$$
\delta(x - x_0) = 0 \quad \text{for all } x \neq x_0
$$

$$
\int_{-\infty}^{\infty} \delta(x - x_0) dx = f(x_0) \quad \text{for any continuous function } f
$$

That is, it "picks out" the value of a function at \( x_0 \). This reflects the idea that the wavefunction collapses to a precise location after measurement.

### Double-Slit Experiment

Even when electrons are sent one at a time through the slits, interference patterns emerge:

$$
\text{The electron interferes with itself.}
$$

This demonstrates the **wave nature** of particles.

### Heisenberg Uncertainty Principle

$$
\sigma_x \sigma_p \geq \frac{\hbar}{2}
$$

The more precisely one measures position, the less precisely momentum can be known, and vice versa. This arises from the properties of Fourier transforms: a function and its transform cannot both be arbitrarily narrow.

### Expectation Values

$$
\langle x \rangle = \int_{-\infty}^{\infty} x |\Psi(x,t)|^2 dx
$$

Operators for position and momentum:

$$
\hat{x} = x, \quad \hat{p} = -i\hbar \frac{d}{dx}
$$

Thus:

$$
\langle x \rangle = \int \Psi^* x \Psi dx, \quad
\langle p \rangle = \int \Psi^* (-i\hbar \frac{d}{dx}) \Psi dx
$$

For a general operator $\hat{Q}$

$$
\langle Q \rangle = \int \Psi^* \hat{Q} \Psi dx, \quad
\langle Q^2 \rangle = \int \Psi^* \hat{Q}^2 \Psi dx
$$

Variance:

$$
\sigma_Q^2 = \langle Q^2 \rangle - \langle Q \rangle^2
$$

### Variance of an Operator

Given a general operator $\hat{Q}$, its expectation values are defined as:

$$
\langle Q \rangle = \int \Psi^*(x) \, \hat{Q} \, \Psi(x) \, dx, \quad
\langle Q^2 \rangle = \int \Psi^*(x) \, \hat{Q}^2 \, \Psi(x) \, dx
$$

The **variance** $\sigma_Q^2$ quantifies the uncertainty of measuring the observable \( Q \) in the quantum state $\Psi$, and is defined as:

$$
\sigma_Q^2 = \langle (\hat{Q} - \langle Q \rangle)^2 \rangle
$$

Expanding the square:

$$
\sigma_Q^2 = \langle \hat{Q}^2 - 2\langle Q \rangle \hat{Q} + \langle Q \rangle^2 \rangle
$$

Since 
$\langle \hat{Q} \rangle = \langle Q \rangle$ is a scalar, we can take it outside the integral:

$$
\sigma_Q^2 = \langle \hat{Q}^2 \rangle - 2\langle Q \rangle^2 + \langle Q \rangle^2 = \langle Q^2 \rangle - \langle Q \rangle^2
$$

Thus, the variance of \( Q \) is:

$$
\boxed{\sigma_Q^2 = \langle Q^2 \rangle - \langle Q \rangle^2}
$$

This expression reflects how much the measurement outcomes of the observable \( Q \) deviate from the mean value $\langle Q \rangle$.

### Solving Schrödinger’s Equation via Separation of Variables

Time-dependent equation:

$$
i\hbar \frac{\partial \Psi}{\partial t} = - \frac{\hbar^2}{2m} \frac{\partial^2 \Psi}{\partial x^2} + V(x) \Psi
$$


Assume a solution of the form:

$$
\Psi(x,t) = \psi(x)\phi(t)
$$

Substitute into the equation:

$$
i\hbar \psi(x) \frac{d\phi}{dt} = -\frac{\hbar^2}{2m} \frac{d^2 \psi}{dx^2} \phi(t) + V(x)\psi(x)\phi(t)
$$

Divide both sides for $\psi(x)\phi(t)$

$$
\frac{i\hbar}{\phi(t)} \frac{d\phi(t)}{dt} = \frac{-\frac{\hbar^2}{2m} \frac{d^2 \psi(x)}{dx^2} + V(x)\psi(x)}{\psi(x)}
$$

compacting ing:

$$
\frac{1}{\phi(t)} \frac{d\phi(t)}{dt} = \frac{1}{\psi(x)} \left( -\frac{\hbar^2}{2m} \frac{d^2 \psi(x)}{dx^2} + V(x)\psi(x) \right)
$$

> 🔗 **Key observation:**  
> The left-hand side depends only on time $t$, and the right-hand side only on position $x$.  
> For this equality to hold for all values of $x$ and $t$, both sides must be equal to a constant E.
> We define this constant as $-\frac{iE}{\hbar}$, where $E$ represents the energy.

$$
\frac{1}{\phi(t)} \frac{d\phi}{dt} = -\frac{i}{\hbar} E, \quad \Rightarrow \phi(t) = e^{-iEt/\hbar}
$$

---
### 🔁 Back to the original Schrödinger equation

We now return to the full time-dependent Schrödinger equation:


Substitute $\Psi(x,t) = \psi(x) e^{-iEt/\hbar}$

**Left-hand side:**

$$
\frac{\partial \Psi}{\partial t} = \psi(x) \cdot \frac{d}{dt} \left( e^{-iEt/\hbar} \right) = \psi(x) \cdot \left( -\frac{iE}{\hbar} \right) e^{-iEt/\hbar}
$$

So:

$$
i\hbar \frac{\partial \Psi}{\partial t} = E \psi(x) e^{-iEt/\hbar}
$$

**Right-hand side:**

$$
\hat{H} \Psi = \left( -\frac{\hbar^2}{2m} \frac{d^2 \psi(x)}{dx^2} + V(x)\psi(x) \right) \cdot e^{-iEt/\hbar}
$$

---

### ⚖️ Equating both sides

$$
E \psi(x) e^{-iEt/\hbar} = \left[ -\frac{\hbar^2}{2m} \frac{d^2 \psi(x)}{dx^2} + V(x)\psi(x) \right] e^{-iEt/\hbar}
$$

Now divide both sides by $e^{-iEt/\hbar}$ (which is nonzero):

$$
E \psi(x) = -\frac{\hbar^2}{2m} \frac{d^2 \psi(x)}{dx^2} + V(x)\psi(x)
$$

---

### 🎯 This gives the **time-independent Schrödinger equation**:

$$
\boxed{\hat{H} \psi(x) = E \psi(x)}
$$


### Stationary States

Solving the time-independent equation yields stationary states $\psi_n(x)$ and corresponding energies $E_n$:

$$
\hat{H} \psi_n = E_n \psi_n
$$

Stationary states are orthogonal and form a complete set:

$$
\int \psi_m^*(x) \psi_n(x) dx = \delta_{mn}
$$

Any quantum state can be written as:

$$
\Psi(x,t) = \sum_{n=1}^\infty c_n \psi_n(x) e^{-iE_n t/\hbar}
$$

Because the $\psi_n(x)$ are eigenfunctions of the Hamiltonian and form a complete orthonormal basis of the space of functions in which quantum states live (typically a Hilbert space such as $L^2(\mathbb{R})$).

In other words: every "acceptable" (normalizable) function can be projected onto the $\psi_n$, just as a vector can be decomposed along an orthogonal basis.

At $t = 0$:

$$
\Psi(x,0) = \sum_{n=1}^\infty c_n \psi_n(x), \quad \text{with } c_n = \int \psi_n^*(x) \Psi(x,0) dx
$$

Normalization condition:

$$
\sum_{n=1}^\infty |c_n|^2 = 1
$$

becuase

$$
\int_{-\infty}^{+\infty} |\Psi(x,t)|^2 dx = 1
$$

### 🎯 Implications
1. **Discrete Energy Levels:** Quantum systems exhibit discrete energy levels $E_n$, obtained by solving the time-independent Schrödinger equation (TISE). These quantized values reflect the allowed stationary states of the system.
  
2. **Wavefunction Expansion:** Any arbitrary quantum state $\Psi(x,t)$ can be expressed as a linear combination (superposition) of stationary states $\psi_n(x)$, with complex coefficients $c_n$. This reflects the completeness of the eigenstates of the Hamiltonian.

3. **Time Evolution:** The time dependence of each component is given by a phase factor $e^{-iE_n t/\hbar}$. The modulus squared of the wavefunction, $|\Psi(x,t)|^2$, remains unchanged over time for stationary states, making them essential for understanding equilibrium properties.

4. **Measurement and Probabilities:** The probability of measuring the energy $E_n$ in a quantum system is $|c_n|^2$, where $c_n$ is the projection of the initial state onto the stationary state $\psi_n$. This connects the mathematical formalism to measurable physical outcomes.

5. **Predictive Power:** By knowing the set of stationary states $\{\psi_n(x)\}$ and their corresponding energies $\{E_n\}$, one can fully reconstruct and predict the dynamics of any quantum state $\Psi(x,t)$, including its time evolution and response to measurements.
