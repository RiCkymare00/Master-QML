- [Operator on a system of two Qubits](#operator-on-a-system-of-two-qubits)
  - [Representation of $\sigma_z^{(1)}$ acting on first qubit](#representation-of-sigma_z1-acting-on-first-qubit)
  - [Representation of $\sigma_z^{(2)}$ acting on the second qubit](#representation-of-sigma_z2-acting-on-the-second-qubit)
- [Action of $\sigma_z$ on both Qubits](#action-of-sigma_z-on-both-qbits)
  - [Example: Action of $\sigma_z^{(2)} = I \otimes \sigma_z$ on a 2-qubit system](#example-action-of-sigma_z2--i-otimes-sigma_z-on-a-2-qubit-system)
  - [Action on both qubits simultanesuly](#action-on-both-qubits-simultanesuly)
- [Expectation value for an Operator](#expectation-value-for-an-operator)
  - [Expectation value of $\sigma_z^{(2)}$ for a 1-qubit state](#expectation-value-of--for-a-1-qubit-state)
  - [Expectation value for an Operator](#expectation-value-for-an-operator)
  - [Expectation value of $\sigma_z$ for a 1-qubit state](#expectation-value-of-sigma_z-for-a-1-qubit-state)
  - [Expectation value from eigenvalues and probabilities](#expectation-value-from-eigenvalues-and-probabilities)
  - [Application to $\sigma_z$](#application-to-sigma_z)
  - [Expectation value of $\sigma_x$ via basis change](#expectation-value-of-sigma_x-via-basis-change)
  - [Expectation value of $\sigma_x^{(1)} \sigma_x^{(2)}$ on a 2-qubit system](#example-action-of--on-a-2-qubit-system)



### Operator on a system of two Qubits

When applying an operator such as the Pauli-Z operator $\sigma_z$ to a system of two qubits, its effect depends on which qubit it acts upon.

For example:

- Applying $\sigma_z$ to the **first** qubit in the tensor product state $|\psi\rangle = |q_1\rangle \otimes |q_2\rangle$ is written as:
  
  $$(\sigma_z \otimes I) |q_1\rangle \otimes |q_2\rangle = (\sigma_z |q_1\rangle) \otimes |q_2\rangle$$

- Applying $\sigma_z$ to the **second** qubit is written as:

  $$(I \otimes \sigma_z) |q_1\rangle \otimes |q_2\rangle = |q_1\rangle \otimes (\sigma_z |q_2\rangle)$$

The Pauli-Z operator flips the phase of the $$|1\rangle$$ state:

 $$\sigma_z |0\rangle = |0\rangle$$  
  $$\sigma_z |1\rangle = -|1\rangle$$

This means that if $\sigma_z$ acts on a basis state like $|01\rangle$, the result is:
  
  $$|0\rangle \otimes (\sigma_z |1\rangle) = -|01\rangle$$

We can also understand the action of the Pauli-Z operator on multi-qubit basis states. Recall the following:

 $$\sigma_z |\uparrow\rangle = +1 |\uparrow\rangle$$
 $$\sigma_z |\downarrow\rangle = -1 |\downarrow\rangle$$

This can also be written in computational basis as:

 $$\sigma_z |0\rangle = |0\rangle$$
 $$\sigma_z |1\rangle = -|1\rangle$$

### representation ofof $\sigma_z^{(1)}$ acting on first qubit

Suppose we act with $\sigma_z$ on the **first qubit** of a two-qubit system. Consider the basis states:

$$
\begin{aligned}
\sigma_z^{(1)} |00\rangle &= \sigma_z |0\rangle \otimes |0\rangle = |0\rangle \otimes |0\rangle = |00\rangle \\
\sigma_z^{(1)} |01\rangle &= \sigma_z |0\rangle \otimes |1\rangle = |0\rangle \otimes |1\rangle = |01\rangle \\
\sigma_z^{(1)} |10\rangle &= \sigma_z |1\rangle \otimes |0\rangle = -|1\rangle \otimes |0\rangle = -|10\rangle \\
\sigma_z^{(1)} |11\rangle &= \sigma_z |1\rangle \otimes |1\rangle = -|1\rangle \otimes |1\rangle = -|11\rangle \\
\end{aligned}
$$


### representation of $\sigma_z^{(2)}$ acting on the second qubit

Now, suppose we act with $\sigma_z$ on the **second qubit** of a two-qubit system. 

$$
\begin{aligned}
\sigma_z^{(2)} |00\rangle &= |0\rangle \otimes \sigma_z |0\rangle = |0\rangle \otimes |0\rangle = |00\rangle \\
\sigma_z^{(2)} |01\rangle &= |0\rangle \otimes \sigma_z |1\rangle = |0\rangle \otimes (-|1\rangle) = -|01\rangle \\
\sigma_z^{(2)} |10\rangle &= |1\rangle \otimes \sigma_z |0\rangle = |1\rangle \otimes |0\rangle = |10\rangle \\
\sigma_z^{(2)} |11\rangle &= |1\rangle \otimes \sigma_z |1\rangle = |1\rangle \otimes (-|1\rangle) = -|11\rangle \\
\end{aligned}
$$



## Action of $\sigma_z$ on both Qubits
Now, let us compute the operator $\sigma_z^{(1)} = \sigma_z \otimes I$.

Recall the matrices:

$$
\sigma_z =
\begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix},
\quad
I =
\begin{pmatrix}
1 & 0 \\
0 & 1
\end{pmatrix}
$$

Their tensor product is:

$$
\sigma_z \otimes I =
\begin{pmatrix}
1 \cdot
\begin{pmatrix}
1 & 0 \\
0 & 1
\end{pmatrix} &
0 \cdot
\begin{pmatrix}
1 & 0 \\
0 & 1
\end{pmatrix} \\
0 \cdot
\begin{pmatrix}
1 & 0 \\
0 & 1
\end{pmatrix} &
-1 \cdot
\begin{pmatrix}
1 & 0 \\
0 & 1
\end{pmatrix}
\end{pmatrix}
=
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & -1 & 0 \\
0 & 0 & 0 & -1
\end{pmatrix}
$$


### Example: Action of $\sigma_z^{(2)} = I \otimes \sigma_z$ on a 2-qubit system

Now let us compute the operator that applies $\sigma_z$ to the **second qubit** while leaving the first qubit unchanged. This operator is:

$$
\sigma_z^{(2)} = I \otimes \sigma_z
$$

Recall the matrices:

$$
I =
\begin{pmatrix}
1 & 0 \\
0 & 1
\end{pmatrix},
\quad
\sigma_z =
\begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix}
$$

We compute their tensor product:

$$
I \otimes \sigma_z =
\begin{pmatrix}
1 \cdot
\begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix} &
0 \cdot
\begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix} \\
0 \cdot
\begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix} &
1 \cdot
\begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix}
\end{pmatrix}
=
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & -1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & -1
\end{pmatrix}
$$

This matrix shows that applying $\sigma_z$ to the second qubit flips the sign of the components where the second qubit is in the $|1\rangle$ state, while leaving the first qubit unchanged.


### Action on both qubits simultanesuly

Now let us consider the operator that applies a Z gate to **both qubits** simultaneously. This is equivalent to applying:

$$
\sigma_z^{(1)} \sigma_z^{(2)} = \sigma_z \otimes \sigma_z
$$

We compute the tensor product between the two $\sigma_z$ matrices:

$$
\sigma_z \otimes \sigma_z =
\begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix}
\otimes
\begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix}
$$

This gives:

$$
\sigma_z \otimes \sigma_z =
\begin{pmatrix}
1 \cdot
\begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix} &
0 \cdot
\begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix} \\
0 \cdot
\begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix} &
-1 \cdot
\begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix}
\end{pmatrix}
=
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & -1 & 0 & 0 \\
0 & 0 & -1 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

This final $4 \times 4$ matrix represents the operator that applies the Pauli-Z gate to both qubits. As a result:
- States with an **odd number of 1s** (i.e., $|01\rangle$, $|10\rangle$) get a **negative sign**,
- States with **zero or two 1s** (i.e., $|00\rangle$, $|11\rangle$) are **unchanged**.

## Expectation value for an Operator

### Expectation value of $\sigma_z$ for a 1-qubit state

Let us consider a general 1-qubit quantum state:

$$
|\psi\rangle = a|0\rangle + b|1\rangle
$$

We want to compute the expectation value of the Pauli-Z operator $\sigma_z$ in this state:

$$
\langle \sigma_z \rangle = \langle \psi | \sigma_z | \psi \rangle
$$

Expanding both bra and ket expressions:

$$
\langle \psi | = a^* \langle 0| + b^* \langle 1|,\quad
|\psi\rangle = a |0\rangle + b |1\rangle
$$

And applying the operator $\sigma_z$:

$$
\sigma_z |\psi\rangle = \sigma_z (a|0\rangle + b|1\rangle) = a |0\rangle - b |1\rangle
$$

Then the full expression becomes:

$$
\langle \psi | \sigma_z | \psi \rangle =
(a^* \langle 0| + b^* \langle 1|)(a |0\rangle - b |1\rangle)
$$

Expanding:

$$
= a^* a \langle 0|0\rangle - a^* b \langle 0|1\rangle + b^* a \langle 1|0\rangle - b^* b \langle 1|1\rangle
$$

Using orthonormality: $\langle 0|0\rangle = 1$, $\langle 1|1\rangle = 1$, $\langle 0|1\rangle = \langle 1|0\rangle = 0$

$$
= |a|^2 - |b|^2
$$

So, the expectation value of $\sigma_z$ in the state $|\psi\rangle$ is:

$$
\langle \sigma_z \rangle  = |a|^2 - |b|^2
$$

---

### Expectation value from eigenvalues and probabilities

In general, the expectation value of an observable $\hat{O}$ in a state $|\psi\rangle$ is given by:

$$
\langle \hat{O} \rangle = \langle \psi | \hat{O} | \psi \rangle = \sum_n \lambda_n P(\lambda_n)
$$

Where:
- $\lambda_n$ are the eigenvalues of the operator $\hat{O}$
- $P(\lambda_n)$ is the probability of obtaining the outcome $\lambda_n$ when measuring $\hat{O}$ in the state $|\psi\rangle$
- $P(\lambda_n) = |\langle \lambda_n | \psi \rangle|^2$

---

### Application to $\sigma_z$

For the Pauli-Z operator, the eigenstates and eigenvalues are:

$$
\sigma_z |0\rangle = +1 |0\rangle,\quad \sigma_z |1\rangle = -1 |1\rangle
$$

Given the state:

$$
|\psi\rangle = a|0\rangle + b|1\rangle
$$

We compute the measurement probabilities:

$$
|\langle 0 | \psi \rangle|^2 = |a|^2,\quad |\langle 1 | \psi \rangle|^2 = |b|^2
$$

Then the expectation value becomes:

$$
\langle \sigma_z \rangle = (+1) \cdot |a|^2 + (-1) \cdot |b|^2 = |a|^2 - |b|^2
$$

This result is consistent with the one obtained using the direct matrix operation, confirming that expectation values can be understood as weighted averages of the eigenvalues.

---

### Expectation value of $\sigma_x$ via basis change

We now compute the expectation value of the Pauli-X operator $\sigma_x$ by changing the basis using a unitary operator $U$.

Let $|\psi\rangle = a|0\rangle + b|1\rangle$, and consider a change of basis defined by a unitary operator:
$$
U = \begin{pmatrix}
\cos \frac{\theta}{2} & e^{i\beta} \sin \frac{\theta}{2} \\
- e^{-i\beta} \sin \frac{\theta}{2} & \cos \frac{\theta}{2}
\end{pmatrix}
$$

We define a unitary rotation:
$$
\theta = \frac{\pi}{2},\quad \phi = 0
$$

This gives the unitary matrix:
$$
U = \begin{pmatrix}
\cos \frac{\pi}{4} & \sin \frac{\pi}{4} \\
\sin \frac{\pi}{4} & -\cos \frac{\pi}{4}
\end{pmatrix}
=
\frac{1}{\sqrt{2}} \begin{pmatrix}
1 & 1 \\
1 & -1
\end{pmatrix}
$$

This is equivalent to the Hadamard gate.

We define the transformed state as:
$$
|\psi'\rangle = U |\psi\rangle
$$

Then, the expectation value of $\sigma_x$ in the original state $|\psi\rangle$ can be written in the rotated basis as:
$$
\langle \psi | \sigma_x | \psi \rangle = \langle \psi' | U \sigma_x U^\dagger | \psi' \rangle
$$


however $U = U^\dagger$

Therefore: 
$$
\sigma_z = U \sigma_x U^\
$$

Verification: $\sigma_z = U \sigma_x U^\dagger$ with the Hadamard gate

Let us verify the identity $\sigma_z = U \sigma_x U^\dagger$ using matrix multiplication, where $U$ is the Hadamard gate:
$$
U = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}, \quad U^\dagger = U
$$

Recall the Pauli matrices:
$$
\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$

**Step 1**: Compute $\sigma_x U$
$$
\sigma_x U = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \cdot \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
= \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}
$$

**Step 2**: Compute $U (\sigma_x U)$

$$
U \cdot (\sigma_x U) = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \cdot \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}
= \frac{1}{2} \begin{pmatrix}
2 & 0 \\
0 & -2
\end{pmatrix}
= \begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix} = \sigma_z
$$

✅ This confirms: $\sigma_z = U \sigma_x U^\dagger$




This allows us to evaluate the expectation value using the transformed state $|\psi'\rangle$ and rotated operator $\sigma'_x$ = $\sigma_z<$>


This method is useful when $|\psi\rangle$ is better expressed or easier to interpret in the rotated basis defined by $U$.

---


### Expectation value of $\sigma_x^{(1)} \sigma_x^{(2)}$ on a 2-qubit system

We now consider a 2-qubit system and compute the expectation value of the joint operator $\sigma_x^{(1)} \sigma_x^{(2)}$, where the Pauli-X operator acts on both qubits.

Let us define the unitary transformation $U$ as the tensor product of two Hadamard gates, one for each qubit:
$$
U = U_1 \otimes U_2 = H_1 \otimes H_2
$$

Since the Hadamard gate is:
$$
H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
$$

Then the tensor product becomes:
$$
U = H \otimes H = \frac{1}{2}
\begin{pmatrix}
1 & 1 & 1 & 1 \\
1 & -1 & 1 & -1 \\
1 & 1 & -1 & -1 \\
1 & -1 & -1 & 1
\end{pmatrix}
$$

This unitary rotates the basis so that we can express the expectation value in terms of a simpler operator (such as $\sigma_z^{(1)} \sigma_z^{(2)}$) in the rotated basis.

Let $|\psi'\rangle = U |\psi\rangle$, then:
$$
\langle \psi | \sigma_x^{(1)} \sigma_x^{(2)} | \psi \rangle = \langle \psi' | U \sigma_x^{(1)} \sigma_x^{(2)} U^\dagger | \psi' \rangle
$$

If we choose the computational basis state $|\psi\rangle = |00\rangle$, then $|\psi'\rangle = U |00\rangle = \frac{1}{2} (|00\rangle + |01\rangle + |10\rangle + |11\rangle)$.

This is an example of how we can simplify the evaluation of expectation values using tensor products of Hadamard gates on multi-qubit systems.

Since $\sigma_x = H \sigma_z H$, we can write:

$$
\sigma_x^{(1)} \sigma_x^{(2)} = (H \otimes H) \cdot \sigma_z^{(1)} \sigma_z^{(2)} \cdot (H \otimes H)
$$


