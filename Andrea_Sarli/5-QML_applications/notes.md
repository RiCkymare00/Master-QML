## 1. Pure States, Mixed States, and Separable Pure States

### Pure States

A **pure state** is the most fundamental description of a quantum system. It is represented by a vector $|\psi\rangle$ in a Hilbert space. For a single qubit, a pure state can be written as:
$$
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle
$$
where $\alpha, \beta \in \mathbb{C}$ and $|\alpha|^2 + |\beta|^2 = 1$.

Pure states contain complete information about the quantum system and can be visualized as points on the surface of the Bloch sphere (for a single qubit).

### Mixed States

A **mixed state** describes a statistical ensemble of pure states. It is used when the system is in one of several possible pure states, each with a certain probability, but we do not know which one. Mixed states are represented by a **density matrix** $\rho$:
$$
\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|
$$
where $p_i$ are probabilities ($0 \leq p_i \leq 1$, $\sum_i p_i = 1$) and $|\psi_i\rangle$ are pure states.

- If $\rho = |\psi\rangle\langle\psi|$, the state is pure.
- If $\rho$ cannot be written as a projector onto a single vector, the state is mixed.

Mixed states are necessary to describe systems with classical uncertainty or entanglement with an environment.

### Separable Pure States

A **separable pure state** is a pure state of a composite system (e.g., two qubits) that can be written as a tensor product of pure states of the subsystems:
$$
|\psi\rangle_{AB} = |\psi\rangle_A \otimes |\phi\rangle_B
$$

For example, for two qubits:
$$
|\psi\rangle_{AB} = (\alpha|0\rangle + \beta|1\rangle) \otimes (\gamma|0\rangle + \delta|1\rangle)
$$

Separable pure states are not entangled; all correlations between the subsystems are classical. In contrast, an **entangled state** cannot be written as a product of states of the subsystems.

**Summary:**
- Pure states: Complete quantum description, single vector.
- Mixed states: Statistical mixture, described by a density matrix.
- Separable pure states: Product states of subsystems, no entanglement.