# The following code is adapted from: https://arnaucasau.github.io/qiskit-machine-learning/tutorials/11_quantum_convolutional_neural_networks.html
# AIM: we wanna prove how introducting a quantum layer in a classical cnn can indeed imporve the feature extraction

'''
Theory:
First, we encode our pixelated image into a quantum circuit using a given feature map, such Qiskit’s ZFeatureMap or ZZFeatureMap or others available in the circuit library.
After encoding our image, we apply alternating convolutional and pooling layers, as defined in the next section. By applying these alternating layers, we reduce the dimensionality of our circuit until we are left with one qubit. We can then classify our input image by measuring the output of this one remaining qubit.
'''

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

class QCNN:
    def __init__(self, n_qubits: int = 8, seed: int = 12345):
        self.n_qubits = n_qubits
        algorithm_globals.random_seed = seed
        self.feature_map = ZZFeatureMap(self.n_qubits)
        self.ansatz = QuantumCircuit(self.n_qubits, name="Anasatz")
        self.circuit = None
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1)])
        self.qnn = None
        self.classifier = None
        self.objective_func_vals = []

    # In a QCNN, one could in principle choose any parametrized circuit 
    # for the convolutional and pooling layers. Here, we base our analysis 
    # on the mathematical result that any 2-qubit unitary U belongs to SU(4).
    #
    # According to Vatan & Williams decomposition:
    #   U = (A1 ⊗ A2) · N(α, β, γ) · (A3 ⊗ A4)
    # where Ai ∈ SU(2) (3 parameters each) and
    #       N(α, β, γ) = exp(i [α σx⊗σx + β σy⊗σy + γ σz⊗σz])
    #       (α, β, γ are 3 real parameters)
    # Total parameters: 4*3 + 3 = 15 (matches SU(4) dimension)
    #
    # N(α, β, γ) is the "non-local" part of the unitary, responsible for entanglement
    # between the two qubits. The Ai gates are local rotations on each qubit.
    #
    # Using all 15 parameters would allow representing any 2-qubit unitary,
    # but training would be inefficient. To simplify, we restrict our ansatz
    # to only N(α, β, γ), i.e., 3 parameters per gate. 
    # This reduces training cost and complexity but limits the accessible
    # Hilbert space, potentially reducing QCNN accuracy.

    @staticmethod
    def _conv_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target
    
    def _conv_layer(self, num_qubits: int, param_prefix: str):
        qc = QuantumCircuit(num_qubits, name = f"Conv_{param_prefix}")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits*3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(self._conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(self._conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()
        qc_full = QuantumCircuit(num_qubits)
        qc_full.append(qc_inst, qubits)
        return qc_full

    @staticmethod
    def _pool_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def _pool_layer(self, sources, sinks, param_prefix: str):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name=f"Pool_{param_prefix}")
        param_index = 0
        params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(self._pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()
        qc_full = QuantumCircuit(num_qubits)
        qc_full.append(qc_inst, range(num_qubits))
        return qc_full
    
    def build_ansatz_8qubits(self):
        if self.n_qubits != 8:
            raise ValueError("build_ansatz_8qubits è implementato solo per 8 qubit.")
        self.ansatz.compose(self._conv_layer(8, "c1"), list(range(8)), inplace=True)
        self.ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        self.ansatz.compose(self._conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        self.ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
        self.ansatz.compose(self._conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        self.ansatz.compose(self._pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    def build_circuit(self, decompose_for_qnn: bool = True):
        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit.compose(self.feature_map, range(self.n_qubits), inplace=True)
        self.circuit.compose(self.ansatz, range(self.n_qubits), inplace=True)
        circ_for_qnn = self.circuit.decompose() if decompose_for_qnn else self.circuit
        self.qnn = EstimatorQNN(
            circuit=circ_for_qnn,
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
        )
        return self.qnn

    def _callback(self, weights, objective_eval):
        clear_output(wait=True)
        self.objective_func_vals.append(objective_eval)
        plt.title("Objective function value vs iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.plot(range(len(self.objective_func_vals)), self.objective_func_vals)
        plt.show()

    def train(self, x_train, y_train, optimizer=None, maxiter=200):
        if self.qnn is None:
            raise RuntimeError("QNN non costruito. Chiamare prima build_circuit().")

        if optimizer is None:
            optimizer = COBYLA(maxiter=maxiter)

        self.objective_func_vals = []

        self.classifier = NeuralNetworkClassifier(
            self.qnn,
            optimizer=optimizer,
            callback=self._callback,
        )

        self.classifier.fit(np.asarray(x_train), np.asarray(y_train))
        return self.classifier

    def predict(self, x):
        if self.classifier is None:
            raise RuntimeError("Classifier non addestrato. Chiamare train() prima.")
        return self.classifier.predict(np.asarray(x))

    def score(self, x, y):
        if self.classifier is None:
            raise RuntimeError("Classifier non addestrato. Chiamare train() prima.")
        return self.classifier.score(np.asarray(x), np.asarray(y))

    def plot_sample_predictions(self, test_images, y_true, y_pred, n_samples: int = 4):
        fig, ax = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={"xticks": [], "yticks": []})
        for i in range(n_samples):
            ax[i // 2, i % 2].imshow(np.asarray(test_images[i]).reshape(2, 4), aspect="equal")
            if y_pred[i] == -1:
                ax[i // 2, i % 2].set_title("Predizione: Linea Orizzontale")
            else:
                ax[i // 2, i % 2].set_title("Predizione: Linea Verticale")
        plt.subplots_adjust(wspace=0.1, hspace=0.5)
        plt.show()