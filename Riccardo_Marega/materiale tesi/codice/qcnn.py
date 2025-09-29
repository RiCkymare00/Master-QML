# The following code is adapted from: https://arnaucasau.github.io/qiskit-machine-learning/tutorials/11_quantum_convolutional_neural_networks.html
# AIM: we wanna prove how introducting a quantum layer in a classical cnn can indeed imporve the feature extraction

'''
Theory:
First, we encode our pixelated image into a quantum circuit using a given feature map, such Qiskit’s ZFeatureMap or ZZFeatureMap or others available in the circuit library.
After encoding our image, we apply alternating convolutional and pooling layers, as defined in the next section. By applying these alternating layers, we reduce the dimensionality of our circuit until we are left with one qubit. We can then classify our input image by measuring the output of this one remaining qubit.
'''

import json
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from PIL import Image
import os
import pylatexenc
import time
import warnings

warnings.filterwarnings("ignore")

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

# --- paths (adatta se necessario) ---
dataset_image_folder = 'archive/data_object_image_2/training/image_2'
dataset_label_folder = 'archive/data_object_label_2/training/label_2'
output_folder = 'output'
# ------------------------------------------------

# Img dimension: 1242 × 375

def conv_circuit(params):
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

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length = num_qubits*3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)

    return target

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=len(sources) * 3) 
    
    qubit_mapping = {}
    local_index = 0
    for source in sources:
        qubit_mapping[source] = local_index
        local_index += 1
    for sink in sinks:
        qubit_mapping[sink] = local_index
        local_index += 1
    
    for source, sink in zip(sources, sinks):
        local_source = qubit_mapping[source]
        local_sink = qubit_mapping[sink]
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [local_source, local_sink])
        qc.barrier()
        param_index += 3

    return qc

# Image resizing to 64 x 32 followed by PCA to 16 dimensions
# --------------------------------------------------------------------------------
resize_shape = (64,32)
num_components = 16

def load_images_as_vectors(folder, resize_shape):
    data = []
    files = sorted(os.listdir(folder))
    for f in files:
        if f.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(folder, f)).convert('L')  
            img = img.resize(resize_shape)                          
            vec = np.asarray(img).flatten() / 255.0                
            data.append(vec)
    return np.array(data)

def load_binary_labels(label_folder):
    labels = []
    files = sorted(os.listdir(label_folder))
    for f in files:
        if f.endswith('.txt'):
            with open(os.path.join(label_folder, f), 'r') as file:
                lines = file.readlines()
                # Se il file è vuoto, nessuna macchina presente
                if len(lines) == 0:
                    labels.append(0)
                else:
                    # Controlliamo se c'è almeno una riga con "Car"
                    car_present = any(len(line.split())>0 and line.split()[0] == "Car" for line in lines)
                    labels.append(1 if car_present else 0)
                    print(f"File {f}: {'Car present' if car_present else 'No Car'}")
    return np.array(labels)

# Carico immagini e labels (devono corrispondere nell'ordine dei file)
x_all = load_images_as_vectors(dataset_image_folder, resize_shape)
y_all = load_binary_labels(dataset_label_folder)

if len(x_all) != len(y_all):
    raise ValueError(f"Numero immagini ({len(x_all)}) e label ({len(y_all)}) diverso. Controlla le cartelle e i nomi dei file.")

# PCA su tutto il dataset, poi split train/test
pca = PCA(n_components=num_components)
X_all_reduced = pca.fit_transform(x_all)

X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(
    X_all_reduced, y_all, test_size=0.2, random_state=42, stratify=y_all
)
# --------------------------------------------------------------------------------

feature_map = ZZFeatureMap(feature_dimension=num_components, reps = 1)

ansatz = QuantumCircuit(16, name="Ansatz")

# First Convolutional Layer (16 qubit)
ansatz.compose(conv_layer(16, "c1"), list(range(16)), inplace=True)

# First Pooling Layer: 16 → 8 qubit
ansatz.compose(pool_layer([0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], "p1"), list(range(16)), inplace=True)

# Second Convolutional Layer (8 qubit, sui qubit 8-15)
ansatz.compose(conv_layer(8, "c2"), list(range(8, 16)), inplace=True)

# Second Pooling Layer: 8 → 4 qubit
ansatz.compose(pool_layer([8, 9, 10, 11], [12, 13, 14, 15], "p2"), list(range(8, 16)), inplace=True)

# Third Convolutional Layer (4 qubit, sui qubit 12-15)
ansatz.compose(conv_layer(4, "c3"), list(range(12, 16)), inplace=True)

# Third Pooling Layer: 4 → 2 qubit
ansatz.compose(pool_layer([12, 13], [14, 15], "p3"), list(range(12, 16)), inplace=True)

# Fourth Convolutional Layer (2 qubit, sui qubit 14-15)
ansatz.compose(conv_layer(2, "c4"), list(range(14, 16)), inplace=True)

# Fourth Pooling Layer: 2 → 1 qubit
ansatz.compose(pool_layer([14], [15], "p4"), list(range(14, 16)), inplace=True)

# Combining the feature map and ansatz
circuit = QuantumCircuit(16)
circuit.compose(feature_map, range(16), inplace=True)
circuit.compose(ansatz, range(16), inplace=True)

# L'observable deve misurare il qubit finale (qubit 15)
observable = SparsePauliOp.from_list([("I"*15 + "Z", 1)])

qnn = EstimatorQNN( circuit = circuit.decompose(),
                    observables=observable,
                    input_params=feature_map.parameters,
                    weight_params=ansatz.parameters,
)

# Preparo le liste per tracciare andamento nel tempo
objective_func_vals = []
times = []
train_accuracies = []
test_accuracies = []

start_time = time.time()

# Per comodità locale usiamo questi riferimenti in callback
X_train_cb = np.asarray(X_train_reduced)
X_test_cb = np.asarray(X_test_reduced)
y_train_cb = np.asarray(y_train)
y_test_cb = np.asarray(y_test)

def callback_graph(weights, obj_func_eval):
    """
    Callback chiamata dall'ottimizzatore. 
    Riceve i pesi correnti e il valore della funzione obiettivo.
    Qui calcoliamo accuracy su train e test usando il QNN con i pesi correnti.
    """
    objective_func_vals.append(obj_func_eval)
    elapsed = time.time() - start_time
    times.append(elapsed)

    # Proviamo a ottenere l'output del QNN con i pesi correnti.
    # L'output atteso è un valore di expectation per ogni sample.
    try:
        out_train = qnn.forward(X_train_cb, weights)
        out_test = qnn.forward(X_test_cb, weights)
        out_train = np.array(out_train).ravel()
        out_test = np.array(out_test).ravel()
    except Exception as e:
        # Se forward non è disponibile oppure fallisce, stampiamo un avviso
        print("Warning: impossibile chiamare qnn.forward() durante la callback.", e)
        print("La callback continuerà registrando solo il valore dell'obiettivo.")
        # Inseriamo NaN per le accuracy
        train_accuracies.append(np.nan)
        test_accuracies.append(np.nan)
        print(f"Iter {len(objective_func_vals)} | Objective value: {obj_func_eval} | elapsed {elapsed:.1f}s")
        return

    # Mappiamo expectation -> classe binaria. 
    # Semplice regola: se expectation > 0 => classe 1, else 0.
    preds_train = (out_train > 0).astype(int)
    preds_test = (out_test > 0).astype(int)
    acc_train = np.mean(preds_train == y_train_cb)
    acc_test = np.mean(preds_test == y_test_cb)

    train_accuracies.append(acc_train)
    test_accuracies.append(acc_test)

    print(f"Iter {len(objective_func_vals)} | Objective value: {obj_func_eval:.4f} | "
          f"Train acc: {acc_train:.4f} | Test acc: {acc_test:.4f} | elapsed {elapsed:.1f}s")


classifier = NeuralNetworkClassifier(
    qnn,
    optimizer=COBYLA(maxiter=200), 
    callback=callback_graph,
)

# Fit sul solo training set ridotto
x_fit = np.asarray(X_train_reduced)
y_fit = np.asarray(y_train)

# Eseguo il training
classifier.fit(x_fit, y_fit)

# score finale classifier su train e test
train_score = np.round(100 * classifier.score(x_fit, y_fit), 2)
test_score = np.round(100 * classifier.score(np.asarray(X_test_reduced), np.asarray(y_test)), 2)
print(f"Accuracy from the train data : {train_score}%")
print(f"Accuracy from the test data  : {test_score}%")

# --- Plotting: objective nel tempo e train vs test accuracy nel tempo ---
plt.rcParams["figure.figsize"] = (12, 8)

fig, axs = plt.subplots(2, 1, sharex=True)

# Top: objective value vs time
axs[0].plot(times, objective_func_vals, marker='o', linestyle='-')
axs[0].set_ylabel("Objective value")
axs[0].set_title("Objective value over time")

# Bottom: Train vs Test accuracy vs time
# Alcune entry potrebbero essere NaN se forward non era disponibile durante callback
axs[1].plot(times, train_accuracies, marker='o', linestyle='-', label='Train accuracy')
axs[1].plot(times, test_accuracies, marker='o', linestyle='-', label='Test accuracy')
axs[1].set_ylabel("Accuracy")
axs[1].set_xlabel("Elapsed time (s)")
axs[1].set_title("Train vs Test accuracy over time")
axs[1].legend()

plt.tight_layout()
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'training_vs_test.png')
fig.savefig(output_path, dpi=300)
print(f"Plot salvato in: {os.path.abspath(output_path)}")
plt.show()
