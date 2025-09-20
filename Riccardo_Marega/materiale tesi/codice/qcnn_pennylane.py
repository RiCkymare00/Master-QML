import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

resize_shape = (64, 32)
num_components = 16
n_qubits = 16
n_params = 126

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

dataset_training_path = 'archive/data_object_image_2/training/image_2'
dataset_testing_path = 'archive/data_object_image_2/training/image_2'
x_train = load_images_as_vectors(dataset_training_path, resize_shape)
x_test  = load_images_as_vectors(dataset_testing_path, resize_shape)
pca = PCA(n_components=num_components)
X_train_reduced = pca.fit_transform(x_train)
X_test_reduced  = pca.transform(x_test)

def load_binary_labels(label_folder):
    labels = []
    files = sorted(os.listdir(label_folder))
    for f in files:
        if f.endswith('.txt'):
            with open(os.path.join(label_folder, f), 'r') as file:
                lines = file.readlines()
                if len(lines) == 0:
                    labels.append(0)
                else:
                    car_present = any(line.split()[0] == "Car" for line in lines)
                    labels.append(1 if car_present else 0)
    return np.array(labels)

train_labels = load_binary_labels("archive/data_object_label_2/training/label_2")
y_train = np.asarray(train_labels)

try:
    dev = qml.device("lightning.gpu", wires=n_qubits)
except Exception:
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)

def feature_map(x):
    for i in range(min(len(x), n_qubits)):
        qml.RZ(x[i] * np.pi, wires=i)
    for i in range(0, n_qubits-1, 2):
        qml.CNOT(wires=[i, i+1])
    for i in range(1, n_qubits-1, 2):
        qml.CNOT(wires=[i, i+1])

def conv_block(params, wires):
    qml.RZ(-np.pi/2, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(np.pi/2, wires=wires[0])

def pool_block(params, source, sink):
    qml.RZ(-np.pi/2, wires=sink)
    qml.CNOT(wires=[sink, source])
    qml.RZ(params[0], wires=source)
    qml.RY(params[1], wires=sink)
    qml.CNOT(wires=[source, sink])
    qml.RY(params[2], wires=sink)

def QCNN_circuit(x, weights):
    x_np = np.array(x)
    w_np = np.array(weights)
    feature_map(x_np)
    idx = 0
    for q1, q2 in zip(range(0,16,2), range(1,16,2)):
        conv_block(w_np[idx:idx+3], [q1, q2]); idx += 3
    for q1, q2 in zip(range(1,16,2), list(range(2,16,2))+[0]):
        conv_block(w_np[idx:idx+3], [q1, q2]); idx += 3
    for s, t in zip(list(range(0,8)), list(range(8,16))):
        pool_block(w_np[idx:idx+3], s, t); idx += 3
    for q1, q2 in zip(range(8,16,2), range(9,16,2)):
        conv_block(w_np[idx:idx+3], [q1, q2]); idx += 3
    for q1, q2 in zip(range(9,16,2), list(range(10,16,2))+[8]):
        conv_block(w_np[idx:idx+3], [q1, q2]); idx += 3
    for s, t in zip(list(range(8,12)), list(range(12,16))):
        pool_block(w_np[idx:idx+3], s, t); idx += 3
    for q1, q2 in zip(range(12,16,2), range(13,16,2)):
        conv_block(w_np[idx:idx+3], [q1, q2]); idx += 3
    pool_block(w_np[idx:idx+3], 12, 14); idx += 3
    pool_block(w_np[idx:idx+3], 13, 15); idx += 3
    conv_block(w_np[idx:idx+3], [14, 15]); idx += 3
    pool_block(w_np[idx:idx+3], 14, 15); idx += 3
    return qml.expval(qml.PauliZ(15))

qnode = qml.QNode(QCNN_circuit, dev, interface='autograd', diff_method='adjoint')

y_mapped = (2*y_train - 1).astype(float)

def loss_fn_scipy(w):
    w_p = pnp.array(w, requires_grad=False)
    total = 0.0
    for i in range(len(X_train_reduced)):
        x = X_train_reduced[i]
        pred = qnode(x, w_p)
        total += (pred - y_mapped[i])**2
    return float(total / len(X_train_reduced))

iter_tracker = {'n': 0}
def callback_scipy(xk):
    iter_tracker['n'] += 1
    print(f"Iter {iter_tracker['n']} | Loss: {loss_fn_scipy(xk):.8f}")

initial_weights = np.random.normal(0, 0.1, size=n_params)
print("Starting training...")
res = minimize(loss_fn_scipy, initial_weights, method='COBYLA', callback=callback_scipy, options={'maxiter':200, 'disp':True})

trained_weights = res.x

n_test = min(100, len(X_train_reduced))
preds = []
for i in range(n_test):
    preds.append(qnode(X_train_reduced[i], pnp.array(trained_weights)))
preds = np.array(preds)
pred_labels = (preds < 0).astype(int)
true_labels = y_train[:n_test]
accuracy = np.mean(pred_labels == true_labels)
print(f"Accuracy on {n_test} samples: {accuracy*100:.2f}%")

plt.figure(figsize=(8,5))
plt.plot([loss_fn_scipy(res.x)]) 
plt.xlabel("Evaluation")
plt.ylabel("Loss")
plt.title("Final Loss")
plt.grid(True)
plt.show()