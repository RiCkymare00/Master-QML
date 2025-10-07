import math
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from qcnn import QCNN
from qiskit_algorithms.utils import algorithm_globals
from ccnn import CCNN 

N_IMAGES = 5000
TEST_SIZE = 0.3
SEED = 246
BUILDER_SEED = 12345
MAXITER = 200  
CCNN_EPOCHS = 30
CCNN_BATCH = 8
CCNN_LR = 1e-3

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
algorithm_globals.random_seed = BUILDER_SEED

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def generate_dataset(num_images: int):
    images = []
    labels = []
    hor_array = np.zeros((6, 8))
    ver_array = np.zeros((4, 8))

    j = 0
    for i in range(0, 7):
        if i != 3:
            hor_array[j][i] = np.pi / 2
            hor_array[j][i + 1] = np.pi / 2
            j += 1

    j = 0
    for i in range(0, 4):
        ver_array[j][i] = np.pi / 2
        ver_array[j][i + 4] = np.pi / 2
        j += 1

    for n in range(num_images):
        rng = algorithm_globals.random.integers(0, 2)
        if rng == 0:
            labels.append(-1)
            random_image = algorithm_globals.random.integers(0, 6)
            images.append(np.array(hor_array[random_image]))
        else:
            labels.append(1)
            random_image = algorithm_globals.random.integers(0, 4)
            images.append(np.array(ver_array[random_image]))

        # Create noise
        for i in range(8):
            if images[-1][i] == 0:
                images[-1][i] = algorithm_globals.random.uniform(0, np.pi / 4)
    return np.asarray(images), np.asarray(labels)


X, y = generate_dataset(N_IMAGES)  

train_images, test_images, train_labels, test_labels = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)


train_labels_ccnn = (train_labels == 1).astype(np.int64)
test_labels_ccnn = (test_labels == 1).astype(np.int64)


def reshape_for_ccnn(images: np.ndarray) -> np.ndarray:
    return images.reshape((-1, 1, 2, 4)).astype(np.float32)

train_images_ccnn = reshape_for_ccnn(train_images)
test_images_ccnn = reshape_for_ccnn(test_images)


train_dataset = TensorDataset(torch.from_numpy(train_images_ccnn), torch.from_numpy(train_labels_ccnn).long())
test_dataset = TensorDataset(torch.from_numpy(test_images_ccnn), torch.from_numpy(test_labels_ccnn).long())

train_loader = DataLoader(train_dataset, batch_size=CCNN_BATCH, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=CCNN_BATCH, shuffle=False)

builder = QCNN(n_qubits=8, seed=BUILDER_SEED)
builder.build_ansatz_8qubits()
builder.build_circuit()

print("Training QCNN in corso...")
builder.train(train_images, train_labels, maxiter=MAXITER)
print("Training QCNN completato.")

train_acc_qcnn = builder.score(train_images, train_labels)
test_pred_qcnn = builder.predict(test_images)
test_acc_qcnn = builder.score(test_images, test_labels)

print(f"QCNN - Accuracy train: {np.round(100 * train_acc_qcnn, 2)}%")
print(f"QCNN - Accuracy test:  {np.round(100 * test_acc_qcnn, 2)}%")


in_dim = (1, 2, 4)
num_classes = 2
model = CCNN(in_dim=in_dim, num_classes=num_classes).to(device)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CCNN_LR)

def train_ccnn_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)  
        log_probs = torch.log(outputs + 1e-9) 
        loss = criterion(log_probs, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)

def evaluate_ccnn_using_class(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds_batch = model.classify(xb).cpu().numpy()
            preds.extend(preds_batch.tolist())
            trues.extend(yb.numpy().tolist())
    return np.array(preds), np.array(trues)

print("Training CCNN in corso...")
for epoch in range(1, CCNN_EPOCHS + 1):
    loss = train_ccnn_one_epoch(model, train_loader, optimizer, criterion, device)
    if epoch % 5 == 0 or epoch == 1:
        preds_train, trues_train = evaluate_ccnn_using_class(model, train_loader, device)
        acc_train = accuracy_score(trues_train, preds_train)
        print(f"Epoch {epoch:02d}/{CCNN_EPOCHS} - Loss: {loss:.4f} - Train acc (via class): {acc_train:.4f}")
print("Training CCNN completato.")

preds_train, trues_train = evaluate_ccnn_using_class(model, train_loader, device)
preds_test, trues_test = evaluate_ccnn_using_class(model, test_loader, device)

train_acc_ccnn = accuracy_score(trues_train, preds_train)
test_acc_ccnn = accuracy_score(trues_test, preds_test)

print(f"CCNN - Accuracy train: {np.round(100 * train_acc_ccnn, 2)}%")
print(f"CCNN - Accuracy test:  {np.round(100 * test_acc_ccnn, 2)}%")

print("\n--- Confronto finale ---")
print(f"QCNN: train {np.round(100*train_acc_qcnn,2)}%  | test {np.round(100*test_acc_qcnn,2)}%")
print(f"CCNN: train {np.round(100*train_acc_ccnn,2)}%  | test {np.round(100*test_acc_ccnn,2)}%")
