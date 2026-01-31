#!/usr/bin/env python3
#Biases have been kept as floats
#Weights have been quantized, from float to int8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))     # -> 32×26×26
        x = F.max_pool2d(x, 2)        # -> 32×13×13
        x = F.relu(self.conv2(x))     # -> 64×11×11
        x = F.max_pool2d(x, 2)        # -> 64×5×5
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

print("Training CNN...")
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: loss={total_loss/len(train_loader):.4f}")


model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

accuracy = 100. * correct / len(test_loader.dataset)
print(f"Test Accuracy: {accuracy:.2f}%")



def quantize_tensor_to_int8(arr: np.ndarray):
    """Symmetric per-tensor quantization to int8 [-127,127].
       Returns (q_int8, scale_float)."""
    max_abs = float(np.max(np.abs(arr)))
    if max_abs == 0.0:
        scale = 1.0
        q = np.zeros_like(arr, dtype=np.int8)
    else:
        scale = max_abs / 127.0
        qf = np.round(arr / scale)
        qf = np.clip(qf, -127, 127)
        q = qf.astype(np.int8)
    return q, float(scale)

header_name = "mnist_cnn_weights_q.h"
with open(header_name, "w") as f:
    f.write("// Auto-generated quantized MNIST CNN weights\n")
    f.write("// Weights are quantized per-tensor to int8 [-127..127].\n")
    f.write("// For inference, dequantize each weight: float(w_q) * <layer>_scale\n\n")
    f.write("#ifndef MNIST_CNN_WEIGHTS_Q_H\n#define MNIST_CNN_WEIGHTS_Q_H\n\n")
    f.write("#include <stdint.h>\n\n")

    for idx, (name, param) in enumerate(model.named_parameters()):
        arr = param.detach().cpu().numpy()
        shape = list(arr.shape)
        cname = name.replace('.', '_')  # e.g., conv1_weight

        
        if name.endswith('weight'):
            q, scale = quantize_tensor_to_int8(arr)
            flat = q.flatten()
            f.write(f"// Layer {idx}: {name}, shape={shape}\n")
            f.write(f"static const int8_t {cname}_q[{flat.size}] = {{\n")
            for i in range(0, flat.size, 16):
                chunk = flat[i:i+16]
                f.write("  " + ", ".join(f"{int(int(x))}" for x in chunk) + ",\n")
            f.write("};\n")
            f.write(f"static const float {cname}_scale = {scale:.8e}f;\n\n")
        else:
            # biases (float)
            flat = arr.flatten()
            f.write(f"// Layer {idx}: {name}, shape={shape}\n")
            f.write(f"static const float {cname}[{flat.size}] = {{\n")
            for i in range(0, flat.size, 8):
                f.write("  " + ", ".join(f"{float(x):.8e}f" for x in flat[i:i+8]) + ",\n")
            f.write("};\n\n")

    f.write("#endif // MNIST_CNN_WEIGHTS_Q_H\n")

print(f"Exported quantized weights to {header_name}")

# Save model state too
torch.save(model.state_dict(), "mnist_cnn.pth")
print("Saved model to mnist_cnn.pth")
