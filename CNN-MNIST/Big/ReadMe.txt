Folder reserved for the inference of a larger CNN model
Input: 28×28×1 (784)
Conv1: 32 filters, 3×3 kernel, stride 1 → 26×26×32
ReLU
MaxPool 2×2 → 13×13×32
Conv2: 64 filters, 3×3 kernel, stride 1 → 11×11×64
ReLU
MaxPool 2×2 → 5×5×64
Flatten → 1600 neurons
FC1: 128 neurons
ReLU
FC2: 10 neurons (output)
Softmax
Weight with fp32: 879KB
