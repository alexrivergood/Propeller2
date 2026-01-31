This is a folder reserved to solving the MNIST problem and doing the inference in the propeller 2.
It is very small model, perfect for implementation in edge devices
Around 8KB in weights
Input: 28×28×1 (grayscale MNIST image)
Conv1: 4 filters, 3×3 kernel, stride 1, same padding → 28×28×4
ReLU activation
Conv2: 8 filters, 3×3 kernel, stride 2, same padding → 14×14×8
ReLU activation
Conv3: 8 filters, 3×3 kernel, stride 1, same padding → 14×14×8
ReLU activation
Conv4: 12 filters, 3×3 kernel, stride 2, same padding → 7×7×12
ReLU activation
Global Average Pooling → 12 features
Dense layer: 12→10 (fully connected)
Softmax → 10 class probabilities
