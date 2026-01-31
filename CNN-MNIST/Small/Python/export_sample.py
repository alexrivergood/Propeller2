"""
Export a random MNIST sample as a C-friendly greyscale matrix
Simple, minimal version - just gets the job done
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load MNIST data
print("Loading MNIST data...")
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

# Pick random index
sample_index = np.random.randint(0, len(x_test))
sample = x_test[sample_index]
label = y_test[sample_index]

# Normalize to 0-1 range (as the model expects)
sample_normalized = sample.astype('float32') / 255.0

print(f"\nSample #{sample_index}")
print(f"Label: {label}")
print(f"Shape: {sample.shape}")
print(f"Min pixel: {sample.min()}, Max pixel: {sample.max()}")
print(f"Normalized min: {sample_normalized.min():.3f}, max: {sample_normalized.max():.3f}")

# Create C header file
with open("mnist_sample.h", "w") as f:
    f.write("/* MNIST sample - greyscale 28x28 matrix */\n")
    f.write(f"/* Label: {label} */\n")
    f.write(f"/* Index: {sample_index} */\n\n")
    f.write("#ifndef MNIST_SAMPLE_H\n")
    f.write("#define MNIST_SAMPLE_H\n\n")
    
    # Write as 2D array
    f.write("float mnist_sample[28][28] = {\n")
    for i in range(28):
        f.write("  {")
        for j in range(28):
            # Write normalized value
            f.write(f"{sample_normalized[i][j]:.6f}")
            if j < 27:
                f.write(", ")
        f.write("}")
        if i < 27:
            f.write(",")
        f.write("\n")
    f.write("};\n\n")
    
    # Write label constant
    f.write(f"#define SAMPLE_LABEL {label}\n")
    f.write(f"#define SAMPLE_INDEX {sample_index}\n\n")
    
    f.write("#endif /* MNIST_SAMPLE_H */\n")

print(f"\n✓ Saved to: mnist_sample.h")
print(f"✓ Format: 28x28 float array, normalized to [0.0, 1.0]")
print(f"✓ To use in C: #include \"mnist_sample.h\"")
print(f"✓ Access: mnist_sample[row][col] (0-27 indices)")