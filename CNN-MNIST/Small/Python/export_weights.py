"""
Simple MNIST CNN weights exporter to C header
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

def export_weights_to_header(model_path, output_file="mnist_weights.h"):
    """
    Export model weights to a C header file
    """
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path)
    
    total_params = model.count_params()
    print(f"Model has {total_params:,} parameters")
    
    with open(output_file, "w") as f:
        # Write header
        f.write("#ifndef MNIST_WEIGHTS_H\n")
        f.write("#define MNIST_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        
        layer_index = 0
        for layer in model.layers:
            weights = layer.get_weights()
            if len(weights) == 0:
                continue
                
            layer_name = layer.name.replace('/', '_').replace(':', '_')
            
            # Handle weights (kernel)
            if len(weights) > 0:
                w = weights[0].flatten()
                f.write(f"// {layer.name} weights - shape: {weights[0].shape}\n")
                f.write(f"const float {layer_name}_weights[{len(w)}] = {{\n")
                
                # Write in rows of 8
                for i in range(0, len(w), 8):
                    chunk = w[i:i+8]
                    values = ", ".join([f"{val:.8f}f" for val in chunk])
                    f.write(f"  {values}")
                    if i + 8 < len(w):
                        f.write(",")
                    f.write("\n")
                f.write("};\n\n")
            
            # Handle biases
            if len(weights) > 1:
                b = weights[1].flatten()
                f.write(f"// {layer.name} biases\n")
                f.write(f"const float {layer_name}_biases[{len(b)}] = {{\n")
                
                for i in range(0, len(b), 8):
                    chunk = b[i:i+8]
                    values = ", ".join([f"{val:.8f}f" for val in chunk])
                    f.write(f"  {values}")
                    if i + 8 < len(b):
                        f.write(",")
                    f.write("\n")
                f.write("};\n\n")
            
            layer_index += 1
        
        # Add useful constants
        f.write("\n// Model info\n")
        f.write(f"#define TOTAL_PARAMS {total_params}\n")
        
        # Close header guard
        f.write("\n#endif // MNIST_WEIGHTS_H\n")
    
    print(f"✓ Saved weights to: {output_file}")
    print(f"✓ Total parameters: {total_params}")
    
    # Verify size
    model_size_kb = total_params * 4 / 1024  # 4 bytes per float
    print(f"✓ Model size: {model_size_kb:.1f} KB")

def main():
    """
    Simple weights exporter main function
    """
    print("="*60)
    print("MNIST CNN WEIGHTS EXPORTER")
    print("="*60)
    
    # Find the most recent model
    model_file = None
    
    # Look for model in mnist_results
    if os.path.exists("mnist_results"):
        latest_run = None
        for root, dirs, files in os.walk("mnist_results"):
            for dir_name in dirs:
                if dir_name.startswith("run_"):
                    run_path = os.path.join(root, dir_name)
                    model_path = os.path.join(run_path, "small_mnist_cnn.keras")
                    if os.path.exists(model_path):
                        latest_run = model_path
        
        if latest_run:
            model_file = latest_run
    
    # If not found, try current directory
    if not model_file and os.path.exists("small_mnist_cnn.keras"):
        model_file = "small_mnist_cnn.keras"
    
    if not model_file:
        print("ERROR: No model found!")
        print("Please train the model first or specify a model path.")
        return
    
    # Export weights
    export_weights_to_header(model_file, "mnist_weights.h")

if __name__ == "__main__":
    main()