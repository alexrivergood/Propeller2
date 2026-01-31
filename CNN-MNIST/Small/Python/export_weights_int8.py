"""
Simple MNIST CNN weights exporter to C header - INT8 version
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

def quantize_to_int8(weights, bits=8):
    """
    Quantize weights to INT8 with symmetric quantization
    Returns quantized weights and scale factor
    """
    # For symmetric INT8 quantization (range: -127 to 127)
    max_val = 127
    w_min = np.min(weights)
    w_max = np.max(weights)
    
    if w_min == 0 and w_max == 0:
        # All weights are zero
        return np.zeros_like(weights, dtype=np.int8), 1.0
    
    # Calculate scale for symmetric quantization
    scale = max(abs(w_min), abs(w_max)) / max_val
    
    # Quantize weights to INT8
    quantized = np.clip(
        np.round(weights / scale),
        -max_val,
        max_val
    ).astype(np.int8)
    
    return quantized, scale

def export_int8_weights_to_header(model_path, output_file="mnist_weights_int8.h"):
    """
    Export model weights to a C header file with INT8 quantization
    """
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path)
    
    total_params = model.count_params()
    print(f"Model has {total_params:,} parameters")
    
    # Calculate memory savings
    original_size_kb = total_params * 4 / 1024  # FP32: 4 bytes per parameter
    quantized_size_kb = total_params * 1 / 1024  # INT8: 1 byte per parameter
    compression_ratio = original_size_kb / quantized_size_kb
    
    with open(output_file, "w") as f:
        # Write header
        f.write("#ifndef MNIST_WEIGHTS_INT8_H\n")
        f.write("#define MNIST_WEIGHTS_INT8_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("/*\n")
        f.write(" * MNIST CNN INT8 Quantized Weights\n")
        f.write(" * Generated with symmetric INT8 quantization\n")
        f.write(" * Format: int8_t weights (signed 8-bit integers)\n")
        f.write(" *         float scales (for dequantization)\n")
        f.write(" * Usage: dequantize = weights * scale\n")
        f.write(" */\n\n")
        
        layer_index = 0
        scale_values = []
        
        for layer in model.layers:
            weights = layer.get_weights()
            if len(weights) == 0:
                continue
                
            layer_name = layer.name.replace('/', '_').replace(':', '_')
            
            # Handle weights (kernel) - quantize to INT8
            if len(weights) > 0:
                w = weights[0]
                w_flat = w.flatten()
                
                # Quantize to INT8
                quantized_weights, scale = quantize_to_int8(w)
                quantized_flat = quantized_weights.flatten()
                
                f.write(f"// Layer: {layer.name} - shape: {w.shape}\n")
                f.write(f"// Original range: [{np.min(w):.6f}, {np.max(w):.6f}]\n")
                f.write(f"// Quantized range: [{quantized_weights.min()}, {quantized_weights.max()}]\n")
                f.write(f"// Scale factor: {scale:.8f}\n")
                f.write(f"const int8_t {layer_name}_weights[{len(quantized_flat)}] = {{\n")
                
                # Write in rows of 12
                for i in range(0, len(quantized_flat), 12):
                    chunk = quantized_flat[i:i+12]
                    values = ", ".join([f"{int(val):4d}" for val in chunk])
                    f.write(f"  {values}")
                    if i + 12 < len(quantized_flat):
                        f.write(",")
                    f.write("\n")
                f.write("};\n\n")
                
                # Store scale for later
                f.write(f"const float {layer_name}_scale = {scale:.8f}f;\n\n")
                scale_values.append(scale)
            
            # Handle biases - keep as float32 for better accuracy
            if len(weights) > 1:
                b = weights[1].flatten()
                f.write(f"// {layer.name} biases (kept as float32)\n")
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
        f.write(f"#define QUANTIZATION_BITS 8\n")
        f.write(f"#define ORIGINAL_SIZE_KB {original_size_kb:.1f}\n")
        f.write(f"#define QUANTIZED_SIZE_KB {quantized_size_kb:.1f}\n")
        f.write(f"#define COMPRESSION_RATIO {compression_ratio:.1f}\n")
        
        # Add scale array
        if scale_values:
            f.write("\n// All scale factors\n")
            f.write(f"const float all_scales[{len(scale_values)}] = {{\n")
            for i, scale in enumerate(scale_values):
                f.write(f"  {scale:.8f}f")
                if i < len(scale_values) - 1:
                    f.write(",")
                f.write("\n")
            f.write("};\n\n")
        
        # Close header guard
        f.write("\n#endif // MNIST_WEIGHTS_INT8_H\n")
    
    print(f"✓ Saved INT8 weights to: {output_file}")
    print(f"✓ Total parameters: {total_params}")
    print(f"✓ Original size (FP32): {original_size_kb:.1f} KB")
    print(f"✓ Quantized size (INT8): {quantized_size_kb:.1f} KB")
    print(f"✓ Compression ratio: {compression_ratio:.1f}x")
    print(f"✓ Memory savings: {original_size_kb - quantized_size_kb:.1f} KB")

def main_int8():
    """
    INT8 weights exporter main function
    """
    print("="*60)
    print("MNIST CNN INT8 WEIGHTS EXPORTER")
    print("="*60)
    
    # Look for model file
    model_file = None
    
    # Look in mnist_result_int8 folder
    if os.path.exists("mnist_result_int8"):
        # Find the most recent run
        runs = []
        for root, dirs, files in os.walk("mnist_result_int8"):
            for dir_name in dirs:
                if dir_name.startswith("run_"):
                    runs.append(dir_name)
        
        if runs:
            runs.sort(reverse=True)  # Most recent first
            latest_run = runs[0]
            run_path = os.path.join("mnist_result_int8", latest_run)
            
            # Look for quantized model
            quantized_path = os.path.join(run_path, "quantized_model_int8.keras")
            original_path = os.path.join(run_path, "original_model_fp32.keras")
            
            if os.path.exists(quantized_path):
                model_file = quantized_path
                print(f"Found INT8 quantized model: {quantized_path}")
            elif os.path.exists(original_path):
                model_file = original_path
                print(f"Found FP32 model (will quantize): {original_path}")
    
    # If not found, try current directory
    if not model_file:
        possible_files = ["quantized_model_int8.keras", "small_mnist_cnn.keras", "model.keras"]
        for file_name in possible_files:
            if os.path.exists(file_name):
                model_file = file_name
                print(f"Found model: {file_name}")
                break
    
    if not model_file:
        print("ERROR: No model found!")
        print("Please run the INT8 quantization experiment first.")
        print("Looking for files in: mnist_result_int8/ or current directory")
        return
    
    # Export INT8 weights
    export_int8_weights_to_header(model_file, "mnist_weights_int8.h")
    
    print("\n" + "="*60)
    print("INT8 WEIGHTS EXPORT COMPLETE")
    print("="*60)
    print("\nHeader file contains:")
    print("  • int8_t arrays for weights")
    print("  • float arrays for biases")
    print("  • Scale factors for dequantization")
    print("  • Model statistics and constants")
    print("\nUse in C/C++ code:")
    print("  #include \"mnist_weights_int8.h\"")
    print("  float weight = int8_weight * scale; // Dequantize")
    print("="*60)

if __name__ == "__main__":
    main_int8()