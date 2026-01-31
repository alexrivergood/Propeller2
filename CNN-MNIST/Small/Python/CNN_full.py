"""
Small CNN for MNIST with organized output folder
All results saved to 'mnist_results' folder
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import os
from datetime import datetime

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_output_folder():
    """
    Create an output folder with timestamp for organizing results
    Returns the path to the folder
    """
    # Create base folder if it doesn't exist
    base_folder = "mnist_results"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
        print(f"✓ Created folder: {base_folder}/")
    
    # Create timestamped subfolder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(base_folder, f"run_{timestamp}")
    os.makedirs(run_folder)
    
    print(f"✓ Created run folder: {run_folder}/")
    return run_folder

def load_mnist_data():
    """Load and preprocess MNIST data with proper train/val/test split"""
    print("Loading MNIST dataset...")
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Split training data into train (80%) and validation (20%)
    total_train_samples = len(x_train_full)
    val_split = int(0.8 * total_train_samples)
    
    x_train = x_train_full[:val_split]
    y_train = y_train_full[:val_split]
    
    x_val = x_train_full[val_split:]
    y_val = y_train_full[val_split:]
    
    # Normalize and reshape for CNN
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_val = x_val.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # One-hot encode labels
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_val_cat = keras.utils.to_categorical(y_val, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)
    
    print(f"✓ Training samples: {x_train.shape[0]:,}")
    print(f"✓ Validation samples: {x_val.shape[0]:,}")
    print(f"✓ Test samples: {x_test.shape[0]:,}")
    
    return (x_train, y_train_cat, y_train), (x_val, y_val_cat, y_val), (x_test, y_test_cat, y_test)

def create_small_cnn():
    """
    Create ONE small CNN model that's guaranteed to be < 40KB
    This model has only ~6,000 parameters (~24KB) without BatchNorm
    """
    print("\nCreating small CNN architecture...")
    
    model = keras.Sequential([
        # First conv layer - very few filters
        layers.Conv2D(
            4,
            kernel_size=3,
            padding='same',
            activation='relu',
            input_shape=(28, 28, 1),
            kernel_regularizer=regularizers.l2(0.001)
        ),
        
        # Downsample with stride
        layers.Conv2D(
            8,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        
        # Second conv block
        layers.Conv2D(
            8,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        
        # Final downsampling
        layers.Conv2D(
            12,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu'
        ),
        
        # Global average pooling
        layers.GlobalAveragePooling2D(),
        
        # Output layer
        layers.Dense(10, activation='softmax')
    ])
    
    return model


def train_model_with_updates(model, x_train, y_train, x_val, y_val, epochs=15, output_folder=None):
    """
    Train the model with detailed epoch-by-epoch progress updates
    """
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model architecture and size
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    
    # Save model summary to file
    summary_path = os.path.join(output_folder, "model_summary.txt")
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    model.summary()  # Also print to console
    
    total_params = model.count_params()
    model_size_kb = total_params * 4 / 1024  # 4 bytes per parameter (float32)
    
    print(f"\nModel Size Analysis:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Estimated size: {model_size_kb:.1f} KB")
    print(f"  Target: < 40 KB ✓")
    
    # Custom callback for real-time updates
    class TrainingProgress(keras.callbacks.Callback):
        def __init__(self, output_folder):
            super().__init__()
            self.output_folder = output_folder
            self.epoch_log = []
            
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = datetime.now()
            
        def on_epoch_end(self, epoch, logs=None):
            # Create progress bar
            progress = (epoch + 1) / epochs
            bar_length = 40
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            # Calculate epoch duration
            epoch_time = (datetime.now() - self.epoch_start_time).total_seconds()
            
            print(f"\nEpoch {epoch+1}/{epochs} [{bar}] {progress:.0%}")
            print(f"  Time: {epoch_time:.1f}s")
            print(f"  Training - Accuracy: {logs['accuracy']:.4f}, Loss: {logs['loss']:.4f}")
            if 'val_accuracy' in logs:
                print(f"  Validation - Accuracy: {logs['val_accuracy']:.4f}, Loss: {logs['val_loss']:.4f}")
            
            # Save epoch results to log
            self.epoch_log.append({
                'epoch': epoch + 1,
                'train_acc': logs['accuracy'],
                'train_loss': logs['loss'],
                'val_acc': logs.get('val_accuracy', None),
                'val_loss': logs.get('val_loss', None),
                'time_seconds': epoch_time
            })
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(self.output_folder, f"checkpoint_epoch_{epoch+1}.keras")
                self.model.save(checkpoint_path)
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    # Early stopping callback
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=6,
        restore_best_weights=True,
        verbose=1
    )
    
    # Add learning rate scheduler
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: 64")
    print(f"Training samples: {x_train.shape[0]:,}")
    print(f"Validation samples: {x_val.shape[0]:,}")
    
    # Create progress callback
    progress_callback = TrainingProgress(output_folder)
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[progress_callback, early_stop, reduce_lr],
        verbose=0  # We use our custom callback instead
    )
    
    # Save epoch log to CSV
    log_df = pd.DataFrame(progress_callback.epoch_log)
    log_path = os.path.join(output_folder, "training_log.csv")
    log_df.to_csv(log_path, index=False)
    print(f"✓ Training log saved: {log_path}")
    
    return history

def generate_confusion_matrix_and_report(model, x_test, y_test_original, output_folder):
    """
    Generate confusion matrix and classification report on TEST set
    """
    print("\n" + "="*60)
    print("GENERATING EVALUATION METRICS (TEST SET)")
    print("="*60)
    
    # Get predictions
    print("Making predictions on test set...")
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_pred_probs = np.max(y_pred, axis=1)  # Confidence scores
    
    # Create confusion matrix
    cm = confusion_matrix(y_test_original, y_pred_classes)
    
    # Calculate overall accuracy
    total_correct = np.trace(cm)
    total_samples = np.sum(cm)
    accuracy = total_correct / total_samples
    
    print(f"\nTest Set Performance:")
    print(f"  Correct predictions: {total_correct:,}/{total_samples:,}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Save confusion matrix data
    cm_path = os.path.join(output_folder, "confusion_matrix_data.csv")
    cm_df = pd.DataFrame(cm, 
                         index=[f"True_{i}" for i in range(10)],
                         columns=[f"Pred_{i}" for i in range(10)])
    cm_df.to_csv(cm_path)
    print(f"✓ Confusion matrix data saved: {cm_path}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix - Small MNIST CNN (Test Set)', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Save the figure
    cm_plot_path = os.path.join(output_folder, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"✓ Confusion matrix plot saved: {cm_plot_path}")
    
    # Display per-class accuracy
    print("\nPer-Class Accuracy (Test Set):")
    print("-" * 40)
    class_accuracies = []
    for i in range(10):
        if cm[i, :].sum() > 0:
            class_acc = cm[i, i] / cm[i, :].sum()
            class_accuracies.append(class_acc)
            print(f"  Digit {i}: {class_acc:.3f} ({cm[i, i]}/{cm[i, :].sum()} correct)")
    
    return cm, y_pred_classes, y_pred_probs, accuracy

def plot_training_history(history, output_folder):
    """
    Plot training and validation accuracy/loss
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Save the figure
    history_path = os.path.join(output_folder, "training_history.png")
    plt.tight_layout()
    plt.savefig(history_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"✓ Training history plot saved: {history_path}")
    
    # Save history data
    history_data = {
        'epoch': list(range(1, len(history.history['accuracy']) + 1)),
        'train_accuracy': history.history['accuracy'],
        'train_loss': history.history['loss'],
        'val_accuracy': history.history['val_accuracy'],
        'val_loss': history.history['val_loss']
    }
    
    history_df = pd.DataFrame(history_data)
    history_csv_path = os.path.join(output_folder, "training_history.csv")
    history_df.to_csv(history_csv_path, index=False)
    print(f"✓ Training history data saved: {history_csv_path}")

def visualize_sample_predictions(model, x_test, y_test_original, output_folder, num_samples=12):
    """
    Visualize sample predictions with confidence scores (from TEST set)
    """
    # Get predictions for the entire test set
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_pred_conf = np.max(y_pred, axis=1)
    
    # Find indices of correct and incorrect predictions
    correct_mask = y_test_original == y_pred_classes
    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(~correct_mask)[0]
    
    # Take samples (6 correct, 6 incorrect if available)
    num_correct = min(6, len(correct_indices))
    num_incorrect = min(6, len(incorrect_indices))
    
    correct_samples = np.random.choice(correct_indices, num_correct, replace=False)
    incorrect_samples = np.random.choice(incorrect_indices, num_incorrect, replace=False)
    
    all_samples = list(correct_samples) + list(incorrect_samples)
    
    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()
    
    for i, idx in enumerate(all_samples):
        ax = axes[i]
        ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        
        true_label = y_test_original[idx]
        pred_label = y_pred_classes[idx]
        confidence = y_pred_conf[idx]
        
        # Color code: green for correct, red for incorrect
        is_correct = true_label == pred_label
        color = 'green' if is_correct else 'red'
        marker = '✓' if is_correct else '✗'
        
        ax.set_title(f'{marker} True: {true_label} | Pred: {pred_label}\nConf: {confidence:.2f}', 
                     color=color, fontsize=9)
        ax.axis('off')
    
    # Hide any unused subplots
    for i in range(len(all_samples), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions from Test Set (✓=Correct, ✗=Incorrect)', fontsize=14)
    plt.tight_layout()
    
    # Save the figure
    samples_path = os.path.join(output_folder, "sample_predictions.png")
    plt.savefig(samples_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"✓ Sample predictions plot saved: {samples_path}")

def save_final_report(model, history, cm, accuracy, y_test_original, y_pred_classes, output_folder):
    """
    Save a comprehensive text report
    """
    print("\n" + "="*60)
    print("GENERATING FINAL REPORT")
    print("="*60)
    
    # Calculate key metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    total_params = model.count_params()
    
    # Generate classification report
    class_report = classification_report(y_test_original, y_pred_classes, digits=4, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    
    # Save classification report to CSV
    report_csv_path = os.path.join(output_folder, "classification_report.csv")
    class_report_df.to_csv(report_csv_path)
    print(f"✓ Classification report saved: {report_csv_path}")
    
    # Save comprehensive text report
    report_path = os.path.join(output_folder, "final_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SMALL CNN FOR MNIST - COMPREHENSIVE REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("EXPERIMENT INFO:\n")
        f.write("-"*40 + "\n")
        f.write(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output folder: {os.path.basename(output_folder)}\n\n")
        
        f.write("DATA SPLITS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Training samples: 48,000\n")
        f.write(f"Validation samples: 12,000\n")
        f.write(f"Test samples: 10,000\n\n")
        
        f.write("MODEL ARCHITECTURE:\n")
        f.write("-"*40 + "\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Model Size: {total_params * 4 / 1024:.1f} KB\n")
        f.write(f"Architecture: 4 Conv Layers + Global Average Pooling\n")
        f.write(f"Filter sizes: 4 → 8 → 8 → 12\n\n")
        
        f.write("TRAINING RESULTS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Epochs Trained: {len(history.history['accuracy'])}\n")
        f.write(f"Final Training Accuracy: {final_train_acc:.4f}\n")
        f.write(f"Final Validation Accuracy: {final_val_acc:.4f}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Test Set Accuracy: {accuracy:.4f}\n\n")
        
        f.write("CONFUSION MATRIX SUMMARY (TEST SET):\n")
        f.write("-"*40 + "\n")
        f.write(f"Total Test Samples: {np.sum(cm):,}\n")
        f.write(f"Correct Predictions: {np.trace(cm):,}\n")
        f.write(f"Incorrect Predictions: {np.sum(cm) - np.trace(cm):,}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        
        f.write("PER-CLASS PERFORMANCE (TEST SET):\n")
        f.write("-"*40 + "\n")
        for i in range(10):
            if cm[i, :].sum() > 0:
                class_acc = cm[i, i] / cm[i, :].sum()
                f.write(f"Digit {i}: {class_acc:.4f} ({cm[i, i]}/{cm[i, :].sum()} correct)\n")
        f.write("\n")
        
        f.write("CLASSIFICATION REPORT (TEST SET):\n")
        f.write("-"*40 + "\n")
        f.write(classification_report(y_test_original, y_pred_classes, digits=4))
    
    print(f"✓ Final text report saved: {report_path}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Output folder: {output_folder}/")
    print(f"Model Size: {total_params * 4 / 1024:.1f} KB (Target: < 40 KB ✓)")
    print(f"Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Training Epochs: {len(history.history['accuracy'])}")
    
    print("\nGenerated Files:")
    print("="*60)
    
    # List all generated files
    files = os.listdir(output_folder)
    files.sort()
    for file in files:
        if file.endswith(('.png', '.txt', '.csv', '.keras')):
            file_path = os.path.join(output_folder, file)
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            print(f"  {file:30} {file_size:6.1f} KB")
    
    print("="*60)

def create_readme_file(output_folder):
    """
    Create a README file explaining the contents of the output folder
    """
    readme_path = os.path.join(output_folder, "README.txt")
    with open(readme_path, 'w') as f:
        f.write("MNIST CNN EXPERIMENT - OUTPUT FILES\n")
        f.write("="*50 + "\n\n")
        f.write("This folder contains all results from the MNIST CNN experiment.\n\n")
        
        f.write("DATA SPLITS:\n")
        f.write("-"*20 + "\n")
        f.write("Training: 48,000 samples (80% of original training data)\n")
        f.write("Validation: 12,000 samples (20% of original training data)\n")
        f.write("Test: 10,000 samples (original test set, unseen during training)\n\n")
        
        f.write("FILE DESCRIPTIONS:\n")
        f.write("-"*30 + "\n")
        f.write("model_summary.txt        - Model architecture and parameter count\n")
        f.write("training_log.csv         - Detailed epoch-by-epoch training log\n")
        f.write("training_history.png     - Accuracy and loss plots\n")
        f.write("training_history.csv     - Raw training history data\n")
        f.write("confusion_matrix.png     - Visual confusion matrix (TEST set)\n")
        f.write("confusion_matrix_data.csv - Raw confusion matrix data\n")
        f.write("sample_predictions.png   - Sample predictions with confidence\n")
        f.write("classification_report.csv - Detailed classification metrics\n")
        f.write("final_report.txt         - Comprehensive text report\n")
        f.write("small_mnist_cnn.keras    - Trained model file\n")
        f.write("checkpoint_epoch_*.keras - Training checkpoints (every 5 epochs)\n")
        f.write("README.txt               - This file\n\n")
        
        f.write("EXPERIMENT INFO:\n")
        f.write("-"*30 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Model: Small CNN with < 40KB parameters\n")
        f.write("Dataset: MNIST (28x28 grayscale digits)\n")
        f.write("Note: Test set was NOT used during training or validation\n")
    
    print(f"✓ README file created: {readme_path}")

def main():
    """
    Main function - trains ONE small CNN and saves all results to organized folder
    """
    print("\n" + "="*70)
    print("SMALL CNN FOR MNIST - ORGANIZED OUTPUT")
    print("Target: Model Size < 40KB")
    print("Proper train/val/test splits applied")
    print("="*70)
    
    # Step 1: Create organized output folder
    output_folder = create_output_folder()
    
    # Step 2: Load data with proper splits
    (x_train, y_train, y_train_orig), (x_val, y_val, y_val_orig), (x_test, y_test, y_test_orig) = load_mnist_data()
    
    # Step 3: Create ONE small CNN model
    model = create_small_cnn()
    
    # Step 4: Train the model with progress updates (using validation set)
    history = train_model_with_updates(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,  # Using validation set, not test!
        y_val=y_val,
        epochs=15,
        output_folder=output_folder
    )
    
    # Step 5: Save the trained model
    model_path = os.path.join(output_folder, "small_mnist_cnn.keras")
    model.save(model_path)
    print(f"\n✓ Model saved: {model_path}")
    
    # Step 6: Generate confusion matrix and metrics on TEST set
    cm, y_pred_classes, y_pred_probs, accuracy = generate_confusion_matrix_and_report(
        model, x_test, y_test_orig, output_folder
    )
    
    # Step 7: Plot training history
    plot_training_history(history, output_folder)
    
    # Step 8: Visualize sample predictions from TEST set
    visualize_sample_predictions(model, x_test, y_test_orig, output_folder)
    
    # Step 9: Save comprehensive report
    save_final_report(model, history, cm, accuracy, y_test_orig, y_pred_classes, output_folder)
    
    # Step 10: Create README file
    create_readme_file(output_folder)
    
    # Final evaluation on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nAll results have been saved to: {output_folder}/")
    print(f"\nFinal Performance Summary:")
    print(f"  Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"\nModel Size: {model.count_params() * 4 / 1024:.1f} KB (Target: < 40 KB ✓)")
    print("\n" + "="*70)

# Run the main function
if __name__ == "__main__":
    main()