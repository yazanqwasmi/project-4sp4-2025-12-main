import sys
import csv
import numpy as np
FOLDER = './data/model/'

# Load weights from CSV files and sparsify them
# store them as csv files in the same folder with the described naming convention

def load_weight_matrix(filename):
    """Load a weight matrix from a CSV file."""
    return np.loadtxt(filename, delimiter=',', dtype=np.float32)

def magnitude_based_pruning(weight_matrix, sparsity_level):
    """
    Apply magnitude-based pruning to a weight matrix.
    
    Args:
        weight_matrix: numpy array of weights
        sparsity_level: percentage of weights to zero out (0-100)
    
    Returns:
        Pruned weight matrix with the smallest magnitude weights set to zero
    """
    # Create a copy to avoid modifying original
    pruned = weight_matrix.copy()
    
    # Flatten to get all weights
    flat_weights = np.abs(pruned.flatten())
    
    # Calculate threshold: find the value at the sparsity percentile
    threshold = np.percentile(flat_weights, sparsity_level)
    
    # Zero out weights below threshold
    pruned[np.abs(pruned) < threshold] = 0.0
    
    return pruned

def save_weight_matrix(weight_matrix, filename):
    """Save a weight matrix to a CSV file."""
    np.savetxt(filename, weight_matrix, delimiter=',', fmt='%.8e')

def main():
    # Load the original weight matrices
    print("Loading original weight matrices...")
    W1 = load_weight_matrix(f'{FOLDER}weights_hidden.csv')
    W2 = load_weight_matrix(f'{FOLDER}weights_output.csv')
    
    print(f"W1 shape: {W1.shape}")
    print(f"W2 shape: {W2.shape}")
    
    # Generate pruned weights for sparsity levels from 50% to 95% in steps of 5%
    sparsity_levels = range(50, 100, 5)
    
    for sparsity in sparsity_levels:
        print(f"\nProcessing sparsity level: {sparsity}%")
        
        # Prune W1
        W1_pruned = magnitude_based_pruning(W1, sparsity)
        actual_sparsity_W1 = 100.0 * np.sum(W1_pruned == 0) / W1_pruned.size
        print(f"  W1: {actual_sparsity_W1:.2f}% sparse ({np.sum(W1_pruned == 0)} zeros out of {W1_pruned.size} elements)")
        
        # Prune W2
        W2_pruned = magnitude_based_pruning(W2, sparsity)
        actual_sparsity_W2 = 100.0 * np.sum(W2_pruned == 0) / W2_pruned.size
        print(f"  W2: {actual_sparsity_W2:.2f}% sparse ({np.sum(W2_pruned == 0)} zeros out of {W2_pruned.size} elements)")
        
        # Save the pruned weights
        W1_filename = f'{FOLDER}{sparsity}_W1.csv'
        W2_filename = f'{FOLDER}{sparsity}_W2.csv'
        
        save_weight_matrix(W1_pruned, W1_filename)
        save_weight_matrix(W2_pruned, W2_filename)
        
        print(f"  Saved {W1_filename}")
        print(f"  Saved {W2_filename}")
    
    print("\nPruning complete!")

if __name__ == "__main__":
    main()
