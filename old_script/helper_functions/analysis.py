"""
Analysis functions for the MMN oddball experiment.
"""
import os
from psychopy import sound
import matplotlib.pyplot as plt

def check_stimuli_duration(stimuli_paths):
    """
    Check that all stimuli have similar durations and warn if not.
    
    Parameters:
    - stimuli_paths: Dictionary mapping stimulus types to file paths
    
    Returns:
    - avg_duration: Average duration of stimuli in seconds
    - warnings: List of warning messages
    """
    durations = []
    warnings = []
    
    # Check each stimulus file
    for stim_type, filepath in stimuli_paths.items():
        if not os.path.exists(filepath):
            warnings.append(f"Error: Stimulus file not found: {filepath}")
            continue
        
        try:
            # Load audio file to check duration
            audio = sound.Sound(filepath)
            duration = audio.getDuration()
            durations.append(duration)
            print(f"Stimulus {stim_type} duration: {duration:.3f} seconds")
        except Exception as e:
            warnings.append(f"Error checking duration of {stim_type}: {e}")
    
    # Calculate average and check for inconsistencies
    if durations:
        avg_duration = sum(durations) / len(durations)
        max_diff = max([abs(d - avg_duration) for d in durations])
        
        if max_diff > 0.05:  # If difference is more than 50ms
            warnings.append(f"Warning: Stimulus durations vary by {max_diff:.3f} seconds, which may affect ERP latency!")
    else:
        avg_duration = 0.3  # Default assumption
        warnings.append("Could not check stimulus durations - using default 0.3s")
    
    # Print any warnings
    for warning in warnings:
        print(warning)
    
    return avg_duration, warnings

def analyze_sequences(blocks, random_seed_used):
    """
    Analyze the generated sequences to verify they meet requirements.
    
    Parameters:
    - blocks: List of block sequences
    - random_seed_used: Random seed used for sequence generation
    
    Returns:
    - stats: Dictionary with sequence statistics
    """
    stats = {
        'total_trials': sum(len(block) for block in blocks),
        'random_seed_used': random_seed_used,  # Include the seed in stats
        'blocks': []
    }
    
    for i, block in enumerate(blocks):
        standard = max(set(block), key=block.count)
        deviants = [s for s in set(block) if s != standard]
        
        # Count deviants and calculate probability
        deviant_count = sum(1 for s in block if s != standard)
        deviant_prob = deviant_count / len(block)
        
        # Check for consecutive deviants
        consecutive_deviants = any(block[j] != standard and block[j+1] != standard 
                                  for j in range(len(block)-1))
        
        # Count standard runs before deviants
        standard_runs = []
        run_length = 0
        
        for j in range(len(block)):
            if block[j] == standard:
                run_length += 1
            else:  # Deviant
                if run_length > 0:
                    standard_runs.append(run_length)
                run_length = 0
        
        # Compute statistics about standard runs
        if standard_runs:
            min_run = min(standard_runs)
            max_run = max(standard_runs)
            avg_run = sum(standard_runs) / len(standard_runs)
        else:
            min_run = max_run = avg_run = 0
        
        block_stats = {
            'block_num': i + 1,
            'standard': standard,
            'deviants': deviants,
            'deviant_count': deviant_count,
            'deviant_probability': deviant_prob,
            'consecutive_deviants': consecutive_deviants,
            'standard_runs': {
                'min': min_run,
                'max': max_run,
                'avg': avg_run
            }
        }
        
        stats['blocks'].append(block_stats)
    
    return stats

def visualize_sequence(block, block_num, filename_prefix):
    """
    Create a visual representation of a sequence to verify its structure.
    
    Parameters:
    - block: Sequence of stimuli
    - block_num: Block number for the plot title
    - filename_prefix: Prefix for the output filename
    
    Returns:
    - plot_filename: Path to the saved plot
    """
    standard = max(set(block), key=block.count)
    
    # Map stimulus types to numeric values for plotting
    stimulus_map = {'N': 1, 'N2': 2, 'Ai': 3}
    numeric_sequence = [stimulus_map[s] for s in block]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(numeric_sequence, 'o-', markersize=3)
    plt.xlabel('Trial Number')
    plt.ylabel('Stimulus Type')
    plt.yticks([1, 2, 3], ['N', 'N2', 'Ai'])
    plt.title(f'Block {block_num} Sequence (Standard: {standard})')
    plt.grid(True, alpha=0.3)
    
    # Highlight deviants
    deviant_indices = [i for i, s in enumerate(block) if s != standard]
    plt.plot(deviant_indices, [stimulus_map[block[i]] for i in deviant_indices], 
             'ro', markersize=6, label='Deviants')
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"{filename_prefix}_block{block_num}_sequence.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved sequence visualization to {plot_filename}")
    
    return plot_filename