"""
Script to generate optimized MMN oddball sequences with precise deviant probabilities.
Run this script separately to generate sequences that can be loaded by the main experiment.
"""
import random
import json
import os
import numpy as np
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def generate_precise_sequence(standard_type, deviant_type, total_trials, target_deviant_prob,
                            min_standards=4, max_standards=9, min_standards_floor=4):
    """
    Generate a sequence with exact deviant probability while maintaining constraints.
    Uses quarter-based allocation to ensure even distribution throughout the sequence.
    
    Parameters:
    - standard_type: Type of stimulus for standards
    - deviant_type: Type of stimulus for deviants
    - total_trials: Total number of trials in sequence
    - target_deviant_prob: Target probability of deviants (e.g., 0.1 for 10%)
    - min_standards: Minimum consecutive standards before a deviant can appear
    - max_standards: Maximum consecutive standards before forcing a deviant
    - min_standards_floor: Absolute minimum value allowed for min_standards
    """
    # Enforce minimum standards floor
    min_standards = max(min_standards, min_standards_floor)
    
    # Calculate exact number of deviants needed
    exact_deviants = round(total_trials * target_deviant_prob)
    exact_standards = total_trials - exact_deviants
    
    # Verify that constraints can be satisfied
    minimum_possible_deviants = max(1, total_trials // (max_standards + 1))
    if exact_deviants < minimum_possible_deviants:
        print(f"WARNING: Target probability {target_deviant_prob} requires {exact_deviants} deviants, "
              f"but constraints force at least {minimum_possible_deviants} deviants")
        print(f"Adjusting max_standards to {int(1/target_deviant_prob) - 1} to allow target probability")
        max_standards = int(1/target_deviant_prob) - 1
    
    # QUARTER-BASED APPROACH: First allocate deviants to each quarter
    quarter_size = total_trials // 4
    last_quarter_size = total_trials - (quarter_size * 3)  # Account for rounding
    
    # Allocate deviants per quarter (approximately equal)
    deviants_per_quarter = [exact_deviants // 4] * 4
    
    # Distribute any remaining deviants
    remaining = exact_deviants - sum(deviants_per_quarter)
    for i in range(remaining):
        deviants_per_quarter[i] += 1
    
    # Create a template sequence with all standards
    sequence = [standard_type] * total_trials
    
    # For each quarter, place deviants with constraints
    for quarter in range(4):
        quarter_start = quarter * quarter_size
        quarter_end = quarter_start + (last_quarter_size if quarter == 3 else quarter_size)
        quarter_length = quarter_end - quarter_start
        
        # Place deviants in this quarter
        deviants_to_place = deviants_per_quarter[quarter]
        standards_runs = []
        
        # Split the quarter's standards into (deviants_to_place + 1) sections
        if deviants_to_place > 0:
            avg_standards_per_section = (quarter_length - deviants_to_place) / (deviants_to_place + 1)
            
            # Create varied lengths of standard runs with some randomization
            remaining_standards = quarter_length - deviants_to_place
            for i in range(deviants_to_place + 1):
                if i == deviants_to_place:  # Last section
                    standards_runs.append(remaining_standards)
                else:
                    # Random variation around the average
                    variation = random.uniform(0.8, 1.2)
                    standards_in_this_run = int(avg_standards_per_section * variation)
                    
                    # Ensure we respect min_standards and don't use too many
                    standards_in_this_run = max(min_standards, 
                                                min(standards_in_this_run, 
                                                   remaining_standards - (deviants_to_place - i) * min_standards,
                                                   max_standards))
                    
                    standards_runs.append(standards_in_this_run)
                    remaining_standards -= standards_in_this_run
        
            # Now place deviants in the quarter according to our calculated runs
            position = quarter_start
            for std_run in standards_runs[:-1]:  # All but the last run
                position += std_run
                if position < quarter_end:  # Safety check
                    sequence[position] = deviant_type
                    position += 1
    
    # Verify the number of deviants and their distribution
    actual_deviants = sum(1 for s in sequence if s == deviant_type)
    if actual_deviants != exact_deviants:
        print(f"WARNING: Generated {actual_deviants} deviants instead of target {exact_deviants}")
    
    # Final check for consecutive standards constraint
    for i in range(total_trials - max_standards):
        window = sequence[i:i+max_standards+1]
        if all(s == standard_type for s in window):
            # Found more consecutive standards than allowed, insert a deviant
            sequence[i + max_standards] = deviant_type
    
    # Verify constraints are met
    consecutive_standards = 0
    max_consecutive = 0
    for s in sequence:
        if s == standard_type:
            consecutive_standards += 1
            max_consecutive = max(max_consecutive, consecutive_standards)
        else:
            consecutive_standards = 0
    
    # Calculate actual probability achieved
    actual_deviants = sum(1 for s in sequence if s == deviant_type)
    actual_prob = actual_deviants / total_trials
    
    # Calculate quarters for debugging
    quarters = [
        sum(1 for i, s in enumerate(sequence) if s == deviant_type and i < total_trials * 0.25),
        sum(1 for i, s in enumerate(sequence) if s == deviant_type and total_trials * 0.25 <= i < total_trials * 0.5),
        sum(1 for i, s in enumerate(sequence) if s == deviant_type and total_trials * 0.5 <= i < total_trials * 0.75),
        sum(1 for i, s in enumerate(sequence) if s == deviant_type and i >= total_trials * 0.75)
    ]
    
    return sequence, actual_prob

def find_best_sequence(standard_type, deviant_type, total_trials, target_deviant_prob,
                     min_standards=4, max_standards=9, num_candidates=20, 
                     min_standards_floor=4, max_standards_cap=None):
    """
    Generate multiple candidate sequences and select the one with best distribution.
    
    Parameters:
    - min_standards_floor: Absolute minimum value allowed for min_standards
    - max_standards_cap: Maximum value allowed for max_standards (None for no cap)
    """

    # Initialize best sequence variables
    best_sequence = None
    best_score = float('-inf')  # Start with negative infinity so any real score is better
    best_actual_prob = 0
    
    # Enforce minimum standards floor
    min_standards = max(min_standards, min_standards_floor)
    
    # Ensure max_standards is compatible with target probability
    adjusted_max = max(max_standards, int(1/target_deviant_prob) - 1)
    
    # Apply cap if specified
    if max_standards_cap is not None:
        adjusted_max = min(adjusted_max, max_standards_cap)
        
    if adjusted_max != max_standards:
        print(f"Adjusting max_standards from {max_standards} to {adjusted_max} to achieve target probability")
        max_standards = adjusted_max
    
    # Calculate target number of deviants per quarter (approximately)
    exact_deviants = round(total_trials * target_deviant_prob)
    target_per_quarter = exact_deviants / 4
    min_per_quarter = max(1, int(target_per_quarter * 0.5))  # At least 50% of expected deviants per quarter
    
    print(f"Generating {num_candidates} candidate sequences...")
    print(f"Target: ~{int(target_per_quarter)} deviants per quarter, minimum {min_per_quarter} per quarter")
    
    valid_sequences = 0
    
    for i in range(num_candidates):
        if i % 5 == 0 and i > 0:
            print(f"  Progress: {i}/{num_candidates} candidates tested, {valid_sequences} valid distributions")
        
        # Generate a candidate sequence with exact probability
        sequence, actual_prob = generate_precise_sequence(
            standard_type, deviant_type, total_trials, target_deviant_prob,
            min_standards, max_standards
        )
        
        # Score the sequence based on distribution quality
        quarters = [
            sum(1 for i, s in enumerate(sequence) if s == deviant_type and i < total_trials * 0.25),
            sum(1 for i, s in enumerate(sequence) if s == deviant_type and total_trials * 0.25 <= i < total_trials * 0.5),
            sum(1 for i, s in enumerate(sequence) if s == deviant_type and total_trials * 0.5 <= i < total_trials * 0.75),
            sum(1 for i, s in enumerate(sequence) if s == deviant_type and i >= total_trials * 0.75)
        ]
        
        # STRICT REQUIREMENT: Reject sequences with too few deviants in any quarter
        if min(quarters) < min_per_quarter:
            continue  # Skip this sequence entirely
        
        valid_sequences += 1
        
        # Calculate gap sizes between deviants
        deviant_positions = [i for i, s in enumerate(sequence) if s == deviant_type]
        gaps = [deviant_positions[i] - deviant_positions[i-1] for i in range(1, len(deviant_positions))]
        
        # Score based on even distribution (heavily weighted) and variability
        distribution_evenness = -np.std(quarters) * 10  # Much stronger weight for evenness
        
        # Additional penalty for any quarter that deviates significantly from target
        quarter_penalty = sum(abs(q - target_per_quarter) for q in quarters) * 0.5
        
        # Small reward for natural variability in gaps
        gap_score = np.std(gaps) * 0.2 if gaps else 0
        
        # Combined score - heavily favor even distribution
        score = distribution_evenness - quarter_penalty + gap_score
        
        # Update best if this is better
        if score > best_score:
            best_score = score
            best_sequence = sequence
            best_actual_prob = actual_prob
    
    # If we couldn't find any valid sequences, relax constraints and try again
    if best_sequence is None:
        print("No sequences met the minimum distribution requirements. Relaxing constraints...")
        return find_best_sequence(standard_type, deviant_type, total_trials, target_deviant_prob, 
                                min_standards, max_standards, num_candidates * 2)
    
    print(f"Best sequence found with deviant probability={best_actual_prob:.6f} (target={target_deviant_prob})")
    
    # Analyze deviant distribution
    deviant_positions = [i for i, s in enumerate(best_sequence) if s == deviant_type]
    first_quarter = sum(1 for pos in deviant_positions if pos < total_trials * 0.25)
    second_quarter = sum(1 for pos in deviant_positions if total_trials * 0.25 <= pos < total_trials * 0.5)
    third_quarter = sum(1 for pos in deviant_positions if total_trials * 0.5 <= pos < total_trials * 0.75)
    fourth_quarter = sum(1 for pos in deviant_positions if pos >= total_trials * 0.75)
    
    print(f"Deviant distribution by quarter: {first_quarter}/{second_quarter}/{third_quarter}/{fourth_quarter}")
    
    return best_sequence, best_actual_prob, min_standards, max_standards

def generate_experiment_blocks(num_blocks, total_trials_per_block, target_deviant_prob,
                             min_standards=4, max_standards=9, num_candidates=20,
                             min_standards_floor=4, max_standards_cap=None):
    """
    Generate optimized blocks for an entire experiment.
    
    Parameters:
    - min_standards_floor: Absolute minimum value allowed for min_standards
    - max_standards_cap: Maximum value allowed for max_standards (None for no cap)
    """
    stimulus_types = ['N', 'N2', 'Ai']
    
    # Generate all possible pairs and classify as control or experimental
    all_pairs = []
    for standard in stimulus_types:
        for deviant in stimulus_types:
            if standard != deviant:
                is_control = (standard == 'N' and deviant == 'N2') or (standard == 'N2' and deviant == 'N')
                all_pairs.append({
                    'standard': standard,
                    'deviant': deviant,
                    'is_control': is_control
                })
    
    # Separate control and experimental pairs
    control_pairs = [pair for pair in all_pairs if pair['is_control']]
    experimental_pairs = [pair for pair in all_pairs if not pair['is_control']]
    
    # Choose blocks according to requirements
    chosen_pairs = [random.choice(control_pairs)]
    remaining_slots = num_blocks - 1
    
    if remaining_slots > 0:
        random.shuffle(experimental_pairs)
        chosen_pairs.extend(experimental_pairs[:remaining_slots])
    
    random.shuffle(chosen_pairs)
    
    # Generate optimized sequences
    blocks = []
    block_info = []
    used_min_standards = min_standards
    used_max_standards = max_standards
    
    for i, pair in enumerate(chosen_pairs):
        print(f"\nGenerating block {i+1}/{num_blocks} (standard: {pair['standard']}, deviant: {pair['deviant']})...")
        
        sequence, actual_prob, block_min, block_max = find_best_sequence(
            standard_type=pair['standard'],
            deviant_type=pair['deviant'],
            total_trials=total_trials_per_block,
            target_deviant_prob=target_deviant_prob,
            min_standards=min_standards,
            max_standards=max_standards,
            num_candidates=num_candidates,
            min_standards_floor=min_standards_floor,
            max_standards_cap=max_standards_cap
    )
    
        # Update the standards used
        used_min_standards = block_min
        used_max_standards = block_max
        
        blocks.append(sequence)
        
        # Store metadata about this block
        block_info.append({
            'block_num': i + 1,
            'standard': pair['standard'],
            'deviant': pair['deviant'],
            'is_control': pair['is_control'],
            'total_trials': total_trials_per_block,
            'deviant_count': sum(1 for s in sequence if s == pair['deviant']),
            'deviant_probability': actual_prob,
            'min_standards': block_min,
            'max_standards': block_max
        })
    
    return blocks, block_info, used_min_standards, used_max_standards

def save_blocks_to_file(blocks, block_info, output_dir="optimized_sequences"):
    """Save the optimized blocks to a JSON file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimized_blocks_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    data = {
        "blocks": blocks,
        "block_info": block_info,
        "generation_time": timestamp
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved optimized blocks to {filepath}")
    return filepath

def analyze_sequence_distribution(sequence, deviant_type, total_trials):
    """Analyze the distribution of deviants in a sequence."""
    deviant_positions = [i for i, s in enumerate(sequence) if s == deviant_type]
    
    # Calculate distribution by quarter
    quarters = [
        sum(1 for pos in deviant_positions if pos < total_trials * 0.25),
        sum(1 for pos in deviant_positions if total_trials * 0.25 <= pos < total_trials * 0.5),
        sum(1 for pos in deviant_positions if total_trials * 0.5 <= pos < total_trials * 0.75),
        sum(1 for pos in deviant_positions if pos >= total_trials * 0.75)
    ]
    
    # Calculate runs of standards
    standards_runs = []
    current_run = 0
    for stim in sequence:
        if stim == deviant_type:
            if current_run > 0:
                standards_runs.append(current_run)
            current_run = 0
        else:
            current_run += 1
    
    if current_run > 0:
        standards_runs.append(current_run)
    
    return {
        'quarter_distribution': quarters,
        'avg_standards_between_deviants': sum(standards_runs) / len(standards_runs) if standards_runs else 0,
        'max_standards_run': max(standards_runs) if standards_runs else 0,
        'min_standards_run': min(standards_runs) if standards_runs else 0
    }

def visualize_blocks(blocks, block_info, output_dir="sequence_visualizations"):
    """
    Create visualizations of the generated block sequences.
    
    Parameters:
    - blocks: List of stimulus sequences
    - block_info: Metadata about each block
    - output_dir: Directory to save visualization images
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create visualizations for each block
    for i, (sequence, info) in enumerate(zip(blocks, block_info)):
        standard_type = info['standard']
        deviant_type = info['deviant']
        total_trials = info['total_trials']
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={'height_ratios': [1, 3]})
        
        # Plot 1: Distribution across quarters
        quarters = [0, 1, 2, 3]
        quarter_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        quarter_sizes = [total_trials//4, total_trials//4, total_trials//4, total_trials - 3*(total_trials//4)]
        
        deviant_positions = [i for i, s in enumerate(sequence) if s == deviant_type]
        quarter_counts = [
            sum(1 for pos in deviant_positions if pos < total_trials * 0.25),
            sum(1 for pos in deviant_positions if total_trials * 0.25 <= pos < total_trials * 0.5),
            sum(1 for pos in deviant_positions if total_trials * 0.5 <= pos < total_trials * 0.75),
            sum(1 for pos in deviant_positions if pos >= total_trials * 0.75)
        ]
        
        # Normalize by quarter size
        quarter_percentages = [count / size * 100 for count, size in zip(quarter_counts, quarter_sizes)]
        
        ax1.bar(quarter_labels, quarter_percentages, color='blue', alpha=0.7)
        ax1.set_ylabel('Deviant %')
        ax1.set_title(f'Block {i+1}: {standard_type} (standard) vs {deviant_type} (deviant), '
                      f'p={info["deviant_probability"]:.3f}')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Plot 2: Sequence visualization
        colors = []
        for s in sequence:
            if s == deviant_type:
                colors.append('red')
            else:
                colors.append('blue')
        
        # Create a sequence representation
        ax2.scatter(range(len(sequence)), [1] * len(sequence), c=colors, marker='|', s=100)
        
        # Add vertical lines to separate quarters
        for q in range(1, 4):
            ax2.axvline(x=q * total_trials // 4, color='black', linestyle='--', alpha=0.5)
        
        # Set axis properties
        ax2.set_xlim(-1, len(sequence))
        ax2.set_yticks([])
        ax2.set_xlabel('Trial position')
        
        # Create legend
        standard_patch = mpatches.Patch(color='blue', label=f'Standard ({standard_type})')
        deviant_patch = mpatches.Patch(color='red', label=f'Deviant ({deviant_type})')
        ax2.legend(handles=[standard_patch, deviant_patch], loc='upper right')
        
        # Add statistics as text
        stats_text = (f"Total trials: {total_trials}\n"
                     f"Deviant count: {info['deviant_count']} ({info['deviant_probability']:.3f})\n"
                     f"Deviants by quarter: {'/'.join(str(q) for q in quarter_counts)}\n"
                     f"Min/Max standards: {info['min_standards']}/{info['max_standards']}")
        ax2.text(0.02, 0.02, stats_text, transform=ax2.transAxes, 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        filename = f"block_{i+1}_{standard_type}_vs_{deviant_type}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150)
        print(f"Saved visualization to {filepath}")
        
        plt.close(fig)
    
    # Create a combined visualization showing all blocks
    plt.figure(figsize=(15, num_blocks * 2))
    for i, (sequence, info) in enumerate(zip(blocks, block_info)):
        ax = plt.subplot(num_blocks, 1, i+1)
        
        # Plot sequence
        colors = ['red' if s == info['deviant'] else 'blue' for s in sequence]
        ax.scatter(range(len(sequence)), [1] * len(sequence), c=colors, marker='|', s=50)
        
        # Add quarter markers
        for q in range(1, 4):
            ax.axvline(x=q * total_trials // 4, color='black', linestyle='--', alpha=0.3)
        
        ax.set_yticks([])
        ax.set_title(f"Block {i+1}: {info['standard']} (std) vs {info['deviant']} (dev), p={info['deviant_probability']:.3f}")
        
        if i == num_blocks - 1:
            ax.set_xlabel('Trial position')
    
    plt.tight_layout()
    combined_filepath = os.path.join(output_dir, "all_blocks_overview.png")
    plt.savefig(combined_filepath, dpi=150)
    print(f"Saved combined visualization to {combined_filepath}")
    plt.close()
    
    return output_dir

if __name__ == "__main__":
    print("MMN Sequence Optimizer")
    print("======================")
    
    # Get parameters with defaults
    num_blocks = int(input("Number of blocks [3]: ") or 3)
    trials_per_block = int(input("Trials per block [320]: ") or 320)
    target_prob = float(input("Target deviant probability [0.1]: ") or 0.1)
    candidates = int(input("Number of candidate sequences to try [20]: ") or 20)
    
    # Calculate recommended max_standards for target probability
    recommended_max = int(1/target_prob) - 1
    
    # Set floor for min_standards
    min_standards_floor = int(input(f"Absolute minimum for min_standards [4]: ") or 4)
    min_standards = int(input(f"Minimum standards before deviant [{min_standards_floor}]: ") or min_standards_floor)
    min_standards = max(min_standards, min_standards_floor)  # Enforce floor
    
    # Set cap for max_standards
    max_standards_cap = int(input(f"Maximum cap for max_standards (empty for no cap): ") or 0)
    max_standards_cap = max_standards_cap if max_standards_cap > 0 else None
    
    # Get user input for max_standards with the cap applied
    cap_info = f" (max {max_standards_cap})" if max_standards_cap is not None else ""
    max_standards = int(input(f"Maximum standards before deviant [{recommended_max}]{cap_info}: ") or recommended_max)
    
    # Apply cap if specified
    if max_standards_cap is not None:
        max_standards = min(max_standards, max_standards_cap)
    
    random_seed = input("Random seed (empty for random): ")
    if random_seed:
        random.seed(int(random_seed))
        print(f"Using random seed: {random_seed}")
    else:
        seed = random.randint(1, 10000)
        random.seed(seed)
        print(f"Using random seed: {seed}")
    
    # Generate blocks
    print(f"\nGenerating {num_blocks} blocks with {trials_per_block} trials each (target p={target_prob})")
    blocks, info, used_min_standards, used_max_standards = generate_experiment_blocks(
        num_blocks=num_blocks,
        total_trials_per_block=trials_per_block,
        target_deviant_prob=target_prob,
        min_standards=min_standards,
        max_standards=max_standards,
        num_candidates=candidates,
        min_standards_floor=min_standards_floor,
        max_standards_cap=max_standards_cap
    )
    
    # Save to file
    output_file = save_blocks_to_file(blocks, info)
    
    # Generate visualizations
    print("\nCreating visualizations...")
    visualization_dir = visualize_blocks(blocks, info)
    
    # Print summary
    print("\nSummary:")
    print(f"Generated with min_standards={used_min_standards}, max_standards={used_max_standards}")
    
    for i, block_data in enumerate(info):
        print(f"Block {i+1}: Standard={block_data['standard']}, "
              f"Deviant={block_data['deviant']}, "
              f"Probability={block_data['deviant_probability']:.6f}")
        
        # Analyze distribution
        analysis = analyze_sequence_distribution(
            blocks[i], 
            block_data['deviant'], 
            block_data['total_trials']
        )
        
        print(f"  Distribution by quarter: {'/'.join(str(q) for q in analysis['quarter_distribution'])}")
        print(f"  Avg standards between deviants: {analysis['avg_standards_between_deviants']:.1f}")
        print(f"  Min/Max standards run: {analysis['min_standards_run']}/{analysis['max_standards_run']}")
    
    print("\nDone! These optimized sequences can be loaded by the main experiment script.")
    print(f"Visualizations saved to: {visualization_dir}")
    print(f"Sequence data saved to: {output_file}")