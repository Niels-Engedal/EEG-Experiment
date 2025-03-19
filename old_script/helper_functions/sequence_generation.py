"""
Functions for generating stimulus sequences for the MMN oddball experiment.
"""
import random

def generate_block_sequence(standard_type, deviant_types, total_trials, deviant_prob, 
                            trigger_codes, min_standards=2, max_standards=5):
    """
    Generate a stimulus sequence for a single block with specified standard and deviant types.
    
    Parameters:
    - standard_type: Stimulus type that serves as standard ('N', 'N2', or 'Ai')
    - deviant_types: List of stimulus types that serve as deviants (typically just one type)
    - total_trials: Total number of trials in the block
    - deviant_prob: Target probability of deviants
    - min_standards: Minimum number of standards before a deviant
    - max_standards: Maximum number of standards before a deviant
    - trigger_codes: Dictionary mapping stimulus types to trigger codes
    
    Returns:
    - sequence: List of stimulus types ('N', 'N2', 'Ai')
    - trigger_sequence: List of corresponding EEG trigger codes
    """
    # If we have multiple deviant types, we'll use just one for a simpler design
    if len(deviant_types) > 1:
        print("Warning: Multiple deviant types provided, but each block should have only one deviant type.")
        deviant_types = [deviant_types[0]]
    
    deviant_type = deviant_types[0]  # Use the single deviant type
    
    # Initialize sequences
    sequence = []
    trigger_sequence = []
    
    # Start with a few standards to establish standard representation
    initial_standards = random.randint(5, 8)
    for _ in range(initial_standards):
        sequence.append(standard_type)
        trigger_sequence.append(trigger_codes[f'{standard_type}_standard'])
    
    # Generate the rest of the sequence with deviants throughout
    standards_counter = initial_standards
    deviants_inserted = 0
    
    # Set up dynamic max standards based on block length
    base_max_standards = max_standards
    
    # Track positions for analysis
    deviant_positions = []
    
    while len(sequence) < total_trials:
        current_position = len(sequence)
        current_progress = current_position / total_trials
        
        # Dynamically adjust max_standards based on deviation from target probability
        if current_position > 0:
            current_deviant_rate = deviants_inserted / current_position
            
            # Calculate the target number of deviants for the whole sequence
            target_deviants = int(total_trials * deviant_prob)
            
            # Calculate remaining deviants needed and trials left
            remaining_trials = total_trials - current_position
            remaining_target_deviants = target_deviants - deviants_inserted
            
            # Calculate dynamic max_standards
            if remaining_trials > 0:
                # If we're below target rate, decrease max_standards to insert more deviants
                if current_deviant_rate < deviant_prob:
                    # Linear scaling from base_max_standards down to min_standards
                    # as we get further from target
                    deviation_factor = max(0, 1 - (deviant_prob - current_deviant_rate) * 5)
                    dynamic_max = int(base_max_standards * deviation_factor)
                    dynamic_max = max(min_standards + 1, dynamic_max)  # Don't go below min_standards+1
                    
                # If we're above target rate, increase max_standards to slow down deviant insertion
                else:
                    # Linear scaling from base_max_standards up to base_max_standards * 2
                    deviation_factor = min(2, 1 + (current_deviant_rate - deviant_prob) * 5)
                    dynamic_max = int(base_max_standards * deviation_factor)
                    
                # Extra check: if we're getting close to the end with too few deviants
                if remaining_trials <= remaining_target_deviants * 2:
                    dynamic_max = min(dynamic_max, min_standards + 2)  # Force more deviants near end
                    
                # Final bounds check
                dynamic_max = max(min_standards + 1, min(15, dynamic_max))  # Cap at 15 max standards
            else:
                dynamic_max = base_max_standards
        else:
            dynamic_max = base_max_standards
            
        # Force occasional deviants in latter parts to ensure distribution
        force_deviant = False
        if current_progress > 0.5:  # In second half of experiment
            # Force a deviant if we haven't seen one in a while
            last_quarter_position = 0.75 * total_trials
            if current_position > last_quarter_position:
                # In last quarter, be more aggressive
                if standards_counter > base_max_standards * 1.5:
                    force_deviant = True
        
        # Determine if we should insert a deviant
        insert_deviant = False
        
        # First condition: Must have enough standards before considering a deviant
        if standards_counter >= min_standards:
            # Second condition: Always insert deviant if we've hit max consecutive standards
            # or if we need to force one
            if standards_counter >= dynamic_max or force_deviant:
                insert_deviant = True
            # Third condition: Probabilistic insertion with fixed probability
            elif random.random() < deviant_prob:  # Use consistent base probability
                insert_deviant = True
        
        # Insert either a deviant or standard
        if insert_deviant:
            # Insert the deviant
            sequence.append(deviant_type)
            trigger_sequence.append(trigger_codes[f'{deviant_type}_deviant'])
            standards_counter = 0
            deviants_inserted += 1
            deviant_positions.append(current_position)
        else:
            # Insert standard
            sequence.append(standard_type)
            trigger_sequence.append(trigger_codes[f'{standard_type}_standard'])
            standards_counter += 1
    
    # Print statistics about the generated sequence
    print(f"Block with standard '{standard_type}' generated:")
    print(f"Total trials: {len(sequence)}")
    print(f"Number of deviants: {deviants_inserted} ({deviants_inserted/total_trials:.3f})")
    
    # Analyze deviant distribution
    if deviant_positions:
        first_quarter = sum(1 for pos in deviant_positions if pos < total_trials * 0.25)
        second_quarter = sum(1 for pos in deviant_positions if total_trials * 0.25 <= pos < total_trials * 0.5)
        third_quarter = sum(1 for pos in deviant_positions if total_trials * 0.5 <= pos < total_trials * 0.75)
        fourth_quarter = sum(1 for pos in deviant_positions if pos >= total_trials * 0.75)
        
        print(f"Deviant distribution by quarter: {first_quarter}/{second_quarter}/{third_quarter}/{fourth_quarter}")
    
    return sequence, trigger_sequence

def create_experiment_blocks(num_blocks, total_trials_per_block, deviant_probability, trigger_codes, min_standards=2, max_standards=5):
    """
    Create stimulus sequences for all blocks in the experiment.
    Each block has one standard type and one deviant type.
    
    Parameters:
    - num_blocks: Number of blocks to generate
    - total_trials_per_block: Number of trials in each block
    - deviant_probability: Probability of deviant stimuli
    - trigger_codes: Dictionary mapping stimulus types to trigger codes
    - min_standards: Minimum number of standards before a deviant
    - max_standards: Maximum number of standards before a deviant
    
    Returns:
    - blocks: List of block sequences
    - trigger_blocks: List of corresponding trigger sequences
    """
    stimulus_types = ['N', 'N2', 'Ai']
    
    # Generate all possible pairs
    all_pairs = []
    for standard in stimulus_types:
        for deviant in stimulus_types:
            if standard != deviant:
                # Classify the pair as control or experimental
                is_control = (standard == 'N' and deviant == 'N2') or (standard == 'N2' and deviant == 'N')
                all_pairs.append({
                    'standard': standard, 
                    'deviant': deviant, 
                    'is_control': is_control
                })
    
    # Separate control and experimental pairs
    control_pairs = [pair for pair in all_pairs if pair['is_control']]
    experimental_pairs = [pair for pair in all_pairs if not pair['is_control']]
    
    # Randomly choose one control pair
    chosen_pairs = [random.choice(control_pairs)]
    
    # Randomly choose remaining pairs from experimental pairs
    remaining_slots = num_blocks - 1
    if remaining_slots > 0:
        random.shuffle(experimental_pairs)
        chosen_pairs.extend(experimental_pairs[:remaining_slots])
    
    # Shuffle the order of the chosen pairs
    random.shuffle(chosen_pairs)
    
    # Generate sequences for chosen pairs
    blocks = []
    trigger_blocks = []
    for pair in chosen_pairs:
        sequence, triggers = generate_block_sequence(
            pair['standard'], [pair['deviant']], 
            total_trials=total_trials_per_block,
            deviant_prob=deviant_probability,
            trigger_codes=trigger_codes,
            min_standards=min_standards,
            max_standards=max_standards
        )
        blocks.append(sequence)
        trigger_blocks.append(triggers)
        
    return blocks, trigger_blocks