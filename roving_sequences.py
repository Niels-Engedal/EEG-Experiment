# Imports
import random
from config import TRIGGER_CODES, DEFAULT_PARAMS

def create_roving_block(params):
    """
    Create a complete roving sequence block with the specified parameters.
    
    Parameters:
    params (dict): Configuration parameters including:
        - stim_types: List of possible stimulus types
        - block_duration: Maximum duration of the block in ms
        - min_seq_length/max_seq_length: Range for sequence length
        - min_jitter/max_jitter: Range for inter-stimulus interval jitter in ms
        - stim_durations: Dict mapping stimulus types to their durations in ms
          (fallback to stim_duration if not provided)
        - stim_duration: Default duration of each stimulus in ms
        - start_stim: Initial standard stimulus
        - transition_codes: Dict mapping transition types to EEG trigger codes
    
    Returns:
    dict: Block data containing sequences, timing, and metadata
    """
    block = {
        'sequences': [],
        'total_duration': 0,
        'total_stimuli': 0,
        'transitions': {}  # Track different transition types
    }

    # Extract parameters
    stim_types = params['stim_types']
    block_duration = params['block_duration']
    min_seq_length = params.get('min_seq_length', 4)
    max_seq_length = params.get('max_seq_length', 8)
    min_jitter = params.get('min_jitter', 600)
    max_jitter = params.get('max_jitter', 750)
    default_stim_duration = params.get('stim_duration', 500)
    stim_durations = params.get('stim_durations', {})  # Map of stim type to duration
    start_stim = params.get('start_stim', 'N1')
    standard_codes = TRIGGER_CODES['standard_codes']
    transition_codes = TRIGGER_CODES['transition_codes']

    # Initialize with a selected start stimulus type
    current_standard = start_stim
    
    # Create sequences until we reach the time limit
    while block['total_duration'] < block_duration:
        # Determine sequence length
        seq_length = random.randint(min_seq_length, max_seq_length)

        # Select next stimulus (deviant) - different from current standard
        available_deviants = [s for s in stim_types if s != current_standard]
        next_deviant = random.choice(available_deviants)
        
        # Track the transition type
        transition_type = f"{current_standard}->{next_deviant}"
        block['transitions'][transition_type] = block['transitions'].get(transition_type, 0) + 1
        
        # Create the sequence
        sequence = {
            'stimuli': [current_standard] * (seq_length - 1) + [next_deviant],
            'jitters': [random.randint(min_jitter, max_jitter) for _ in range(seq_length)],
            'standard': current_standard,
            'deviant': next_deviant,
            'transition_type': transition_type,
            'trigger_code': transition_codes.get(transition_type)
        }
        
        # Calculate sequence duration with variable stimulus durations
        seq_duration = sum(sequence['jitters'])
        for stim in sequence['stimuli']:
            # Use specific duration if available, otherwise default
            stim_duration = stim_durations.get(stim, default_stim_duration)
            seq_duration += stim_duration
            
        sequence['duration'] = seq_duration
        
        # Add to block if we don't exceed the time limit
        if block['total_duration'] + seq_duration <= block_duration:
            block['sequences'].append(sequence)
            block['total_duration'] += seq_duration
            block['total_stimuli'] += seq_length
            
            # Roving paradigm: deviant becomes the new standard
            current_standard = next_deviant
        else:
            # We would exceed the time limit, so stop adding sequences
            break
    
    return block


def get_full_stimulus_list(block, params=None):
    """
    Flatten the block structure into a sequential list of stimuli with timing.
    Useful for experiment presentation.
    
    Parameters:
    block (dict): Block data from create_roving_block
    params (dict, optional): Parameter dictionary containing trigger codes
    
    Returns:
    list: Sequential stimulus presentations with timing information
    """
    stimulus_list = []
    cumulative_time = 0
    stim_durations = block.get('stim_durations', {})
    default_duration = block.get('stim_duration', 100)
    
    # Get standard codes from imported TRIGGER_CODES
    standard_codes = TRIGGER_CODES['standard_codes']
    
    for sequence in block['sequences']:
        for i, (stim, jitter) in enumerate(zip(sequence['stimuli'], sequence['jitters'])):
            is_deviant = (i == len(sequence['stimuli']) - 1)
            
            stimulus = {
                'type': stim,
                'onset': cumulative_time,
                'jitter': jitter,
                'is_deviant': is_deviant,
                'duration': stim_durations.get(stim, default_duration)
            }
            
            # Add transition info for deviants
            if is_deviant:
                stimulus['transition_type'] = sequence.get('transition_type')
                stimulus['trigger_code'] = sequence.get('trigger_code')
            else:
                # For standards, use the standard codes
                stimulus['trigger_code'] = standard_codes.get(stim, 100)
            
            stimulus_list.append(stimulus)
            cumulative_time += jitter + stimulus['duration']
    
    return stimulus_list

# Add these functions after your existing code

def analyze_block_balance(block):
    """
    Analyze a block for stimulus type and transition balance.
    
    Parameters:
    block (dict): Block data from create_roving_block
    
    Returns:
    dict: Analysis metrics including stimulus and transition balance scores
    """
    # Count stimulus occurrences
    stim_counts = {}
    total_stimuli = 0
    
    # Count transition occurrences (already in the block data)
    transition_counts = block['transitions']
    
    # Get flat stimulus list
    for sequence in block['sequences']:
        for stim in sequence['stimuli']:
            stim_counts[stim] = stim_counts.get(stim, 0) + 1
            total_stimuli += 1
    
    # Calculate stimulus balance score (lower is better)
    # Standard deviation as a percentage of the mean
    if stim_counts:
        mean_stim_count = total_stimuli / len(stim_counts)
        stim_variance = sum((count - mean_stim_count)**2 for count in stim_counts.values()) / len(stim_counts)
        stim_balance_score = (stim_variance**0.5) / mean_stim_count
    else:
        stim_balance_score = float('inf')
    
    # Calculate transition balance score
    if transition_counts:
        mean_transition_count = sum(transition_counts.values()) / len(transition_counts)
        transition_variance = sum((count - mean_transition_count)**2 for count in transition_counts.values()) / len(transition_counts)
        transition_balance_score = (transition_variance**0.5) / mean_transition_count
    else:
        transition_balance_score = float('inf')
    
    # Combined score (weighted average)
    combined_score = (stim_balance_score + 2 * transition_balance_score) / 3
    
    return {
        'stimulus_counts': stim_counts,
        'transition_counts': transition_counts,
        'stimulus_balance_score': stim_balance_score,
        'transition_balance_score': transition_balance_score,
        'combined_score': combined_score
    }

def generate_optimized_blocks(params, num_candidates=50, num_to_select=3):
    """
    Generate multiple candidate blocks and select the most balanced ones.
    
    Parameters:
    params (dict): Configuration parameters for block generation
    num_candidates (int): Number of candidate blocks to generate
    num_to_select (int): Number of best blocks to select
    
    Returns:
    list: Selected blocks with the best balance metrics
    """
    # Generate candidate blocks
    candidate_blocks = []
    candidate_metrics = []
    
    print(f"Generating {num_candidates} candidate blocks...")
    for i in range(num_candidates):
        # Generate a block with 'N1' as the start stimulus
        block_params = params.copy()
        block_params['start_stim'] = 'N1'
        
        # Create the block
        block = create_roving_block(block_params)
        
        # Analyze balance
        metrics = analyze_block_balance(block)
        
        # Store block and metrics
        candidate_blocks.append(block)
        candidate_metrics.append(metrics)
        
        # Optional progress update
        if (i+1) % 10 == 0:
            print(f"Generated {i+1}/{num_candidates} blocks")
    
    # Sort blocks by combined balance score (lower is better)
    sorted_indices = sorted(range(len(candidate_metrics)), 
                            key=lambda i: candidate_metrics[i]['combined_score'])
    
    # Select the best blocks
    selected_blocks = [candidate_blocks[i] for i in sorted_indices[:num_to_select]]
    selected_metrics = [candidate_metrics[i] for i in sorted_indices[:num_to_select]]
    
    # Print metrics of selected blocks
    print("\nSelected blocks metrics:")
    for i, metrics in enumerate(selected_metrics):
        print(f"\nBlock {i+1}:")
        print(f"  Stimulus balance score: {metrics['stimulus_balance_score']:.4f}")
        print(f"  Transition balance score: {metrics['transition_balance_score']:.4f}")
        print(f"  Combined score: {metrics['combined_score']:.4f}")
        print("  Stimulus counts:", dict(sorted(metrics['stimulus_counts'].items())))
        print("  Transition counts:", dict(sorted(metrics['transition_counts'].items())))
    
    return selected_blocks

#==============================================================================

# Example usage
if __name__ == "__main__":
    print("\nGenerating and selecting optimized blocks...")
    optimized_blocks = generate_optimized_blocks(
        DEFAULT_PARAMS,
        num_candidates=5000,  # Generate 50 candidate blocks
        num_to_select=3     # Select the 3 most balanced blocks
    )
    
    # Convert the optimized blocks to stimulus lists for experiment
    optimized_stim_lists = [get_full_stimulus_list(block) for block in optimized_blocks]
    
    # Print summary statistics
    print(f"\nGenerated {len(optimized_blocks)} optimized blocks")
    for i, block in enumerate(optimized_blocks):
        print(f"Block {i+1}: {block['total_stimuli']} stimuli, duration: {block['total_duration']/1000:.1f}s")
    
    # Fix the print statement that was causing an error
    print("\nStimulus types in first block:", [stim['type'] for stim in optimized_stim_lists[0][:10]], "...")
