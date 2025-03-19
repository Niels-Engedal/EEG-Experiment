"""
Timing calculation functions for the MMN oddball experiment.
"""

def calculate_experiment_parameters(max_duration_minutes=15, 
                                   num_blocks=3,
                                   block_break_duration_s=30,
                                   stimulus_duration_ms=300,
                                   isi_ms=500,
                                   deviant_delay_ms=0,
                                   deviant_probability=0.1,
                                   setup_time_s=60):  # Time for instructions, etc.
    """
    Calculate optimal experiment parameters to fit within the time limit.
    
    Parameters:
    - max_duration_minutes: Maximum experiment duration in minutes
    - num_blocks: Number of blocks
    - block_break_duration_s: Duration of breaks between blocks in seconds
    - stimulus_duration_ms: Duration of each stimulus in milliseconds
    - isi_ms: Inter-stimulus interval in milliseconds
    - deviant_delay_ms: Additional delay after deviants in milliseconds
    - deviant_probability: Probability of deviants (0-1)
    - setup_time_s: Time for instructions, setup, etc. in seconds
    
    Returns:
    - dict: Dictionary with calculated parameters
    """
    # Convert max duration to seconds
    max_duration_s = max_duration_minutes * 60
    
    # Calculate time taken by breaks between blocks
    break_time_s = (num_blocks - 1) * block_break_duration_s
    
    # Calculate available time for trials
    available_trial_time_s = max_duration_s - break_time_s - setup_time_s
    
    # Calculate average time per trial
    trial_time_ms = stimulus_duration_ms + isi_ms
    
    # Add average additional time for deviant delay (weighted by probability)
    if deviant_delay_ms > 0:
        trial_time_ms += deviant_delay_ms * deviant_probability
    
    trial_time_s = trial_time_ms / 1000
    
    # Calculate maximum trials per block
    max_trials_per_block = int(available_trial_time_s / (trial_time_s * num_blocks))
    
    # Make it divisible by 10 for easier math (round down)
    max_trials_per_block = (max_trials_per_block // 10) * 10
    
    # Calculate estimated actual duration
    actual_trial_time_s = max_trials_per_block * num_blocks * trial_time_s
    actual_duration_s = actual_trial_time_s + break_time_s + setup_time_s
    actual_duration_min = actual_duration_s / 60
    
    return {
        'max_trials_per_block': max_trials_per_block,
        'estimated_duration_minutes': actual_duration_min,
        'estimated_duration_seconds': actual_duration_s,
        'trials_per_minute': round(60 / trial_time_s, 1)
    }