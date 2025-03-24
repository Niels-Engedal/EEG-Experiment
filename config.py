"""Configuration for EEG experiment with trigger codes and parameters"""

# Trigger codes for EEG markers
TRIGGER_CODES = {
    # Standard stimulus codes (100-199)
    'standard_codes': {
        "N1": 110,
        "N2": 120, 
        "Ai1": 130,
        "Ai2": 140
    },
    
    # Transition/deviant codes (200-255)
    'transition_codes': {
        # From N1 (1) to...
        "N1->N2":  212,   # 2(deviant) + 1(from N1) + 2(to N2)
        "N1->Ai1": 213,   # 2(deviant) + 1(from N1) + 3(to Ai1)
        "N1->Ai2": 214,   # 2(deviant) + 1(from N1) + 4(to Ai2)
        
        # From N2 (2) to...
        "N2->N1":  221,   # 2(deviant) + 2(from N2) + 1(to N1)
        "N2->Ai1": 223,   # 2(deviant) + 2(from N2) + 3(to Ai1) 
        "N2->Ai2": 224,   # 2(deviant) + 2(from N2) + 4(to Ai2)
        
        # From Ai1 (3) to...
        "Ai1->N1": 231,   # 2(deviant) + 3(from Ai1) + 1(to N1)
        "Ai1->N2": 232,   # 2(deviant) + 3(from Ai1) + 2(to N2)
        "Ai1->Ai2": 234,  # 2(deviant) + 3(from Ai1) + 4(to Ai2)
        
        # From Ai2 (4) to...
        "Ai2->N1": 241,   # 2(deviant) + 4(from Ai2) + 1(to N1)
        "Ai2->N2": 242,   # 2(deviant) + 4(from Ai2) + 2(to N2)
        "Ai2->Ai1": 243,  # 2(deviant) + 4(from Ai2) + 3(to Ai1)
    },
    
    # Special event codes (0-99)
    'special_codes': {
        'break_start': 10,
        'break_end': 11,
        'block_start': 1,
        'block_end': 2,
        "incorrect_response": 3,
        "correct_reponse": 4
    }
}

# Default experiment parameters
DEFAULT_PARAMS = {
    'stim_types': ['N1', 'N2', 'Ai1', 'Ai2'],
    'block_duration': 300000,  # 5 minutes in ms
    'min_seq_length': 4,
    'max_seq_length': 8,
    'min_jitter': 700,
    'max_jitter': 850,
    'stim_duration': 500,
    'stim_durations': {
        'N1': 519,
        'N2': 559,
        'Ai1': 626,
        'Ai2': 626
    }
}