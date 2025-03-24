"""
Roving MMN EEG Experiment

This script runs a roving MMN paradigm for EEG recording, presenting auditory stimuli
in sequences where the standard stimulus changes throughout the experiment.

Author: [Niels Engedal, Mikkel Glavind]
Date: March 2025
"""
import psychtoolbox as ptb # need this to be installed on the machine!
from psychopy import visual, core, event, gui, sound, data, logging, prefs
import os
import sys
import pandas as pd
import numpy as np
import json
import random
from datetime import datetime
import pickle

# Import custom modules
from config import TRIGGER_CODES, DEFAULT_PARAMS
from roving_sequences import create_roving_block, get_full_stimulus_list, generate_optimized_blocks


# Import parallel port functions
try:
    # Use the triggers module approach from the example
    from triggers import setParallelData
except:
    print("Error importing triggers module. Creating a simulated trigger function.")
    # Define a fallback version if module not found
    def setParallelData(code=1):
        print(f"SIMULATED TRIGGER: {code}")
        
# Setup trigger pull-down delay
t_clear = 2 / 1000  # 2 ms

# Setup hardware preferences
prefs.hardware['audioLib'] = 'PTB' # type: ignore
prefs.hardware['audioLatencyMode'] = 3 # type: ignore
prefs.hardware['audioDevice'] = 14 # type: ignore
#prefs.hardware['audioLib'] = 'pyo'  # Alternative audio backend

# ----- Folder Setup -----
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Create folders if they don't exist
DATA_FOLDER = os.path.join(_thisDir, 'data')
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

OPTIMIZED_BLOCKS_FOLDER = os.path.join(_thisDir, 'optimized_blocks')
if not os.path.exists(OPTIMIZED_BLOCKS_FOLDER):
    os.makedirs(OPTIMIZED_BLOCKS_FOLDER)

STIMULI_FOLDER = os.path.join(_thisDir, 'stimuli')

# ----- Utility Functions -----

def save_optimized_blocks(blocks, filename):
    """
    Save optimized blocks to a pickle file for later use.
    
    Parameters:
    blocks (list): List of optimized block dictionaries
    filename (str): Filename to save to (without extension)
    
    Returns:
    str: Full path to the saved file
    """
    full_path = os.path.join(OPTIMIZED_BLOCKS_FOLDER, f"{filename}.pkl")
    with open(full_path, 'wb') as f:
        pickle.dump(blocks, f)
    
    # Also save a JSON version for human readability
    json_path = os.path.join(OPTIMIZED_BLOCKS_FOLDER, f"{filename}.json")
    # Convert NumPy arrays to lists for JSON serialization if needed
    json_friendly_blocks = []
    
    for block in blocks:
        json_block = {k: v for k, v in block.items()}
        json_friendly_blocks.append(json_block)
    
    with open(json_path, 'w') as f:
        json.dump(json_friendly_blocks, f, indent=2)
    
    return full_path

def load_optimized_blocks(filename):
    """
    Load optimized blocks from a pickle file.
    
    Parameters:
    filename (str): Filename to load from (without extension)
    
    Returns:
    list: List of optimized block dictionaries
    """
    full_path = os.path.join(OPTIMIZED_BLOCKS_FOLDER, f"{filename}.pkl")
    
    if not os.path.exists(full_path):
        print(f"Optimized blocks file not found: {full_path}")
        return None
    
    with open(full_path, 'rb') as f:
        blocks = pickle.load(f)
    
    return blocks

def generate_or_load_blocks(params, block_name=None, force_regenerate=False):
    """
    Generate or load optimized blocks based on parameters.
    
    Parameters:
    params (dict): Parameters for block generation
    block_name (str): Name to use for the block file (default: based on params)
    force_regenerate (bool): Force regeneration even if file exists
    
    Returns:
    list: List of optimized blocks
    """
    # Generate a default name based on key parameters if none provided
    if block_name is None:
        block_name = (f"blocks_duration{params['block_duration']}_"
                     f"min{params['min_seq_length']}_max{params['max_seq_length']}")
    
    file_path = os.path.join(OPTIMIZED_BLOCKS_FOLDER, f"{block_name}.pkl")
    
    # Check if we need to generate new blocks
    if force_regenerate or not os.path.exists(file_path):
        print(f"Generating new optimized blocks with parameters: {params}")
        blocks = generate_optimized_blocks(
            params, 
            num_candidates=50,  # Generate 50 candidate blocks
            num_to_select=3      # Select the 3 most balanced blocks
        )
        save_optimized_blocks(blocks, block_name)
    else:
        print(f"Loading existing optimized blocks: {file_path}")
        blocks = load_optimized_blocks(block_name)
    
    return blocks

def blocks_to_csv(blocks, stim_lists, participant_id, participant_responses=None):
    """
    Convert blocks and stimulus lists to a CSV file for data analysis.
    
    Parameters:
    blocks (list): List of block dictionaries
    stim_lists (list): List of stimulus lists
    participant_id (str): Participant identifier
    participant_responses (list): List of participant response data
    
    Returns:
    str: Path to the saved CSV file
    """
    # Create a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"p{participant_id}_{timestamp}_blocks.csv"
    file_path = os.path.join(DATA_FOLDER, filename)
    
    # Prepare data for CSV
    all_trials = []
    
    # Create a lookup for responses by block and trial
    # Create a lookup for responses by block and trial
    responses_by_trial = {}
    if participant_responses:
        for response in participant_responses:
            block_idx = response['block']
            trial_idx = response['trial']
            key = f"{block_idx}_{trial_idx}"
            if key not in responses_by_trial:
                responses_by_trial[key] = []
            responses_by_trial[key].append(response['time_in_trial'])  # Store time in trial instead of absolute time
    
    for block_idx, stim_list in enumerate(stim_lists):
        for stim_idx, stim in enumerate(stim_list):
            trial_data = {
                'participant_id': participant_id,
                'block_idx': block_idx + 1,
                'trial_idx': stim_idx + 1,
                'stimulus_type': stim['type'],
                'is_deviant': stim['is_deviant'],
                'onset_time': stim['onset'],
                'jitter': stim['jitter'],
                'duration': stim['duration'],
                'trigger_code': stim['trigger_code']
            }
            
            # Add transition information for deviants
            if stim['is_deviant'] and 'transition_type' in stim:
                trial_data['transition_type'] = stim['transition_type']
            else:
                trial_data['transition_type'] = ''
            
            # Add participant response data
            key = f"{block_idx+1}_{stim_idx+1}"
            trial_data['response_count'] = 0
            trial_data['response_times'] = ""
            
            if key in responses_by_trial:
                trial_data['response_count'] = len(responses_by_trial[key])
                trial_data['response_times'] = ";".join([f"{t:.3f}" for t in responses_by_trial[key]])
                
            all_trials.append(trial_data)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(all_trials)
    df.to_csv(file_path, index=False)
    
    print(f"Block data saved to {file_path}")
    return file_path

# ----- Experiment Interface -----
def show_experiment_info_dialog():
    """
    Display dialog to gather experiment and participant information.
    
    Returns:
    dict: Experiment information and settings
    bool: Whether the dialog was OK'd or canceled
    """
    # Prepare the experiment info dialog
    exp_info = {
        'participant_id': '',
        'age': '',
        'gender': ['male', 'female', 'non-binary', 'other', 'prefer not to say'],
        'nationality': '',
        'max_duration_minutes': 15,
        'isi_min_ms': 700,
        'isi_max_ms': 850,
        'stim_duration_ms': 100,
        'regenerate_blocks': False,
        'eeg_enabled': True,
        'debug_mode': False
    }
    
    # Show the dialog
    dlg = gui.DlgFromDict(
        dictionary=exp_info,
        title='Roving MMN Experiment',
        fixed=['regenerate_blocks'],
        order=['participant_id', 'age', 'gender', 'nationality', 
               'max_duration_minutes', 'isi_min_ms', 'isi_max_ms', 
               'stim_duration_ms', 'eeg_enabled', 'debug_mode']
    )
    
    if dlg.OK:
        # Format participant ID with leading zeros
        exp_info['participant_id'] = exp_info['participant_id'].zfill(3)
        
        # Calculate block duration based on max experiment time
        # Divide by desired number of blocks (3) with some margin for breaks
        exp_info['block_duration'] = int((exp_info['max_duration_minutes'] * 60 * 1000) / 3.5)
        
        # Return experiment info if dialog was OK'd
        return exp_info, True
    else:
        # Return None if dialog was canceled
        return None, False

def prepare_experiment_parameters(exp_info):
    """
    Prepare experiment parameters based on GUI input.
    
    Parameters:
    exp_info (dict): Information from the experiment dialog
    
    Returns:
    dict: Parameters for block generation
    """
    # Start with default parameters
    params = DEFAULT_PARAMS.copy()
    
    # Detect actual audio file durations
    detected_durations = detect_audio_durations()
    
    # Create our stim_durations dictionary
    if 'stim_durations' in params:
        # Start with defaults if they exist
        stim_durations = params['stim_durations'].copy()
    else:
        # Otherwise create a new dict
        stim_durations = {}
    
    # Override with detected durations
    for stim_type, duration in detected_durations.items():
        stim_durations[stim_type] = duration
    
    # Calculate block duration based on max experiment time from GUI
    block_duration = int((exp_info['max_duration_minutes'] * 60 * 1000) / 3.5)
    
    # Override parameters with GUI values (explicitly listed for clarity)
    params.update({
        'block_duration': block_duration,  # From GUI (max_duration_minutes)
        'min_jitter': exp_info['isi_min_ms'],  # From GUI
        'max_jitter': exp_info['isi_max_ms'],  # From GUI
        'stim_duration': exp_info['stim_duration_ms'],  # From GUI (default stim duration)
        'stim_durations': stim_durations  # Combined from default + detected
    })
    
    # Print the final parameter set for verification
    print("\nExperiment parameters after overrides:")
    for key, value in params.items():
        if key != 'stim_durations':
            print(f"  {key}: {value}")
        else:
            print(f"  {key}:")
            for stim, dur in value.items():
                print(f"    {stim}: {dur} ms")
    
    return params

def load_stimuli():
    """
    Load audio stimuli from files with basic parameters.
    
    Returns:
    dict: Dictionary mapping stimulus types to PsychoPy sound objects
    """
    stimuli = {}
    
    # Define the stimulus file mapping
    stim_files = {
        'N1': 'N1.wav',
        'N2': 'N2.wav',
        'Ai1': 'Ai1.wav',
        'Ai2': 'Ai2.wav'
    }
    
    # Try to verify file existence first
    for stim_type, filename in stim_files.items():
        file_path = os.path.join(STIMULI_FOLDER, filename)
        if os.path.exists(file_path):
            print(f"File exists: {file_path}")
        else:
            print(f"WARNING: File does not exist: {file_path}")
    
    # Load each stimulus
    for stim_type, filename in stim_files.items():
        file_path = os.path.join(STIMULI_FOLDER, filename)
        
        try:
            # Use minimal parameters that are more compatible
            stimuli[stim_type] = sound.Sound(
                value=file_path,
                secs=-1,
                volume=0.8
            )
            print(f"Loaded stimulus: {stim_type} from {file_path}")
        except Exception as e:
            print(f"Error loading stimulus {stim_type}: {e}")
            print(f"Trying alternative loading method for {stim_type}")
            
            try:
                # Fall back to even simpler initialization if needed
                stimuli[stim_type] = sound.Sound(file_path)
                print(f"Successfully loaded {stim_type} with alternative method")
            except Exception as e2:
                print(f"Alternative loading also failed: {e2}")
                stimuli[stim_type] = sound.Sound(440, secs=0.1)  # Fallback tone # type: ignore
    
    return stimuli

def detect_audio_durations():
    """
    Automatically detect the duration of audio files.
    
    Returns:
    dict: Mapping of stimulus types to their actual durations in ms
    """
    durations = {}
    
    # Define the stimulus file mapping
    stim_files = {
        'N1': 'N1.wav',
        'N2': 'N2.wav',
        'Ai1': 'Ai1.wav',
        'Ai2': 'Ai2.wav'
    }
    
    for stim_type, filename in stim_files.items():
        file_path = os.path.join(STIMULI_FOLDER, filename)
        
        try:
            # Create a temporary sound object to get duration
            temp_sound = sound.Sound(file_path)
            duration_sec = temp_sound.getDuration()
            duration_ms = int(duration_sec * 1000)
            durations[stim_type] = duration_ms
            print(f"Detected {stim_type} duration: {duration_ms} ms")
        except Exception as e:
            print(f"Error detecting duration for {stim_type}: {e}")
            # Use a default value
            durations[stim_type] = 500
            print(f"Using default duration for {stim_type}: 500 ms")
    
    return durations

def save_and_quit(win, blocks, stim_lists, exp_info, participant_responses, early_exit=False):
    """
    Save all data and exit the experiment gracefully.
    
    Parameters:
    win (visual.Window): PsychoPy window
    blocks (list): List of block dictionaries
    stim_lists (list): List of stimulus lists
    exp_info (dict): Experiment information
    participant_responses (list): List of participant responses
    early_exit (bool): Whether this is an early exit
    """
    try:
        # Create early exit marker in filename if needed
        suffix = "_early_exit" if early_exit else ""
        
        # Save the block data to CSV
        print(f"Saving data for participant {exp_info['participant_id']}{suffix}...")
        
        # Modify the save function to include the suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"p{exp_info['participant_id']}_{timestamp}{suffix}_blocks.csv"
        file_path = os.path.join(DATA_FOLDER, filename)
        
        # Prepare data for CSV (simplified version of blocks_to_csv)
        all_trials = []
        
        # Create a lookup for responses by block and trial
        # Create a lookup for responses by block and trial
        responses_by_trial = {}
        if participant_responses:
            for response in participant_responses:
                block_idx = response['block']
                trial_idx = response['trial']
                key = f"{block_idx}_{trial_idx}"
                if key not in responses_by_trial:
                    responses_by_trial[key] = []
                responses_by_trial[key].append(response['time_in_trial'])  # Store time in trial instead of absolute time
        
        for block_idx, stim_list in enumerate(stim_lists):
            for stim_idx, stim in enumerate(stim_list):
                # Only include trials that might have been presented
                # (i.e., up to the current block and trial if early exit)
                if early_exit and (block_idx > exp_info.get('current_block', 0) or 
                                  (block_idx == exp_info.get('current_block', 0) and 
                                   stim_idx > exp_info.get('current_trial', 0))):
                    continue
                    
                trial_data = {
                    'participant_id': exp_info['participant_id'],
                    'block_idx': block_idx + 1,
                    'trial_idx': stim_idx + 1,
                    'stimulus_type': stim['type'],
                    'is_deviant': stim['is_deviant'],
                    'onset_time': stim['onset'],
                    'jitter': stim['jitter'],
                    'duration': stim['duration'],
                    'trigger_code': stim['trigger_code']
                }
                
                # Add transition information for deviants
                if stim['is_deviant'] and 'transition_type' in stim:
                    trial_data['transition_type'] = stim['transition_type']
                else:
                    trial_data['transition_type'] = ''
                
                # Add participant response data
                key = f"{block_idx+1}_{stim_idx+1}"
                trial_data['response_count'] = 0
                trial_data['response_times'] = ""
                
                if key in responses_by_trial:
                    trial_data['response_count'] = len(responses_by_trial[key])
                    trial_data['response_times'] = ";".join([f"{t:.3f}" for t in responses_by_trial[key]])
                    
                all_trials.append(trial_data)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_trials)
        df.to_csv(file_path, index=False)
        print(f"Block data saved to {file_path}")
        
        # Also save the raw responses to a separate file
        response_filename = os.path.join(DATA_FOLDER, f"p{exp_info['participant_id']}_{timestamp}{suffix}_responses.csv")
        if participant_responses:
            response_df = pd.DataFrame(participant_responses)
            response_df.to_csv(response_filename, index=False)
            print(f"Response data saved to {response_filename}")
        
        # Show message if window is still open
        if not win.closed:
            if early_exit:
                msg = "Experiment exited early.\nYour data has been saved.\n\nPress any key to close."
            else:
                msg = "Data has been successfully saved.\n\nPress any key to exit."
                
            saved_text = visual.TextStim(
                win=win,
                text=msg,
                font='Arial',
                height=0.07,
                color='white'
            )
            saved_text.draw()
            win.flip()
            event.waitKeys()
            win.close()
            
    except Exception as e:
        print(f"Error saving data: {e}")
        logging.log(level=logging.ERROR, msg=f"Error during save_and_quit: {e}")
    finally:
        if not win.closed:
            win.close()
        core.quit()

def run_experiment(exp_info):
    """
    Run the main experiment with optimized PsychToolBox timing for EEG integration.
    
    Parameters:
    exp_info (dict): Experiment information and settings
    """
    # Initialize logging
    log_file = os.path.join(DATA_FOLDER, f"p{exp_info['participant_id']}_log.txt")
    logging.console.setLevel(logging.WARNING)
    logging.LogFile(log_file, level=logging.INFO)
    
    # Prepare parameters
    params = prepare_experiment_parameters(exp_info)
    
    # Generate or load optimized blocks
    block_name = f"p{exp_info['participant_id']}_blocks"
    blocks = generate_or_load_blocks(params, block_name, force_regenerate=exp_info['regenerate_blocks'])
    
    if not blocks:
        print("Error generating or loading blocks. Exiting.")
        return
    
    stim_lists = []
    for block in blocks:
        # Apply detected durations to the block
        block['stim_durations'] = params['stim_durations']
        block['stim_duration'] = params['stim_duration']
        # Get stimulus list with updated durations
        stim_list = get_full_stimulus_list(block)
        stim_lists.append(stim_list)

    # Verify audio durations in stimulus lists
    print("\nVerifying audio durations in stimulus lists:")
    for block_idx, stim_list in enumerate(stim_lists):
        durations_in_block = {}
        for stim in stim_list:
            if stim['type'] not in durations_in_block:
                durations_in_block[stim['type']] = stim['duration']
        
        print(f"Block {block_idx+1} stimulus durations:")
        for stim_type, duration in sorted(durations_in_block.items()):
            print(f"  {stim_type}: {duration} ms")
    
    # Load stimuli
    stimuli = load_stimuli()
    
    # Set up window
    win = visual.Window(
        size=(1024, 768),
        fullscr=not exp_info['debug_mode'],  # Fullscreen unless in debug mode
        screen=0,
        allowGUI=False,
        allowStencil=False,
        monitor='testMonitor',
        color='black',
        colorSpace='rgb',
        blendMode='avg',
        useFBO=True
    )
    
    # Create visual components
    welcome_text = visual.TextStim(
        win=win,
        name='welcome',
        text="Welcome to the experiment!\n\nWe appreciate your participation.",
        font='Arial',
        pos=(0, 0),
        height=0.07,
        wrapWidth=1.8,
        color='white'
    )
    
    instruction_text = visual.TextStim(
        win=win,
        name='instructions',
        text=(
            "In this experiment, you will hear different sounds.\n\n"
            "Your task is to focus on the fixation cross and listen attentively.\n\n"
            "Press the SPACE bar whenever you think there has been a switch\n"
            "between a Human and AI voice (in either direction).\n\n"
            "When you press SPACE, the fixation cross will briefly turn blue\n"
            "to confirm your response was recorded.\n\n"
            "There will be three blocks with short breaks between them.\n\n"
            "The FIRST voice is a Human Voice.\n\n"
            "Press the SPACE bar to begin."
        ),
        font='Arial',
        pos=(0, 0),
        height=0.06,
        wrapWidth=1.8,
        color='white'
    )
    
    fixation = visual.TextStim(
        win=win,
        name='fixation',
        text="+",
        font='Arial',
        pos=(0, 0),
        height=0.1,
        color='white'
    )
    
    # Blue fixation for response feedback
    blue_fixation = visual.TextStim(
        win=win,
        name='blue_fixation',
        text="+",
        font='Arial',
        pos=(0, 0),
        height=0.1,
        color='blue'
    )
    
    end_text = visual.TextStim(
        win=win,
        name='end',
        text=(
            "The experiment is now complete.\n\n"
            "Thank you for your participation!\n\n"
            "Press SPACE to exit."
        ),
        font='Arial',
        pos=(0, 0),
        height=0.07,
        wrapWidth=1.8,
        color='white'
    )
    
    # Show welcome screen
    welcome_text.draw()
    win.flip()
    event.waitKeys(keyList=['space'])
    
    # Show instructions
    instruction_text.draw()
    win.flip()
    event.waitKeys(keyList=['space'])
    
    # Track participant responses
    participant_responses = []
    
    # Run each block
    for block_idx, stim_list in enumerate(stim_lists):
        print(f"Starting block {block_idx+1}/{len(stim_lists)}")
        
        # Show ready message
        ready_text = visual.TextStim(
            win=win,
            text=f"Block {block_idx+1} of {len(stim_lists)}\n\nPress SPACE to begin",
            font='Arial',
            height=0.07,
            color='white'
        )
        ready_text.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
        
        # Send block start trigger
        if exp_info['eeg_enabled']:
            setParallelData(TRIGGER_CODES['special_codes']['block_start'])
            ptb.WaitSecs(t_clear)
            setParallelData(0) # Clear the trigger after 1 msec
        
        # Show fixation cross
        fixation.draw()
        win.flip()
        
        # Pre-block fixation period with PTB precision
        fixation_end = ptb.GetSecs() + 2.0  # 2 second pre-block fixation
        while ptb.GetSecs() < fixation_end:
            ptb.WaitSecs(0.01)  # Minimal wait
        
        # Clear any previous key presses
        event.clearEvents()
        
        # Present stimuli in this block
        for trial_idx, stim in enumerate(stim_list):            
            start_frame = win.getFutureFlipTime(clock="ptb") # clock="ptb" ensures compatibility with ptb.clock
            
            sound_stim = stimuli[stim['type']]
            measured_stim_onset = 0
            # Play the sound with precise onset time
            sound_stim.play(when = start_frame)
            
            def on_stimuli():
                # Present EEG trigger immediately after scheduling sound
                if exp_info['eeg_enabled']:
                    setParallelData(stim['trigger_code'])        
                
                # Record exact stimulus onset time
                nonlocal measured_stim_onset
                measured_stim_onset = ptb.GetSecs()
                                               
            win.callOnFlip(on_stimuli)
            
            # Draw fixation for this trial
            fixation.draw()
            win.flip()

            stim['actual_onset'] = measured_stim_onset
            stim['onset_precision'] = measured_stim_onset - start_frame
            
            # Log the precision
            logging.log(level=logging.INFO, 
                        msg=f"Stimulus {stim['type']} onset precision: {stim['onset_precision']*1000:.2f} ms")
            
            # Ensure we're using the detected duration
            detected_duration = params['stim_durations'].get(stim['type'])
            if detected_duration and detected_duration != stim['duration']:
                print(f"Warning: Duration mismatch for {stim['type']} - " 
                    f"CSV has {stim['duration']}ms but detected {detected_duration}ms")

            # Calculate jittered end time
            jitter = stim['jitter'] / 1000  # Convert ms to seconds
            detected_duration /= 1000
            trial_end_time = measured_stim_onset + detected_duration + jitter
            
            while (now := ptb.GetSecs()) < trial_end_time:
                
                # Check for space bar presses

                keys = event.getKeys(['space', 'escape', 'q'])
                
                time_since_stim = now - measured_stim_onset
                # Find during which part of the trial the response was made
                if time_since_stim > detected_duration:
                    during = 'jitter'
                else:
                    during = 'stimulus'
                
                response = 0

                if 'space' in keys:
                    # Record response time with high precision
                    response_time = ptb.GetSecs()
                    time_since_stim = response_time - start_frame
                    
                    correctness = stim['is_deviant'] and "ai" in stim.get('transition_type', '').lower() and "n" in stim.get('transition_type', '').lower()

                    if not correctness:
                        response = TRIGGER_CODES['special_codes']["incorrect_response"]
                    else:
                        response = TRIGGER_CODES["special_codes"]["correct_response"]

                    # Log response with precise timing
                    participant_responses.append({
                        'block': block_idx + 1,
                        'trial': trial_idx + 1,
                        'time': response_time,
                        'time_in_trial': time_since_stim,
                        'stimulus_type': stim['type'],
                        'is_deviant': stim['is_deviant'],
                        'transition_type': stim.get('transition_type', '') if stim['is_deviant'] else '',
                        'precision': 'PTB',  # Mark as using PTB precision
                        'during': during   # Indicate response during jitter period
                    })
                    
                    # Log the response
                    logging.log(level=logging.INFO, 
                                msg=f"Response at block {block_idx+1}, trial {trial_idx+1} ({during}), "
                                    f"time {response_time:.6f}s, stim {stim['type']}")
                                        
                    # Precise feedback duration
                    feedback_end = ptb.GetSecs() + 0.2  # 200ms feedback
                    while ptb.GetSecs() < feedback_end:
                        blue_fixation.draw()
                        
                    win.flip()
                
                if 'escape' in keys or 'q' in keys:
                    # Update current position info for partial data saving
                    exp_info['current_block'] = block_idx
                    exp_info['current_trial'] = trial_idx
                    # Save data and exit
                    save_and_quit(win, blocks, stim_lists, exp_info, participant_responses, early_exit=True)
                
                # Minimal wait to avoid busy loop while maintaining precision
                ptb.WaitSecs(t_clear)
                
                if exp_info["eeg_enabled"]:
                    setParallelData(response) # Clear the trigger after 1 msec
                
        
        # Send block end trigger
        if exp_info['eeg_enabled']:
            setParallelData(TRIGGER_CODES['special_codes']['block_end'])
            ptb.WaitSecs(t_clear)
            setParallelData(0) # Clear the trigger after 1 msec
        
        # Show break between blocks (except after the last block)
        if block_idx < len(stim_lists) - 1:
            # Send break start trigger if EEG is enabled
            if exp_info['eeg_enabled']:
                setParallelData(TRIGGER_CODES['special_codes']['break_start'])
                ptb.WaitSecs(t_clear)
                setParallelData(0) # Clear the trigger after 1 msec
            
            # Create break text components
            break_text = visual.TextStim(
                win=win,
                name='break',
                text=(
                    "Break time!\n\n"
                    "Please take a 30-second rest.\n\n"
                    "Remember to press SPACE when you think you hear a switch\n"
                    "between Human and AI voices in the next block.\n\n"
                    "The break will end automatically in 30 seconds."
                ),
                font='Arial',
                pos=(0, 0),
                height=0.06,
                wrapWidth=1.8,
                color='white'
            )
            
            # Timer display
            timer_text = visual.TextStim(
                win=win,
                name='timer',
                text="30",
                font='Arial',
                pos=(0, -0.4),
                height=0.08,
                color='white'
            )
            
            # Calculate break end time with PTB precision
            break_duration = 30  # 30 seconds
            break_end_time = ptb.GetSecs() + break_duration
            
            # Display break message with countdown
            while ptb.GetSecs() < break_end_time:
                # Update timer display
                seconds_left = int(break_end_time - ptb.GetSecs() + 0.5)  # Round to nearest second
                timer_text.setText(f"{seconds_left}")
                
                # Draw both texts
                break_text.draw()
                timer_text.draw()
                win.flip()
                
                # Brief pause to avoid busy loop
                ptb.WaitSecs(0.05)  # Check time approximately 20 times per second
                
                # Check for escape key
                keys = event.getKeys(['escape', 'q'])
                if 'escape' in keys or 'q' in keys:
                    # Update current position info for partial data saving
                    exp_info['current_block'] = block_idx
                    exp_info['current_trial'] = len(stim_list) - 1  # End of current block
                    # Save data and exit
                    save_and_quit(win, blocks, stim_lists, exp_info, participant_responses, early_exit=True)
            
            # Break is over, show ready message
            ready_text = visual.TextStim(
                win=win,
                text=f"Break complete!\n\nBlock {block_idx+2} of {len(stim_lists)} will begin now.\n\nPress SPACE to continue.",
                font='Arial',
                height=0.07,
                color='white'
            )
            ready_text.draw()
            win.flip()
            
            # Send break end trigger if EEG is enabled
            if exp_info['eeg_enabled']:
                setParallelData(TRIGGER_CODES['special_codes']['break_end'])
                ptb.WaitSecs(t_clear)
                setParallelData(0) # Clear the trigger after 1 msec
                
            # Wait for space to continue
            event.waitKeys(keyList=['space'])
    
    # Show completion message
    end_text.draw()
    win.flip()
    event.waitKeys(keyList=['space'])
    
    # Save data and exit normally
    save_and_quit(win, blocks, stim_lists, exp_info, participant_responses, early_exit=False)

# ----- Main Program -----

if __name__ == "__main__":
    # Show dialog to get experiment info
    exp_info, dialog_ok = show_experiment_info_dialog()
    
    if dialog_ok:
        # Run the experiment
        run_experiment(exp_info)
    else:
        print("Experiment canceled by user")