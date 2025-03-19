"""
MMN Oddball Experiment Stimulus Generator

This script generates a randomized sequence for an EEG-based MMN experiment where
participants hear three types of auditory stimuli (N, N2, Ai) in an oddball paradigm.

Stimuli:
- "N" = First human recording of "car" (from speaker Niels)
- "N2" = Second human recording of "car" (to control for natural human speech variability)
- "Ai" = AI-generated "car" based on N's voice

The script handles:
1. Randomized deviant placement maintaining ~10% deviant probability
2. No consecutive deviants
3. 2-3 standard trials before each deviant
4. Balanced blocks where all stimuli types serve as both standard and deviant
"""

from psychopy import visual, core, sound, gui, data, event, logging
import os, sys, random, csv, time, json
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from psychopy.constants import (NOT_STARTED, STARTED, FINISHED)

from helper_functions.sequence_generation import generate_block_sequence, create_experiment_blocks
from helper_functions.analysis import analyze_sequences, visualize_sequence, check_stimuli_duration
from helper_functions.timing import calculate_experiment_parameters
from helper_functions.io_utils import export_sequences_to_csv, load_stimuli, combine_csv_files, create_folder_structure

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# ---- Config Constants ----
EXPERIMENT_NAME = 'MMN Oddball Paradigm'
DATA_FOLDER = 'data'
STIMULI_FOLDER = 'stimuli'
HUMAN_N_STIMULUS = os.path.join(STIMULI_FOLDER, 'N', 'car.wav')  # First human recording
HUMAN_N2_STIMULUS = os.path.join(STIMULI_FOLDER, 'N2', 'car.wav')  # Second human recording
AI_STIMULUS = os.path.join(STIMULI_FOLDER, 'Ai', 'car.wav')  # AI recording
VIDEO_FOLDER = os.path.join(STIMULI_FOLDER, 'video')
DEFAULT_VIDEO = os.path.join(VIDEO_FOLDER, 'GymTracker-Website-Demo-3Marts.mov')
TOTAL_TRIALS_PER_BLOCK = 500
DEVIANT_PROBABILITY = 0.1
NUM_BLOCKS = 3
TRIGGER_CODES = {
    # Standards
    'N_standard': 1,
    'N2_standard': 2,
    'Ai_standard': 3,
    # Deviants
    'N_deviant': 11,
    'N2_deviant': 12,
    'Ai_deviant': 13,
    # Block markers
    'block_start': 200,
    'block_end': 201
}

# Create a dictionary of stimuli paths
stimuli_paths = {
    'N': HUMAN_N_STIMULUS,
    'N2': HUMAN_N2_STIMULUS,
    'Ai': AI_STIMULUS
}

# Create data folder if it doesn't exist
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# Store info about the experiment session
exp_info = {
    'participant_id': '',
    'age': '',
    'gender': ['male', 'female', 'non-binary', 'prefer not to say'],
    'eeg': ['yes', 'no'],  # Whether EEG recording is enabled
    'sound_check': ['no', 'yes'],  # Whether to perform a sound check before experiment # I have just changed this to be default no I hope.
    'view_sequence': ['no', 'yes'],  # Whether to view the sequence distribution before running
    'max_duration_minutes': '15',  # Maximum experiment duration in minutes
    'auto_calculate_trials': ['yes', 'no'],  # Auto-calculate trials based on time limit
    'isi_ms': '500',  # Inter-stimulus interval in milliseconds
    'stimulus_duration_ms': '300',  # Duration of each stimulus
    'block_break_duration_s': '30',  # Duration of break between blocks in seconds
    'min_standards': '2',  # Minimum number of standard trials before a deviant
    'max_standards': '14',  # Maximum number of standard trials before a deviant
    'deviant_probability': '0.1',  # Probability of deviant stimuli
    'debug_mode': ['no', 'yes'],  # Whether to run in debug mode with fewer trials
    'trials_per_block': '500',  # Number of trials per block (can be auto-calculated)
    'num_blocks': '3',  # Number of blocks to run
    'deviant_delay_ms': '0',  # Additional delay after deviant stimuli (0 = no special delay)
    'random_seed': '',  # Random seed for sequence generation (empty for random)
    'video_background': ['no', 'yes'],  # Whether to play a video in the background
    'video_path': '',  # Path to the video file
}

# Present a dialog to get participant info
dlg = gui.DlgFromDict(dictionary=exp_info, sortKeys=False, title=EXPERIMENT_NAME)
if not dlg.OK:
    core.quit()  # User pressed cancel



# Format the participant ID with leading zeros
exp_info['participant_id'] = exp_info['participant_id'].zfill(3)

# Create a filename for the data file
exp_info['date'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"{DATA_FOLDER}/p{exp_info['participant_id']}_{exp_info['date']}"
logfile = logging.LogFile(f"{filename}.log", level=logging.EXP)

# Convert string parameters to appropriate types
isi_ms = int(exp_info['isi_ms'])
stimulus_duration_ms = int(exp_info['stimulus_duration_ms'])
block_break_duration_s = int(exp_info['block_break_duration_s'])
eeg_enabled = exp_info['eeg'] == 'yes'
sound_check = exp_info['sound_check'] == 'yes'
view_sequence = exp_info['view_sequence'] == 'yes'
debug_mode = exp_info['debug_mode'] == 'yes'
deviant_delay_ms = int(exp_info['deviant_delay_ms'])
max_duration_minutes = float(exp_info['max_duration_minutes'])
auto_calculate_trials = exp_info['auto_calculate_trials'] == 'yes'
num_blocks = int(exp_info['num_blocks'])
min_standards = int(exp_info['min_standards'])
max_standards = int(exp_info['max_standards'])
deviant_probability = float(exp_info['deviant_probability'])
use_video = exp_info['video_background'] == 'yes'
video_path = exp_info['video_path']

# Validate min_standards and max_standards
if min_standards < 1:
    min_standards = 1
    print("Warning: Minimum standards cannot be less than 1. Setting to 1.")

if max_standards < min_standards:
    max_standards = min_standards + 1
    print(f"Warning: Maximum standards cannot be less than minimum standards. Setting to {max_standards}.")

# Only try to find a video if video background is enabled
if use_video:
    # Check for user-specified path first
    if exp_info['video_path'] and os.path.exists(exp_info['video_path']):
        video_path = exp_info['video_path']
        print(f"Using user-specified video: {video_path}")
    # Then check for default video
    elif os.path.exists(DEFAULT_VIDEO):
        video_path = DEFAULT_VIDEO
        print(f"Using default video: {DEFAULT_VIDEO}")
    # If neither works, disable video
    else:
        use_video = False
        video_path = ""
        print("No video file found. Using fixation cross.")
else:
    # Video is disabled, set an empty path
    video_path = ""

# make global for easy access
DEVIANT_PROBABILITY = deviant_probability

# Pass all needed parameters to the helper functions
config = {
    'HUMAN_N_STIMULUS': HUMAN_N_STIMULUS,
    'HUMAN_N2_STIMULUS': HUMAN_N2_STIMULUS,
    'AI_STIMULUS': AI_STIMULUS,
    'TRIGGER_CODES': TRIGGER_CODES,
    'TOTAL_TRIALS_PER_BLOCK': TOTAL_TRIALS_PER_BLOCK,
    'DEVIANT_PROBABILITY': DEVIANT_PROBABILITY,
    'NUM_BLOCKS': num_blocks,
    'DATA_FOLDER': DATA_FOLDER,
    'filename': filename,
    'exp_info': exp_info,
    'min_standards': min_standards,
    'max_standards': max_standards,
}

# Check stimuli durations for consistency
avg_stimulus_duration, stimuli_warnings = check_stimuli_duration(stimuli_paths)

# Auto-calculate trials per block if selected
if auto_calculate_trials:
    calculated_params = calculate_experiment_parameters(
        max_duration_minutes=max_duration_minutes,
        num_blocks=num_blocks,
        block_break_duration_s=block_break_duration_s,
        stimulus_duration_ms=stimulus_duration_ms,
        isi_ms=isi_ms,
        deviant_delay_ms=deviant_delay_ms,
        deviant_probability=deviant_probability
    )
    
    TOTAL_TRIALS_PER_BLOCK = calculated_params['max_trials_per_block']
    config['TOTAL_TRIALS_PER_BLOCK'] = TOTAL_TRIALS_PER_BLOCK
    
    # Show calculation results to user
    auto_calc_info = {
        'calculated_trials_per_block': str(TOTAL_TRIALS_PER_BLOCK),
        'estimated_duration_minutes': f"{calculated_params['estimated_duration_minutes']:.1f}",
        'trials_per_minute': str(calculated_params['trials_per_minute']),
        'proceed': ['yes', 'no']
    }
    
    calc_dlg = gui.DlgFromDict(
        dictionary=auto_calc_info, 
        title="Time-Based Calculation", 
        fixed=['calculated_trials_per_block', 'estimated_duration_minutes', 'trials_per_minute']
    )
    
    if not calc_dlg.OK or auto_calc_info['proceed'] == 'no':
        core.quit()
    
    print(f"Auto-calculated trials per block: {TOTAL_TRIALS_PER_BLOCK}")
    print(f"Estimated experiment duration: {calculated_params['estimated_duration_minutes']:.1f} minutes")
else:
    # Use user-specified value
    TOTAL_TRIALS_PER_BLOCK = int(exp_info['trials_per_block'])
    config['TOTAL_TRIALS_PER_BLOCK'] = TOTAL_TRIALS_PER_BLOCK

# Set random seed if provided
if exp_info['random_seed']:
    try:
        seed_value = int(exp_info['random_seed'])
        random.seed(seed_value)
        np.random.seed(seed_value)
        print(f"Using random seed: {seed_value}")
    except ValueError:
        print("Invalid random seed provided. Using system-generated seed.")
        # Generate and record a random seed for reuse
        seed_value = int(time.time())
        random.seed(seed_value)
        np.random.seed(seed_value)
else:
    # Generate and record a random seed for reuse
    seed_value = int(time.time())
    random.seed(seed_value)
    np.random.seed(seed_value)

# Save the used seed in the experiment info for reference
exp_info['used_random_seed'] = seed_value
config['exp_info'] = exp_info

# Global variables for experiment settings
fullscreen = True
print_triggers = eeg_enabled
show_trial_count = False
show_time_estimate = False
trial_interval_factor = 1.0

if debug_mode:
    # Show an additional debug configuration dialog
    debug_info = {
        'window_size': ['fullscreen', 'windowed'],
        'print_triggers': ['yes', 'no'],
        'show_trial_count': ['yes', 'no'],
        'show_time_estimate': ['yes', 'no'],
        'trial_interval_factor': '1.0',  # Speed factor for testing (1.0 = normal, 0.5 = twice as fast)
        'override_random_seed': '',  # Option to override the random seed
    }
    
    debug_dlg = gui.DlgFromDict(dictionary=debug_info, sortKeys=False, title="Debug Settings")
    if not debug_dlg.OK:
        core.quit()
    
    # Process debug settings
    fullscreen = debug_info['window_size'] == 'fullscreen'
    print_triggers = debug_info['print_triggers'] == 'yes'
    show_trial_count = debug_info['show_trial_count'] == 'yes'
    show_time_estimate = debug_info['show_time_estimate'] == 'yes'
    
    # Override random seed if provided in debug
    if debug_info['override_random_seed']:
        try:
            seed_value = int(debug_info['override_random_seed'])
            random.seed(seed_value)
            np.random.seed(seed_value)
            exp_info['used_random_seed'] = seed_value
            config['exp_info'] = exp_info
            print(f"Debug: Overriding random seed with: {seed_value}")
        except ValueError:
            print("Invalid debug random seed. Keeping original seed.")
    
    try:
        trial_interval_factor = float(debug_info['trial_interval_factor'])
        if trial_interval_factor <= 0:
            trial_interval_factor = 1.0
    except ValueError:
        trial_interval_factor = 1.0
        
    # Adjust timing based on factor
    isi_ms = int(isi_ms * trial_interval_factor)
    stimulus_duration_ms = int(stimulus_duration_ms * trial_interval_factor)
    block_break_duration_s = int(block_break_duration_s * trial_interval_factor)

# ---- PsychoPy Experiment Implementation ----
def initialize_experiment_window(use_video_param, video_path, DEFAULT_VIDEO):
    """
    Initialize the PsychoPy window and visual stimuli.
    
    Returns:
    - win: PsychoPy window object
    - visual_elements: Dictionary of visual elements for the experiment
    """
    # Create a local copy to avoid modifying the parameter
    use_video_final = use_video_param

    # Initialize window
    win = visual.Window(
        size=[1280, 720],
        fullscr=fullscreen,  # Use the fullscreen variable 
        screen=0,
        allowGUI=not fullscreen,  # Allow GUI only in windowed mode
        allowStencil=False,
        monitor='testMonitor',
        color='black',
        colorSpace='rgb',
        blendMode='avg',
        useFBO=True,
        waitBlanking=False, # Don't wait for screen refresh (may affect timing precision)
    )
    
    # Create visual elements
    visual_elements = {
        'fixation': visual.TextStim(
            win=win, name='fixation', text='+',
            font='Arial', pos=(0, 0), height=0.1,
            wrapWidth=None, ori=0, color='white',
            colorSpace='rgb', opacity=1, depth=0.0
        ),
        'instructions': visual.TextStim(
            win=win, name='instructions',
            text="In this experiment, you will hear a series of sounds.\n\n"
                 "Please maintain focus on the fixation cross (+) in the center of the screen.\n\n"
                 "You don't need to respond - just listen to the sounds attentively.\n\n"
                 f"This experiment has {num_blocks} blocks with short breaks between them.\n\n"
                 "Press SPACE to begin.",
            font='Arial', pos=(0, 0), height=0.06,
            wrapWidth=1.5, ori=0, color='white',
            colorSpace='rgb', opacity=1, depth=0.0
        ),
        'break_text': visual.TextStim(
            win=win, name='break_text',
            text="Take a short break.\n\n"
                 "The experiment will continue automatically after the break.\n\n"
                 "(Or press SPACE to continue now)",
            font='Arial', pos=(0, 0), height=0.06,
            wrapWidth=1.5, ori=0, color='white',
            colorSpace='rgb', opacity=1, depth=0.0
        ),
        'block_progress': visual.TextStim(
            win=win, name='block_progress',
            text="",  # Will be set during experiment
            font='Arial', pos=(0, -0.4), height=0.04,
            wrapWidth=None, ori=0, color='white',
            colorSpace='rgb', opacity=1, depth=0.0
        ),
        'completion': visual.TextStim(
            win=win, name='completion',
            text="Experiment completed.\n\nThank you for your participation!",
            font='Arial', pos=(0, 0), height=0.08,
            wrapWidth=None, ori=0, color='white',
            colorSpace='rgb', opacity=1, depth=0.0
        )
    }

    # Add video if enabled
    if use_video_final:
        try:
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                if os.path.exists(DEFAULT_VIDEO):
                    video_path = DEFAULT_VIDEO
                    print(f"Using default video: {DEFAULT_VIDEO}")
                else:
                    print("Default video not found. Using fixation cross.")
                    raise FileNotFoundError("Video file not found")
            
            # Create movie stimulus with looping enabled
            visual_elements['video'] = visual.MovieStim(
                win=win,
                name='movie',
                filename=video_path,
                pos=(0, 0),
                size=None,  # Keep original aspect ratio
                units='norm',
                loop=True,  # Loop if video ends before experiment
                noAudio=True,  # Ensure audio is disabled
                opacity=1.0
            )
            print(f"Video loaded successfully: {video_path}")
            
            # Update instructions text to mention video
            visual_elements['instructions'].text += "\n\nA silent video will play in the background."
            
        except Exception as e:
            print(f"Error loading video: {e}. Using fixation cross instead.")
            use_video_final = False
    
    return win, visual_elements, use_video_final

def send_trigger(code):
    """
    Send trigger code to EEG recording system.
    
    Parameters:
    - code: Numeric trigger code to send
    """
    if not eeg_enabled:
        if print_triggers and debug_mode:
            print(f"DEBUG TRIGGER: {code}")
        return
    
    # Print for both real mode and debug mode if print_triggers is enabled
    if print_triggers:
        print(f"TRIGGER: {code}")
    
    # Placeholder for actual trigger implementation
    # This would typically use a parallel port or USB trigger box
    
    # Example implementation for parallel port (requires pyparallel or similar):
    # try:
    #     import parallel
    #     port = parallel.Parallel(0)  # usually 0x0378 or 0x03BC
    #     port.setData(code)
    #     core.wait(0.001)  # 1ms pulse
    #     port.setData(0)  # Reset
    # except ImportError:
    #     logging.warning("Parallel port module not available. EEG triggers disabled.")

def run_sound_check(win, stimuli_dict, visual_elements, use_video):
    """
    Run a simple sound check before starting the experiment.
    """

    # Set up check text with video mention if video is enabled
    check_text_content = "Sound Check\n\n"
    check_text_content += "You will now hear each sound used in the experiment.\n\n"
    check_text_content += "Please confirm you can hear all three sounds clearly.\n\n"
    
    if use_video:
        check_text_content += "A silent video will play during the experiment.\n\n"
    
    check_text_content += "Press SPACE to begin sound check."
    
    check_text = visual.TextStim(
        win=win, name='sound_check',
        text=check_text_content,
        font='Arial', pos=(0, 0), height=0.06,
        wrapWidth=1.5, ori=0, color='white',
        colorSpace='rgb', opacity=1, depth=0.0
    )
    
    # Show instructions
    check_text.draw()
    win.flip()
    event.waitKeys(keyList=['space'])
    
    # Play each sound with accompanying visual label
    for stim_type in ['N', 'N2', 'Ai']:
        label = visual.TextStim(
            win=win, name=f'label_{stim_type}',
            text=f"Playing sound: {stim_type}",
            font='Arial', pos=(0, 0), height=0.06,
            wrapWidth=None, ori=0, color='white',
            colorSpace='rgb', opacity=1, depth=0.0
        )
        
        label.draw()
        win.flip()
        
        # Play sound
        stimuli_dict[stim_type].play()
        core.wait(1.0)  # Wait for sound to finish
        
        # Brief pause between sounds
        win.flip()
        core.wait(0.5)
    
    # Ask for confirmation
    confirm_text = visual.TextStim(
        win=win, name='confirm',
        text="Did you hear all three sounds clearly?\n\n"
             "Press Y to continue to the experiment\n"
             "Press N to adjust your volume and repeat the check",
        font='Arial', pos=(0, 0), height=0.06,
        wrapWidth=1.5, ori=0, color='white',
        colorSpace='rgb', opacity=1, depth=0.0
    )
    
    confirm_text.draw()
    win.flip()
    
    keys = event.waitKeys(keyList=['y', 'n', 'escape'])
    
    if 'escape' in keys:
        win.close()
        core.quit()
    elif 'n' in keys:
        # Allow volume adjustment and repeat check
        adjust_text = visual.TextStim(
            win=win, name='adjust',
            text="Please adjust your volume.\n\n"
                 "Press SPACE when ready for another sound check.",
            font='Arial', pos=(0, 0), height=0.06,
            wrapWidth=1.5, ori=0, color='white',
            colorSpace='rgb', opacity=1, depth=0.0
        )
        
        adjust_text.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
        
        # Use iterative approach instead of recursion to avoid stack issues
        run_sound_check(win, stimuli_dict, visual_elements)

def run_block(win, block_num, sequence, trigger_sequence, stimuli_dict, visual_elements, use_video):
    """
    Run a single block of the MMN experiment.
    
    Parameters:
    - win: PsychoPy window
    - block_num: Block number (1-indexed)
    - sequence: List of stimulus types for this block
    - trigger_sequence: List of EEG trigger codes for this block
    - stimuli_dict: Dictionary of loaded audio stimuli
    - visual_elements: Dictionary of visual elements
    """
    # Calculate estimated time for this block
    trial_time_ms = stimulus_duration_ms + isi_ms
    if deviant_delay_ms > 0:
        # Roughly estimate number of deviants
        num_deviants = int(len(sequence) * DEVIANT_PROBABILITY)
        trial_time_ms += (deviant_delay_ms * num_deviants) / len(sequence)
    
    block_time_s = (trial_time_ms * len(sequence)) / 1000
    
    # Start the video if it's enabled
    if use_video and 'video' in visual_elements:
        visual_elements['video'].play()
    
    # Draw initial display (either video or fixation cross)
    if use_video and 'video' in visual_elements:
        visual_elements['video'].draw()
    else:
        visual_elements['fixation'].draw()
    
    visual_elements['block_progress'].setText(f"Block {block_num}/{num_blocks}")
    visual_elements['block_progress'].draw()
    win.flip()
    
    # Send block start trigger
    send_trigger(TRIGGER_CODES['block_start'])
    
    # Record block start time
    block_start_time = core.getTime()
    
    # Wait for a moment before starting trials
    core.wait(2.0)
    
    # Determine the standard type for this block (most common stimulus)
    standard_type = max(set(sequence), key=sequence.count)
    
    # Run trials
    for trial_idx, (stim_type, trigger) in enumerate(zip(sequence, trigger_sequence)):
        # Progress update (not shown to participant, just for debugging)
        if trial_idx % 50 == 0 or debug_mode:
            print(f"Block {block_num}, Trial {trial_idx+1}/{len(sequence)}")
            
            if debug_mode and trial_idx > 0:
                elapsed_time = core.getTime() - block_start_time
                trials_completed = trial_idx
                trials_remaining = len(sequence) - trial_idx
                time_per_trial = elapsed_time / trials_completed
                estimated_remaining = time_per_trial * trials_remaining
                
                print(f"Elapsed time: {elapsed_time:.1f}s, Estimated remaining: {estimated_remaining:.1f}s")
        
        # Draw either video or fixation
        if use_video and 'video' in visual_elements:
            # Update and draw the video frame
            visual_elements['video'].draw()
        else:
            visual_elements['fixation'].draw()
        
        # Show trial counter and time estimate in debug mode
        if debug_mode and show_trial_count:
            display_text = f"Trial {trial_idx+1}/{len(sequence)}\nStimulus: {stim_type}"
            
            if show_time_estimate and trial_idx > 5:  # After a few trials to get accurate estimate
                elapsed_time = core.getTime() - block_start_time
                trials_completed = trial_idx
                trials_remaining = len(sequence) - trial_idx
                time_per_trial = elapsed_time / trials_completed
                estimated_remaining = time_per_trial * trials_remaining
                
                display_text += f"\nEst. remaining: {estimated_remaining:.1f}s"
            

            trial_text = visual.TextStim(
                win=win,
                text=display_text,
                font='Arial',
                pos=(0, -0.3),
                height=0.03,
                color='white'
            )
            trial_text.draw()
        
        win.flip()
        
        # Send stimulus trigger and play sound
        send_trigger(trigger)
        stimuli_dict[stim_type].play()
        
        # Wait for stimulus duration
        core.wait(stimulus_duration_ms / 1000.0)
        
        # Determine if this was a deviant
        is_deviant = stim_type != standard_type

        # Regular inter-stimulus interval with video or blank screen
        if use_video and 'video' in visual_elements:
            visual_elements['video'].draw()
        else:
            # Blank screen - don't draw anything
            pass
        
        # Regular inter-stimulus interval
        win.flip()  # Blank screen
        core.wait(isi_ms / 1000.0)
        
        # Add extra delay after deviants if configured
        if is_deviant and deviant_delay_ms > 0:
            core.wait(deviant_delay_ms / 1000.0)
            
        # Check for quit request
        if event.getKeys(['escape']):
            win.close()
            core.quit()
    
    # Send block end trigger
    send_trigger(TRIGGER_CODES['block_end'])
    
    # Record and print actual block duration
    block_duration = core.getTime() - block_start_time
    print(f"Block {block_num} completed in {block_duration:.2f}s (estimated: {block_time_s:.2f}s)")
    
    return block_duration

def run_experiment():
    """
    Main function to run the entire experiment.
    """
    global use_video
    # Create organized folder structure
    folders = create_folder_structure(
        participant_id=exp_info['participant_id'],
        date_str=exp_info['date'],
        data_folder=DATA_FOLDER
    )
    
    # Update filenames to use the new folder structure
    base_filename = os.path.join(folders['participant'], f"p{exp_info['participant_id']}")
    csv_base_filename = os.path.join(folders['csv'], f"p{exp_info['participant_id']}")
    stats_filename = os.path.join(folders['stats'], f"p{exp_info['participant_id']}")
    plots_filename = os.path.join(folders['plots'], f"p{exp_info['participant_id']}")
    
    # Generate stimulus sequences for all blocks
    blocks, trigger_blocks = create_experiment_blocks(
        num_blocks=num_blocks,
        total_trials_per_block=TOTAL_TRIALS_PER_BLOCK, 
        deviant_probability=DEVIANT_PROBABILITY, 
        trigger_codes=TRIGGER_CODES,
        min_standards=min_standards,
        max_standards=max_standards
    )
    
    # Export sequences to CSV for record-keeping (will be in the csv_files subfolder)
    csv_files = export_sequences_to_csv(blocks, trigger_blocks, csv_base_filename)
    
    # Create a combined CSV file in the main participant folder
    combined_csv = combine_csv_files(csv_files, os.path.join(folders['participant'], f"p{exp_info['participant_id']}_all_trials.csv"))
    
    # Analyze and validate sequences
    sequence_stats = analyze_sequences(blocks, exp_info)
    
    # Add experiment parameters to sequence stats
    sequence_stats['experiment_parameters'] = {
        'isi_ms': isi_ms,
        'stimulus_duration_ms': stimulus_duration_ms,
        'block_break_duration_s': block_break_duration_s,
        'deviant_delay_ms': deviant_delay_ms,
        'random_seed': exp_info['used_random_seed'],
        'total_trials_per_block': TOTAL_TRIALS_PER_BLOCK,
        'min_standards': min_standards,
        'max_standards': max_standards,
        'deviant_probability': DEVIANT_PROBABILITY,
        'num_blocks': num_blocks,
    }
    
    # Save sequence statistics as JSON
    with open(f"{stats_filename}_sequence_stats.json", 'w') as f:
        json.dump(sequence_stats, f, indent=2)
    
    # Visualize sequences if requested
    if view_sequence:
        plot_files = []
        for i, block in enumerate(blocks):
            plot_file = visualize_sequence(block, i+1, plots_filename)
            plot_files.append(plot_file)
        
        # Show the first plot (can't show all at once in script mode)
        if plot_files:
            plt.show()
    
    # Load stimuli
    stimuli_dict = load_stimuli(stimuli_paths)
    
    # Initialize experiment window
    win, visual_elements, use_video = initialize_experiment_window(use_video, video_path, DEFAULT_VIDEO)
    
    # Run sound check if enabled
    if sound_check:
        run_sound_check(win, stimuli_dict, visual_elements, use_video)
    
    # Display instructions and wait for key press
    visual_elements['instructions'].draw()
    win.flip()
    event.waitKeys(keyList=['space'])
    
    # Run each block
    for block_idx in range(len(blocks)):
        # Run the block
        run_block(win, block_idx+1, blocks[block_idx], trigger_blocks[block_idx], 
                 stimuli_dict, visual_elements, use_video)
        
        # Break between blocks (unless it's the last block)
        if block_idx < len(blocks) - 1:
            # Store video position if video is enabled
            video_pos = None
            if use_video and 'video' in visual_elements:
                try:
                    # Store current position
                    video_pos = visual_elements['video'].getCurrentFrameTime()
                    # Pause the video during break
                    visual_elements['video'].pause()
                except Exception as e:
                    print(f"Could not store video position: {e}")
                    
            # Display break message
            visual_elements['break_text'].draw()
            win.flip()
            
            # Wait for break duration or space key press
            timer = core.CountdownTimer(block_break_duration_s)
            while timer.getTime() > 0:
                # Update countdown every second
                remaining = int(timer.getTime())
                if remaining % 5 == 0:  # Update every 5 seconds
                    visual_elements['break_text'].setText(
                        f"Take a short break.\n\n"
                        f"The experiment will continue in {remaining} seconds.\n\n"
                        f"(Or press SPACE to continue now)"
                    )
                    visual_elements['break_text'].draw()
                    win.flip()
                
                # Check for space key to continue early
                if event.getKeys(['space']):
                    break
                
                # Small wait to avoid busy loop
                core.wait(0.1)
            
            # Resume video from previous position if applicable
            if use_video and 'video' in visual_elements:
                try:
                    if video_pos is not None:
                        visual_elements['video'].seek(video_pos)
                    visual_elements['video'].play()
                    print(f"Resuming video playback from position {video_pos:.2f}s")
                except Exception as e:
                    print(f"Could not resume video: {e}")
    
    # Show completion message
    visual_elements['completion'].draw()
    win.flip()
    core.wait(3.0)
    
    # Clean up
    win.close()
    core.quit()

if __name__ == '__main__':
    run_experiment()