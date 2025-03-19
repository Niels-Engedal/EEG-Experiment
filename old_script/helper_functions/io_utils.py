"""
Input/output utility functions for the MMN oddball experiment.
"""
import os
import pandas as pd
from psychopy import sound

def load_stimuli(stimuli_paths):
    """
    Load audio stimuli for the experiment.
    
    Parameters:
    - stimuli_paths: Dictionary mapping stimulus types to file paths
    
    Returns:
    - stimuli_dict: Dictionary mapping stimulus types to audio objects
    """
    stimuli_dict = {}
    
    # Check if stimulus files exist
    for stim_type, filepath in stimuli_paths.items():
        if not os.path.exists(filepath):
            print(f"Error: Stimulus file not found: {filepath}")
            return None
        else:
            stimuli_dict[stim_type] = sound.Sound(filepath)
    
    return stimuli_dict

def export_sequences_to_csv(blocks, trigger_blocks, filename_prefix):
    """
    Export the stimulus sequences to CSV files for record and analysis.
    
    Parameters:
    - blocks: List of block sequences
    - trigger_blocks: List of corresponding trigger sequences
    - filename_prefix: Prefix for the CSV filenames
    
    Returns:
    - csv_files: List of paths to the exported CSV files
    """
    csv_files = []
    
    for i, (block, triggers) in enumerate(zip(blocks, trigger_blocks)):
        # Determine the standard (most common stimulus)
        standard = max(set(block), key=block.count)
        
        # Get the deviant type (should be only one in our paradigm)
        deviant_types = set([b for b in block if b != standard])
        deviant = list(deviant_types)[0] if deviant_types else "unknown"
        
        # Determine condition (Control vs Experimental)
        if (standard == 'N' and deviant == 'N2') or (standard == 'N2' and deviant == 'N'):
            condition = 'Control'
        else:
            condition = 'Experimental'
        
        # Create a dataframe with the sequence data
        block_df = pd.DataFrame({
            'trial_num': range(1, len(block) + 1),
            'stimulus': block,
            'trigger_code': triggers,
            'is_deviant': [s != standard for s in block],
            'condition': condition,
            'standard_type': standard,
            'deviant_type': deviant,
            'block_num': i+1
        })
        
        # Add a column indicating whether this trial follows a deviant
        block_df['post_deviant'] = [False] + list(block_df['is_deviant'][:-1])
        
        # Export to CSV
        csv_filename = f"{filename_prefix}_block{i+1}.csv"
        block_df.to_csv(csv_filename, index=False)
        csv_files.append(csv_filename)
    
    return csv_files

def combine_csv_files(csv_files, output_filename):
    """
    Combine multiple CSV files into a single file.
    
    Parameters:
    - csv_files: List of CSV file paths to combine
    - output_filename: Path for the output combined CSV file
    
    Returns:
    - output_filename: Path to the combined CSV file
    """
    # Read all CSV files and concatenate
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    # Combine all dataframes
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv(output_filename, index=False)
        print(f"Created combined data file: {output_filename}")
        return output_filename
    else:
        print("No CSV files to combine.")
        return None

def create_folder_structure(participant_id, date_str, data_folder="data"):
    """
    Create an organized folder structure for experiment data.
    
    Parameters:
    - participant_id: Participant ID
    - date_str: Date string for the session
    - data_folder: Main data folder path
    
    Returns:
    - folders: Dictionary of created folder paths
    """
    # Main participant folder
    participant_folder = os.path.join(data_folder, f"p{participant_id}_{date_str}")
    
    # Subfolders
    csv_folder = os.path.join(participant_folder, "csv_files")
    plots_folder = os.path.join(participant_folder, "plots")
    stats_folder = os.path.join(participant_folder, "stats")
    
    # Create folders
    for folder in [participant_folder, csv_folder, plots_folder, stats_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # Return folder structure
    return {
        'participant': participant_folder,
        'csv': csv_folder,
        'plots': plots_folder,
        'stats': stats_folder
    }