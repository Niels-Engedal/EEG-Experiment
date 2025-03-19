### Make sure the following packages are installed. Should work with python 3.8.18
# psychopy
# psychtoolbox

###Changes on testing day
#volume setting changed to 1
#triggers are no longer pulled down
#window flip after instructions

##TODO##
# Transposition trigger or not?
# Define length of experiment. Currently 10 trials at 1.2 seconds per chord correpsonds to 49 seconds

# load modules
from psychopy import visual, core, event, data, gui, sound, prefs
import psychtoolbox as ptb
import random
import ppc
from triggers import setParallelData

#set preferences
prefs.hardware['audioLib'] = 'PTB'
prefs.hardware['audioLatencyMode'] = 3
prefs.hardware['audioDevice'] = 14

# define window
win = visual.Window(fullscr = True, color = "black")

# Create a logfile
logfile = 'experiment_log.csv'
log_columns = ['Key_Transpose', 'Transposition', 'Stimulus_Type']
log_data = []

############## text ###################
# instructions
instruction = '''
Welcome to the Experiment!\n\n
In a moment you will hear sequences of chords. \n\n
Press any key to continue.
'''

goodbye = '''
The experiment is done. Thank you for your participation.'''

### Objects

# Tone to Hz dictionary
G2 = 97.999
A2 = 110.00
C3 = 130.81
C_3 = 138.59
D3 = 146.83
D_3 = 155.56
E3 = 164.81
F3 = 174.61
F_3 = 185.00
G3 = 196.00
G_3 = 207.65
A3 = 220.00
A_3 = 233.08
B3 = 246.94
C4 = 261.63
C_4 = 277.18
D4 = 293.66 
D_4 = 311.13
E4 = 329.63
F4 = 349.23
F_4 = 369.99
G4 = 392.00
G_4 = 415.30
A4 = 440.00
A_4 = 466.16
B4 = 493.88
C5 = 523.25
C_5 = 554.37
E5 = 659.26

# Chords - note that there are now multiple versions of each chord:
i   = E3 #[E3, C4, G4, C5]
iv	= F3 #[F3, C4, F4, A4]
v	= G3 #[G3, D4, B4, G4]
#v_alt = [G2, B3, G4, B4]
vi	= A3 #[A3, C4, E4, A4, E5]
viii = C3 #[C3, C4, E4, G4, C5]
viii_dissonant = A_3 #[C3, A_3, C4, F_4, C_5]

# Phrases (these are the stimuli categories to be sampled from):
Phrase_perfect_cad = [i, iv, v, viii]
Phrase_deceptive_cad = [i, iv, v, vi]
Phrase_dissonant_end = [i, iv, v, viii_dissonant]

######### Defining Functions #############
def transpition_trigger(key, direction):
    return 100 + (key * direction)

### Function for showing text and waiting for key press
def msg(txt):
    message = visual.TextStim(win, text = txt, alignText = "left", height = 0.05)
    message.draw()
    win.flip()
    event.waitKeys()

### Function for sampling from the three possible stimuli types:
def sample_with_probability(lists, probabilities):

    if len(lists) != len(probabilities):
        raise ValueError("Number of lists and probabilities must be the same.")
    
    # Normalize probabilities to sum up to 1
    total_prob = sum(probabilities)
    normalized_probabilities = [p / total_prob for p in probabilities]
    
    # Sample a list index according to the probabilities
    list_index = random.choices(range(len(lists)), weights=normalized_probabilities)[0]
    
    # Sample an element from the selected list
    selected_list = lists[list_index]
    sampled_element = random.choice(selected_list)
    
    return sampled_element # this corresponds to a single phrase

### Defining lists and probabilities for the sampler:
lists = [[Phrase_perfect_cad], [Phrase_deceptive_cad], [Phrase_dissonant_end]]
probabilities = [0.8, 0.1, 0.1]  # list 1: 80%, list 2: 10%, list 3: 10%

### Setting a transposition constant (if needed):
k = 1

#### Defining player function:
def play_chord(chord, time_interval_between_chords):

    # Get the current time
    now = 1 #ptb.GetSecs()
    
    # Create Sound objects for each tone in the chord
    sounds = sound.Sound(chord * k, secs = time_interval_between_chords, volume = 1)
    
    # Schedule all sounds to be played at the current time
    sounds.play(when=now+time_interval_between_chords)

######### New experiment playback section

######### begin the experiment ###########

# Show instructions:
msg(instruction)

win.flip()
### This loop samples a phrase, transposes it randomly, and plays the chords in the phrase.

# Parameters:
#n_trials = range(12) # Set the number of trial ~ 1 min(?)
n_trials = range(120) # Set the number of trial ~ 10 min(?)
time_interval_between_chords = 1.2 # Set timing between each chord

# Stimuli playback loop:
for trial in n_trials:
    
    # An escape function (for testing purposes):
    keys = event.getKeys()
    
    if 'escape' in keys:
        print("Experiment was quit using the esc key")
        core.quit()
    
    # Picking key transposition values:
    key_transpose = random.randint(0, 12)
    transpose_dir = random.choice([1, -1])
    print(key_transpose, transpose_dir)
    
    # Log key_transpose value
    log_data.append({'Key_Transpose': key_transpose/transpose_dir})
    
    # Sampling one of the phrases:
    sampled_phrase = sample_with_probability(lists, probabilities)
    print(sampled_phrase)

    # Tranposes the sampled phrase (only if the key_transpose drawn is > 0!):
    if key_transpose > 0:
        transposed_phrase = [chord * (key_transpose ** (transpose_dir * 1/12)) for chord in sampled_phrase]
    else: 
        transposed_phrase = sampled_phrase

    # Sending the chords in the transposed phrase to the player:
    for chord in transposed_phrase:

        # get trigger for type of phrase
        if sampled_phrase == Phrase_perfect_cad:
            print("Perfect cadence")
            stimulus_type = 'Perfect Cadence'
            phrase_trigger = 0

        elif sampled_phrase == Phrase_deceptive_cad:
            print("Deceptive cadence")
            stimulus_type = 'Deceptive cadence'
            phrase_trigger = 100

        else:
            print("Dissonant end")
            stimulus_type = 'Dissonant end'
            phrase_trigger = 200
        
        # Log the stimulus type
        log_data.append({'Stimulus_Type': stimulus_type})

        #get index of the chord
        chord_index = transposed_phrase.index(chord)

        #calculate trigger
        final_trigger = chord_index + 1 + phrase_trigger
        
        #send trigger
        setParallelData(final_trigger) 
        
        # play tones
        now = 1
        sounds = sound.Sound(chord * k, secs = time_interval_between_chords, volume = 0.2)
        sounds.play(when=now+time_interval_between_chords)
        
        #schedule chord to be played
        #play_chord(chord, time_interval_between_chords)
        

        #wait until chord should be playing
        core.wait(time_interval_between_chords)
        
        # log only every 5th chord to reduce log size
        log_data.append({'Transposition': chord})


# show goodbye message
msg(goodbye)

import csv

with open(logfile, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=log_columns)
    writer.writeheader()
    writer.writerows(log_data)
