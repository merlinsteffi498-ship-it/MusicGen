import os
import pretty_midi
import numpy as np
import pandas as pd

def parse_midi_file(file_path):
    """
    Reads a MIDI file and extracts the tempo, note density, and a sequence of notes.
    """
    try:
        # Load the MIDI file
        midi_data = pretty_midi.PrettyMIDI(file_path)
        
        # 1. Extract BPM (Tempo)
        # pretty_midi returns an array of tempos and their timestamps. We take the first one.
        tempos = midi_data.get_tempo_changes()
        bpm = tempos[1][0] if len(tempos[1]) > 0 else 120.0 
        
        # 2. Extract Notes and calculate Note Density
        all_notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum: # Skip drums for the melodic engine
                for note in instrument.notes:
                    # We store: [Pitch (0-127), Velocity (0-127), Duration in seconds]
                    duration = note.end - note.start
                    all_notes.append([note.pitch, note.velocity, duration])
        
        # Sort notes by when they are played (though this basic loop grabs them mostly in order)
        total_time = midi_data.get_end_time()
        note_density = len(all_notes) / total_time if total_time > 0 else 0
        
        return {
            'bpm': bpm,
            'density': note_density,
            'sequence': all_notes
        }
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def main():
    print("Testing the Lakh MIDI Parser...")
    
    # NOTE: We will test this on a single MIDI file first before running the whole dataset
    # Replace 'test_song.mid' with the path to one of your Lakh MIDI files
    test_file = 'test_song.mid' 
    
    if os.path.exists(test_file):
        parsed_data = parse_midi_file(test_file)
        if parsed_data:
            print(f"\n--- Extraction Successful ---")
            print(f"Detected BPM: {parsed_data['bpm']:.2f}")
            print(f"Detected Note Density: {parsed_data['density']:.2f} notes/second")
            print(f"Total Notes Extracted: {len(parsed_data['sequence'])}")
            print(f"First 5 Notes (Pitch, Velocity, Duration): \n{parsed_data['sequence'][:5]}")
    else:
        print(f"Could not find '{test_file}'. Please add a MIDI file to the folder to test.")

if __name__ == "__main__":
    main()