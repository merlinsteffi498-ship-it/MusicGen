import pretty_midi

def main():
    print("Generating a test MIDI file...")
    # Initialize a new MIDI file with a tempo of 120 BPM
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    
    # Create an Acoustic Grand Piano instrument
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    # Add 4 simple notes (C, D, E, F)
    notes = [
        pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5),
        pretty_midi.Note(velocity=100, pitch=62, start=0.5, end=1.0),
        pretty_midi.Note(velocity=100, pitch=64, start=1.0, end=1.5),
        pretty_midi.Note(velocity=100, pitch=65, start=1.5, end=2.0)
    ]
    
    # Add the notes to the piano, and the piano to the MIDI file
    piano.notes.extend(notes)
    midi.instruments.append(piano)
    
    # Save the file
    midi.write('test_song.mid')
    print("Success! 'test_song.mid' has been saved to your folder.")

if __name__ == "__main__":
    main()