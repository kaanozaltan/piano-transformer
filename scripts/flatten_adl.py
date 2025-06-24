from pathlib import Path
import symusic

GENRE_TOKENS = [
    "Ambient", "Blues", "Children", "Classical", "Country",
    "Electronic", "Folk", "Jazz", "Latin", "Pop", "Rap",
    "Reggae", "Religious", "Rock", "Soul", "Soundtracks", "World"
]

def preprocess_midi_with_genre_in_track_name(source_dir: Path, target_dir: Path):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all MIDI files with common extensions
    midi_files = list(source_dir.glob("**/*.mid"))
    
    processed_count = 0
    
    for midi_file in midi_files:
        try:
            # Extract genre from folder path
            relative_path = midi_file.relative_to(source_dir)
            genre = "Unknown"
            
            if relative_path.parts:
                potential_genre = relative_path.parts[0]
                for genre_token in GENRE_TOKENS:
                    if genre_token.lower() == potential_genre.lower():
                        genre = genre_token
                        break
            
            # Load and modify MIDI file
            score = symusic.Score(midi_file)
            
            for i, track in enumerate(score.tracks):
                old_name = track.name if track.name else f"Track_{i}"
                track.name = f"{genre}_{old_name}"
            
            # Create unique target filename
            target_path = target_dir / midi_file.name
            counter = 1
            original_target = target_path
            while target_path.exists():
                stem = original_target.stem
                suffix = original_target.suffix
                target_path = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            score.dump_midi(target_path)
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} files...")
            
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")
    
    print(f"Total files processed: {processed_count}")

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    # Go one level up to the project root
    project_root = script_dir.parent
    
    # Set paths relative to project root
    source_directory = project_root / "data" / "adl-piano-midi"
    target_directory = project_root / "data" / "adl-piano-midi-processed"
    
    preprocess_midi_with_genre_in_track_name(source_directory, target_directory)