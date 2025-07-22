import json
from pathlib import Path
from collections import Counter, defaultdict
import symusic as sm
from typing import Dict, List
from tqdm import tqdm
import random


def analyze_genres(metadata_path: Path):
    """Print all genres and their counts."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    genre_counter = Counter()
    for item in metadata.values():
        if 'metadata' in item and 'genre' in item['metadata']:
            genre = item['metadata']['genre']
            if isinstance(genre, list):
                genre_counter.update(genre)
            elif isinstance(genre, str):
                genre_counter[genre] += 1
        else:
            genre_counter["none"] += 1
    
    print(f"Found {len(genre_counter)} unique genres:")
    for genre, count in genre_counter.most_common():
        print(f"  {genre}: {count}")


def flatten_dataset(source_path: Path, output_path: Path):
    """Flatten dataset with genre embedded in track names."""
    metadata_path = source_path / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    data_path = source_path / "data"
    midi_files = list(data_path.rglob("*.mid")) + list(data_path.rglob("*.midi"))
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed = 0
    skipped = 0
    
    for midi_file in tqdm(midi_files, desc="Processing MIDI files"):
        base_num = midi_file.name.split('_')[0].lstrip('0') or "0"
        
        # Get genre from metadata
        genre = "Unknown"
        if base_num in metadata and 'metadata' in metadata[base_num] and 'genre' in metadata[base_num]['metadata']:
            g = metadata[base_num]['metadata']['genre']
            genre = g[0] if isinstance(g, list) else g
        
        # Process MIDI file
        try:
            midi = sm.Score.from_file(str(midi_file))
            for track in midi.tracks:
                track.name = f"{genre.capitalize()}_{track.name}" if track.name else f"{genre.capitalize()}_Track"
            
            output_file = output_path / midi_file.name
            midi.dump_midi(str(output_file))
            processed += 1
        except Exception as e:
            tqdm.write(f"Error with {midi_file}: {e}")
            skipped += 1
    
    print(f"Done! Processed {processed} files, skipped {skipped} invalid files")


def create_balanced_dataset(source_path: Path, output_path: Path, files_per_genre: int = 1000):
    """Create a balanced dataset with equal numbers of files from classical, pop, soundtrack, and jazz."""
    target_genres = ["classical", "pop", "soundtrack", "jazz"]
    
    metadata_path = source_path / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    data_path = source_path / "data"
    midi_files = list(data_path.rglob("*.mid")) + list(data_path.rglob("*.midi"))
    
    # Group files by genre
    genre_files = defaultdict(list)
    
    for midi_file in tqdm(midi_files, desc="Grouping files by genre"):
        base_num = midi_file.name.split('_')[0].lstrip('0') or "0"
        
        if base_num in metadata and 'metadata' in metadata[base_num] and 'genre' in metadata[base_num]['metadata']:
            g = metadata[base_num]['metadata']['genre']
            genre = g[0].lower() if isinstance(g, list) else g.lower()
            
            # Simple matching for target genres
            for target in target_genres:
                if target in genre:
                    genre_files[target].append(midi_file)
                    break
    
    # Print stats and determine how many files we can use
    min_files = min(len(genre_files[g]) for g in target_genres)
    actual_files = min(files_per_genre, min_files)
    
    print(f"Available files per genre: {[len(genre_files[g]) for g in target_genres]}")
    print(f"Using {actual_files} files per genre")
    
    # Create balanced dataset
    output_path.mkdir(parents=True, exist_ok=True)
    
    for genre in target_genres:
        selected = random.sample(genre_files[genre], actual_files)
        
        for midi_file in tqdm(selected, desc=f"Processing {genre}"):
            try:
                midi = sm.Score.from_file(str(midi_file))
                for track in midi.tracks:
                    track.name = f"{genre.capitalize()}_{track.name}" if track.name else f"{genre.capitalize()}_Track"
                
                output_file = output_path / f"{genre}_{midi_file.name}"
                midi.dump_midi(str(output_file))
            except Exception as e:
                print(f"Error with {midi_file}: {e}")
    
    print(f"Created balanced dataset with {actual_files * len(target_genres)} files total")


def main():
    source_path = Path("data/aria-midi-v1-deduped-ext")
    output_path = Path("data/aria-midi-flattened")
    balanced_output_path = Path("data/aria-midi-genre-balanced")
    
    # Analyze genres
    analyze_genres(source_path / "metadata.json")
    
    # Create balanced dataset
    print("\nCreating balanced dataset...")
    create_balanced_dataset(source_path, balanced_output_path, files_per_genre=20000)
    
    # Uncomment to also create the flattened dataset
    # flatten_dataset(source_path, output_path)


if __name__ == "__main__":
    main()