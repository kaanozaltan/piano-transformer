import os
import time
from mido import MidiFile

FOLDER_PATH = "/hpcwork/yh522379/moonbeam/checkpoints/fine-tuned/relevant/ft_309M_peft_ctx1024_bs16_gradacc4_clipTrue_clipthresh1.0_lr1e-4_cosine_gamma0.99_temp1.1_ep150_20250807_155620/generation_results"

def is_valid_midi(file_path, verbose=True):
    """
    Check if a MIDI file is valid by attempting to load it.
    
    Args:
        file_path: Path to the MIDI file
        verbose: Whether to print detailed validation info
    
    Returns:
        tuple: (is_valid, error_message, midi_info)
    """
    try:
        start_time = time.time()
        midi_file = MidiFile(file_path)
        load_time = time.time() - start_time
        
        # Get MIDI file information
        num_tracks = len(midi_file.tracks)
        total_messages = sum(len(track) for track in midi_file.tracks)
        duration = midi_file.length
        
        midi_info = {
            'tracks': num_tracks,
            'messages': total_messages,
            'duration': round(duration, 2),
            'load_time': round(load_time * 1000, 2)  # in milliseconds
        }
        
        return True, None, midi_info
        
    except Exception as e:
        error_msg = str(e)
        if verbose:
            relative_path = os.path.relpath(file_path, FOLDER_PATH)
            print(f"INVALID: {relative_path} | Error: {error_msg}")
        return False, error_msg, None

def find_midi_files(folder_path):
    """
    Recursively find all MIDI files in folder and subdirectories.
    
    Args:
        folder_path: Root folder path to search
        
    Returns:
        list: List of full paths to MIDI files
    """
    midi_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                full_path = os.path.join(root, file)
                midi_files.append(full_path)
    
    return midi_files

def get_directory_stats(folder_path):
    """
    Get statistics about directory structure.
    
    Args:
        folder_path: Root folder path
        
    Returns:
        dict: Directory statistics
    """
    total_dirs = 0
    total_files = 0
    
    for root, dirs, files in os.walk(folder_path):
        total_dirs += len(dirs)
        total_files += len(files)
    
    return {
        'directories': total_dirs,
        'total_files': total_files
    }

def delete_invalid_midis(folder_path, verbose=True):
    """
    Process all MIDI files in a folder and subdirectories, validate them, and delete invalid ones.
    
    Args:
        folder_path: Path to the root folder containing MIDI files
        verbose: Whether to print detailed processing info
    """
    global FOLDER_PATH
    FOLDER_PATH = folder_path
    
    print("=" * 80)
    print("RECURSIVE MIDI FILE VALIDATION AND CLEANUP")
    print("=" * 80)
    
    # Check if folder exists
    if not os.path.isdir(folder_path):
        print(f"ERROR: Invalid folder path: {folder_path}")
        return
    
    print(f"Processing folder (recursive): {folder_path}")
    
    # Get directory statistics
    dir_stats = get_directory_stats(folder_path)
    print(f"Directory structure: {dir_stats['directories']} subdirectories, {dir_stats['total_files']} total files")
    
    # Find all MIDI files recursively
    print("Scanning for MIDI files in all subdirectories...")
    midi_files = find_midi_files(folder_path)
    
    print(f"Found {len(midi_files)} MIDI files across all directories")
    
    if not midi_files:
        print("No MIDI files found to process.")
        return
    
    # Show directory distribution
    if verbose:
        dir_counts = {}
        for midi_file in midi_files:
            dir_name = os.path.dirname(os.path.relpath(midi_file, folder_path))
            if dir_name == "":
                dir_name = "." # root directory
            dir_counts[dir_name] = dir_counts.get(dir_name, 0) + 1
        
        print(f"\nMIDI files distribution:")
        for dir_name, count in sorted(dir_counts.items()):
            print(f"   {dir_name}: {count} files")
    
    # Initialize counters
    valid_count = 0
    invalid_count = 0
    deleted_count = 0
    error_count = 0
    
    valid_files = []
    invalid_files = []
    
    print(f"\nStarting validation of {len(midi_files)} MIDI files...")
    print("-" * 80)
    
    # Process each MIDI file
    for i, file_path in enumerate(midi_files, 1):
        is_valid, error_msg, midi_info = is_valid_midi(file_path, verbose)
        
        if is_valid:
            valid_count += 1
            valid_files.append((file_path, midi_info))
        else:
            invalid_count += 1
            invalid_files.append((file_path, error_msg))
            
            # Try to delete invalid file
            try:
                os.remove(file_path)
                deleted_count += 1
                if verbose:
                    relative_path = os.path.relpath(file_path, folder_path)
                    print(f"    Successfully deleted: {relative_path}")
            except Exception as delete_error:
                error_count += 1
                relative_path = os.path.relpath(file_path, folder_path)
                print(f"    Could not delete {relative_path}: {delete_error}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Root folder: {folder_path}")
    print(f"Total MIDI files processed: {len(midi_files)}")
    print(f"Valid MIDI files: {valid_count}")
    print(f"Invalid MIDI files: {invalid_count}")
    print(f"Files deleted: {deleted_count}")
    print(f"Deletion errors: {error_count}")
    print(f"Success rate: {valid_count/len(midi_files)*100:.1f}%")
    
    if invalid_files:
        print(f"\nINVALID FILES REMOVED:")
        for file_path, error in invalid_files:
            relative_path = os.path.relpath(file_path, folder_path)
            print(f"   {relative_path}: {error}")
    
    print("=" * 80)
    print("Processing complete!")

if __name__ == "__main__":
    delete_invalid_midis(FOLDER_PATH, verbose=True)