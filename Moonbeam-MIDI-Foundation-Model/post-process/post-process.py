import os
import time
from mido import MidiFile

FOLDER_PATH = "/hpcwork/yh522379/moonbeam/checkpoints/fine-tuned/ft_839M_peft_ctx512_bs32_lr1e-4_cosine_gamma0.99_temp1.1_ep150_20250803_045343/137-20.safetensors/moonbeam_839M/from_scratch/temperature_1.1_top_p_0.95_genlen_256"

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
        
        if verbose:
            file_size = round(os.path.getsize(file_path) / 1024, 2)  # KB
            print(f"✅ VALID: {os.path.basename(file_path)} | {num_tracks} tracks | {total_messages} msgs | {duration:.1f}s | {file_size}KB | {load_time*1000:.1f}ms")
        
        return True, None, midi_info
        
    except Exception as e:
        error_msg = str(e)
        if verbose:
            file_size = round(os.path.getsize(file_path) / 1024, 2) if os.path.exists(file_path) else 0
            print(f"❌ INVALID: {os.path.basename(file_path)} | {file_size}KB | Error: {error_msg}")
        return False, error_msg, None

def delete_invalid_midis(folder_path, verbose=True):
    """
    Process all MIDI files in a folder, validate them, and delete invalid ones.
    
    Args:
        folder_path: Path to the folder containing MIDI files
        verbose: Whether to print detailed processing info
    """
    print("=" * 80)
    print("🎵 MIDI FILE VALIDATION AND CLEANUP")
    print("=" * 80)
    
    # Check if folder exists
    if not os.path.isdir(folder_path):
        print(f"❌ ERROR: Invalid folder path: {folder_path}")
        return
    
    print(f"📁 Processing folder: {folder_path}")
    
    # Find all MIDI files
    all_files = os.listdir(folder_path)
    midi_files = [f for f in all_files if f.lower().endswith(('.mid', '.midi'))]
    
    print(f"📊 Found {len(midi_files)} MIDI files out of {len(all_files)} total files")
    
    if not midi_files:
        print("ℹ️  No MIDI files found to process.")
        return
    
    # Initialize counters
    valid_count = 0
    invalid_count = 0
    deleted_count = 0
    error_count = 0
    
    valid_files = []
    invalid_files = []
    
    print(f"\n🔍 Starting validation of {len(midi_files)} MIDI files...")
    print("-" * 80)
    
    # Process each MIDI file
    for i, filename in enumerate(midi_files, 1):
        full_path = os.path.join(folder_path, filename)
        
        if verbose:
            print(f"[{i:3d}/{len(midi_files)}] ", end="")
        
        is_valid, error_msg, midi_info = is_valid_midi(full_path, verbose)
        
        if is_valid:
            valid_count += 1
            valid_files.append((filename, midi_info))
        else:
            invalid_count += 1
            invalid_files.append((filename, error_msg))
            
            # Try to delete invalid file
            try:
                os.remove(full_path)
                deleted_count += 1
                if verbose:
                    print(f"    🗑️  Successfully deleted: {filename}")
            except Exception as delete_error:
                error_count += 1
                print(f"    ⚠️  Could not delete {filename}: {delete_error}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📋 PROCESSING SUMMARY")
    print("=" * 80)
    print(f"📁 Folder: {folder_path}")
    print(f"📊 Total files processed: {len(midi_files)}")
    print(f"✅ Valid MIDI files: {valid_count}")
    print(f"❌ Invalid MIDI files: {invalid_count}")
    print(f"🗑️  Files deleted: {deleted_count}")
    print(f"⚠️  Deletion errors: {error_count}")
    print(f"📈 Success rate: {valid_count/len(midi_files)*100:.1f}%")
    
    if valid_files:
        print(f"\n🎵 VALID FILES STATISTICS:")
        total_duration = sum(info['duration'] for _, info in valid_files)
        avg_tracks = sum(info['tracks'] for _, info in valid_files) / len(valid_files)
        avg_messages = sum(info['messages'] for _, info in valid_files) / len(valid_files)
        avg_duration = total_duration / len(valid_files)
        
        print(f"   Total music duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        print(f"   Average tracks per file: {avg_tracks:.1f}")
        print(f"   Average messages per file: {avg_messages:.0f}")
        print(f"   Average duration per file: {avg_duration:.1f} seconds")
    
    if invalid_files:
        print(f"\n❌ INVALID FILES DETAILS:")
        for filename, error in invalid_files:
            print(f"   • {filename}: {error}")
    
    print("=" * 80)
    print("✅ Processing complete!")

if __name__ == "__main__":
    delete_invalid_midis(FOLDER_PATH, verbose=True)

