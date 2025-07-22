import os
from mido import MidiFile

FOLDER_PATH = "/path/to/your/midi/folder"

def is_valid_midi(file_path):
    try:
        MidiFile(file_path)
        return True
    except Exception as e:
        print(f"[DELETING INVALID MIDI] {file_path} - {e}")
        return False

def delete_invalid_midis(folder_path):
    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
        return

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.mid', '.midi')):
            full_path = os.path.join(folder_path, filename)
            if not is_valid_midi(full_path):
                try:
                    os.remove(full_path)
                    print(f"✅ Deleted: {filename}")
                except Exception as delete_error:
                    print(f"Could not delete {filename} - {delete_error}")

if __name__ == "__main__":
    delete_invalid_midis(FOLDER_PATH)

