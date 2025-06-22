import subprocess
from pathlib import Path

project_root = Path().resolve()

wav_path = project_root / "output" / "test_jonathan_wav"
soundfont_path = project_root / "assets" / "SalC5Light2.sf2"
midi_dir = project_root / "output" / "test_jonathan"

for midi_file in midi_dir.glob("*.midi"):

    wav_file = wav_path / (midi_file.stem + ".wav")

    command = [
                "fluidsynth",
                "-ni",
                "-F", str(wav_file),
                "-r", "44100",
                str(soundfont_path),
                str(midi_file),
            ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ Converted {midi_file.name} -> {wav_file.name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting {midi_file.name}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)