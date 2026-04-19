import argparse
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_path", type=str, required=True)
    p.add_argument("output_path", type=str, required=True)
    p.add_argument("soundfont_path", type=str, default="assets/SalC5Light2.sf2")
    p.add_argument("--quiet", action="store_false", default=True)
    args = p.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    def convert_file(midi_file):
        wav_filename = midi_file.stem + ".wav"
        wav_path = output_path / wav_filename

        command = [
            "fluidsynth",
            "-ni",
            *(["-q"] if args.quiet else []),
            args.soundfont_path,
            str(midi_file),
            "-F",
            str(wav_path),
            "-r",
            "44100",
        ]

        subprocess.run(command, check=True)

    if input_path.is_dir():
        for midi_file in input_path.iterdir():
            if midi_file.is_file() and midi_file.suffix.lower() == ".midi":
                convert_file(midi_file)
    elif input_path.is_file() and input_path.suffix.lower() == ".midi":
        convert_file(input_path)
    else:
        raise ValueError(
            "Input path must be a .midi file or a directory containing .midi files"
        )


if __name__ == "__main__":
    main()
