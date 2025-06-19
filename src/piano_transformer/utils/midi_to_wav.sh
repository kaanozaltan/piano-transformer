#!/bin/bash

# Just a bash wrapper for our midi2wav
# Usage:
# ./midi2wav.sh input_path output_path soundfont_file [--no-quiet]

# Parse arguments
INPUT_PATH="Moonbeam-MIDI-Foundation-Model/checkpoints/fine-tuned/fine-tuned_3_epoch_839M/0-370/moonbeam_839M.pt/temperature_0.9_top_p_0.95"
OUTPUT_PATH="Moonbeam-MIDI-Foundation-Model/waves/"
SOUNDFONT="assets/SalC5Light2.sf2"
QUIET="True"

if [[ "$4" == "--no-quiet" ]]; then
    QUIET="False"
fi

# Set PYTHONPATH to the root of your repo so Python can find the package
export PYTHONPATH="/home/yh522379/lab/piano-transformer"

python3 -c "
from src.piano_transformer.utils.midi import midi2wav
midi2wav('$INPUT_PATH', '$OUTPUT_PATH', '$SOUNDFONT', quiet=$QUIET)
"
