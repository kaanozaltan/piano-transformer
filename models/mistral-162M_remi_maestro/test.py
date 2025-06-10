from pathlib import Path
from transformers import AutoModelForCausalLM
from miditok import REMI
from miditok.pytorch_data import DatasetMIDI
from piano_transformer.utils.evaluation import generate_and_save, evaluate

# Config
EXPERIMENT_NAME = "mistral-162M_remi_maestro_v1"
EXPERIMENT_DIR = Path("/hpcwork/lect0148/experiments") / EXPERIMENT_NAME

MODEL_PATH = EXPERIMENT_DIR / "model"
TOKENIZER_PATH = EXPERIMENT_DIR / "tokenizer.json"
TEST_DATA_DIR = EXPERIMENT_DIR / "data_processed" / "maestro_test"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs" / "test"


# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = REMI(params=TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).cuda().eval()

# Prepare test dataset
midi_files = sorted(TEST_DATA_DIR.glob("**/*.midi"))
dataset = DatasetMIDI(
    midi_files,
    tokenizer=tokenizer,
    max_seq_len=1024,
    bos_token_id=tokenizer["BOS_None"],
    eos_token_id=tokenizer["EOS_None"],
)

# Generate
print(f"Generating...")
generate_and_save(model, tokenizer, dataset, save_dir=OUTPUT_DIR, mode="eval")

# Evaluate
print("Evaluating...")
print(
    "Evaluation results:", 
    evaluate(ref_dir=TEST_DATA_DIR, gen_dir=OUTPUT_DIR)
)
