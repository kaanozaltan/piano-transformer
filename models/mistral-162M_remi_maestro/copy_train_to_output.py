from pathlib import Path
import shutil

src_dir = Path(
    "/hpcwork/lect0148/experiments/mistral-162M_remi_maestro_v1/data_processed/maestro_train"
)
dst_dir = Path("/hpcwork/lect0148/experiments/mistral-162M_remi_maestro_v1/output/train")

dst_dir.mkdir(parents=True, exist_ok=True)

for i, file in enumerate(src_dir.rglob("*.midi")):
    if file.is_file():
        new_name = f"{i}_prompt.midi"  # keeps original filename and extension
        target_file = dst_dir / new_name
        shutil.copy2(file, target_file)
