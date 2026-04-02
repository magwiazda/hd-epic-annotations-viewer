#!/bin/bash
#SBATCH --job-name="twin_coffee_machine"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --mail-type=None
#SBATCH --output=/usr/prakt/s0021/slurm/logs/slurm-%j.out

set -e
source /usr/prakt/s0021/miniconda3/etc/profile.d/conda.sh

echo '=== Step 2: SAM-3D-Objects reconstruction ==='
conda run -n sam3d-objects python /usr/prakt/s0021/EH/hd-epic-annotations-viewer/run_sam3d.py \
    --image   /usr/prakt/s0021/EH/hd-epic-annotations-viewer/digital_twins/P01-20240202-161948/frame682_coffee_machine/frame.jpg \
    --mask    /usr/prakt/s0021/EH/hd-epic-annotations-viewer/digital_twins/P01-20240202-161948/frame682_coffee_machine/mask.png \
    --out-dir /usr/prakt/s0021/EH/hd-epic-annotations-viewer/digital_twins/P01-20240202-161948/frame682_coffee_machine

echo '=== Done ==='
