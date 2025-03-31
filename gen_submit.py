import os

num_layers = 44
seeds = [17, 19, 37, 42]
epochs = [3, 4, 5, 6]

os.makedirs('sbatch_scripts', exist_ok=True)

submit_script_lines = []

for s in seeds:
    for e in epochs:
        filename = f'sbatch_scripts/job_L{num_layers}_E{e}_S{s}.sh'
        command = f"""#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:8
#SBATCH -c 32
#SBATCH --mem=512G
#SBATCH --job-name=DS1.5B-L{num_layers}-E{e}-S{s}
#SBATCH -o logs/L{num_layers}E{e}-{s}.o
#SBATCH -e logs/L{num_layers}E{e}-{s}.e
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenzhuoming911@gmail.com

nvidia-smi
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r1

MODEL=ckpts/L{num_layers}/E{e}
MODEL_ARGS=\"pretrained=$MODEL,dtype=bfloat16,data_parallel_size=8,seed={s},max_model_length=32768,gpu_memory_utilization=0.9,generation_parameters={{max_new_tokens:30768,temperature:0.6,top_p:0.95}}\"

OUTPUT_DIR=data/evals/L{num_layers}/E{e}/seed{s}

for TASK in amc23 aime24 gsm8k math_500; do
    lighteval vllm $MODEL_ARGS \"custom|$TASK|0|0\" \
        --custom-tasks src/open_r1/evaluate.py \
        --use-chat-template \
        --output-dir $OUTPUT_DIR
done
"""

        with open(filename, 'w') as file:
            file.write(command)

        submit_script_lines.append(f'sbatch {filename}')

with open('submit_all.sh', 'w') as submit_file:
    submit_file.write("#!/bin/bash\n")
    submit_file.write("\n".join(submit_script_lines))

print("Generated all sbatch scripts and submission script (submit_all.sh).")