git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
uv sync --extra torch --extra metrics --prerelease=allow
uv pip install deepspeed==0.14.4
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp311-cp311-linux_x86_64.whl

cd ../
rm -rf LLaMA-Factory/data/

uv run --with datasets scripts/prepare_dataset.py 
mv ../configs/config.yaml LLaMA-Factory/

uv add wandb

uv run wandb login
export WANDB_PROJECT="partial-edits"

# uv run huggingface-cli login

cd LLaMA-Factory
# uv run --prerelease=allow llamafactory-cli train config.yaml