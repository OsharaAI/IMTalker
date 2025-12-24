conda create -n IMTalker python=3.10
conda activate IMTalker
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
conda install -c conda-forge ffmpeg 
pip install -r requirement.txt

git clone https://huggingface.co/cbsjtu01/IMTalker
cd IMTalker
git lfs install
git lfs pull

cd ..
mv IMTalker/renderer.ckpt checkpoints/
mv IMTalker/generator.ckpt checkpoints/
mv IMTalker/wav2vec2-base-960h/ checkpoints/

PYTHONPATH=. python generator/generate.py \
    --ref_path "./assets/chandler.jpg" \
    --aud_path "./assets/audio_2.wav" \
    --res_dir "./results/" \
    --generator_path "./checkpoints/generator.ckpt" \
    --renderer_path "./checkpoints/renderer.ckpt" \
    --a_cfg_scale 3 \
    --crop

pip install psutil