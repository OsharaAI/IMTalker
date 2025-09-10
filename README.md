
# IMTalk: Implicit Motion Learning for Efficient Audio-driven Talking Face Generation


## Environment
```
conda create -n IMTalk python=3.10.18 
conda activate IMTalk
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirement.txt
```

## Weight
```
./checkpoints
|-- imf.ckpt                                       # IMF model
|-- fm.ckpt                                        # flow matching model
|-- wav2vec2-base-960h/                             # audio encoder
|   |-- .gitattributes
|   |-- config.json
|   |-- feature_extractor_config.json
|   |-- model.safetensors
|   |-- preprocessor_config.json
|   |-- pytorch_model.bin
|   |-- README.md
|   |-- special_tokens_map.json
|   |-- tf_model.h5
|   |-- tokenizer_config.json
|   '-- vocab.json
```
## Download
imf.ckpt fm.ckpt: 

百度网盘：链接:https://pan.baidu.com/s/1qNR8qSkZ7jaP3NU6XS03Uw?pwd=rwp4 提取码:rwp4 复制这段内容后打开百度网盘手机App，操作更方便哦

wav2vec2: 
```
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./checkpoints/wav2vec2-base-960h
```
## Inference
```
python fm_spade/generate_pose.py 
        --input_root "path/to/image_audio"  
        --imf_path "./checkpoints/imf.ckpt"
        --fm_path "./checkpoints/fm_ckpt"
        --no_crop 
        --a_cfg_scale 2 
        --res_dir "./exps/" 
        --nfe 10
```
