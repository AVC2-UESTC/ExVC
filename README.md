# ExVC: Leveraging Mixture of Experts Models for Efficient Zero-shot Voice Conversion

Abstract:Zero-shot voice conversion (VC) aims to alter the speaker identity in a voice to resemble that of the target speaker using only a short reference speech. While existing methods have achieved notable success in generating intelligible speech, balancing the trade-off between quality and similarity of the converted voice remains a challenge, especially when using a short target reference. To address this, we propose ExVC, a zero-shot VC model that leverages the mixture of experts (MoE) layers and Conformer modules to enhance the expressiveness and overall performance. Additionally, to efficiently condition the model on speaker embedding, we employ feature-wise linear modulation (FiLM), which modulates the network based on the input speaker embedding, thereby improving the ability to adapt to various unseen speakers. Objective and subjective evaluations demonstrate that the proposed model outperforms the baseline models in terms of naturalness and quality. Audio samples are
provided at: https://tksavy.github.io/exvc/.

## Instructions For Inference (updated)

1. Clone this repo 

2. CD into this repo: `cd ExvC`

3. Install python requirements: `pip install -r requirements.txt`

```python
# inference 
CUDA_VISIBLE_DEVICES=0 python convert_exvc.py --txtpath convert.txt --outdir output_exvc
```
4. The checkpoints will be downloaded automatically from Huggingface.  If you encounter network issues, please use the following command:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Training procedure

1. Download the [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) dataset 

2. Download the [LibriTTS-R] (train_clean_360) https://www.openslr.org/resources/141/train_clean_360.tar.gz . Follow instruction on how to create a artificial dataset from [here](https://github.com/AVC2-UESTC/ExVC/blob/main/DATASET_GENERATION_HINTS.md). 

3. Preprocess
```python
python downsample.py --in_dir </path/to/VCTK/wavs> ln -s dataset/vctk-16k DUMMY

CUDA_VISIBLE_DEVICES=0  python gen_art_dataset.py 

# We assume that you use the train-val list provided. we will add the file later. 
[TODO] add train-val-test split

# ExVC uses pretrained speaker encoder, thus you need to run this
CUDA_VISIBLE_DEVICES=0 python preprocess_spk.py

# run this to preprocess the ssl features
CUDA_VISIBLE_DEVICES=0 python preprocess_ssl.py
```
2. Train

```python
# train exvc
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/exvc.json -m exvc
```
## References
- [FreeVC] (https://github.com/OlaWod/FreeVC)
- [Conformer](https://github.com/sooftware/conformer)
- [MoE] (https://gist.github.com/ruvnet/0928768dd1e4af8816e31dde0a0205d5- Mixture of experts)
- [FilM] https://arxiv.org/abs/1709.07871 
- [WavLM] https://github.com/microsoft/unilm/tree/master/wavlm

### Contact
Email: 2672291403ATqq.com
