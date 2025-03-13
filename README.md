 
# 🎨AdvPaint: Protecting Images from Inpainting Manipulation via Adversarial Attention Disruption
This repository contains the official implementation of **AdvPaint**, a novel defensive framework that generates adversarial perturbations that effectively
disrupt the adversary’s inpainting tasks. We will be presenting AdvPaint at _ICLR 2025_ in Singapore. Check out the project page and the paper!

![test](https://github.com/JoonsungJeon/AdvPaint/blob/main/figs/test.png)
**AdvPaint: Protecting Images from Inpainting Manipulation via Adversarial Attention Disruption**  
_Joonsung Jeon, Woo Jae Kim, Suhyeon Ha, Sooel Son*, and Sung-Eui Yoon*._  
[Paper] | [Project](https://sgvr.kaist.ac.kr/~joonsung/AdvPaint/) | 

## 🖌️Requirements
All experiments are tested on Ubuntu (20.04 or 22.04) with a single RTX 3090.

- Python 3.8
- CUDA 11.x
```
conda create --name AdvPaint python=3.8
conda activate AdvPaint

pip install -r requirements.txt
```

## 🖌️Code Instruction
### 1. Mask Generation: GroundedSAM
We used _GroundedSAM_ to generate segmentation mask, bounding-box(BB) mask, and the enlarged BB mask.
```
TBW
```
### 2. Optimizing Perturbation
If you have an input image and its enlarged BB mask, then you are ready!

```
python AdvPaint.py \
--input_dir "./test/clean/bear.png" \
--mask_dir "./test/mask/OptimBox" \
--output_dir "./test/adv" \
--prompt "A bear" \
--eps 0.06 \
--step_size 0.03 \
--iters 250
```
- --input_dir: dir to your input image
- --mask_dir: dir to the enlarged BB mask of your input image
- --output_dir: dir where the protected image will be saved
- --prompt: simple description of your input image (e.g., A bear, A cow, A frog)
- --eps: η (default = 0.06)
- --step_size: controls the amount of update per iteration (default = 0.03)
- --iters: # of iterations (default = 250)


### 3. Inference
Here, we compare the inpainted results of the clean image and the protected image.
```
python SD_inpaint.py \
--clean_dir "./test/clean/bear.png" \
--adv_dir "./test/adv/bear_AdvPaint_eps0.06_step0.03_iter250.png" \
--output_dir "./test/inpaint" \
--mask_dir "./test/mask/bear_mask2_seg.png" \
--prompt "A bear at a cherry blossom grove."
```
- --clean_dir: dir to your clean input image
- --adv_dir: dir to your protected image
- --output_dir: dir where the inpainted results will be saved
- --mask_dir: dir to the mask
- --prompt: editing text-prompt

## 🖌️Bibtex
If you find this code useful for your research, please consider citing our paper:
```
@inproceedings{
jeon2025advpaint,
title={AdvPaint: Protecting Images from Inpainting Manipulation via Adversarial Attention Disruption},
author={Joonsung Jeon and Woo Jae Kim and Suhyeon Ha and Sooel Son and Sung-eui Yoon},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=m73tETvFkX}
}
```
