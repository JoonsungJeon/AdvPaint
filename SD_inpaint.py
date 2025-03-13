import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import os
import argparse

pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    safety_checker = None,
    requires_safety_checker = False,

)


pipeline = pipeline.to("cuda")

def main():


    clean_image = Image.open(args.clean_dir).convert("RGB")
    adv_image = Image.open(args.adv_dir).convert("RGB")
    mask_image = Image.open(args.mask_dir).convert("RGB")

    prompt = args.prompt



    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            

    SEED = args.seed

    torch.manual_seed(SEED)
    image = pipeline(prompt=prompt, image=clean_image, mask_image=mask_image).images[0]

    save_dir = args.output_dir+f"/Clean_{args.mask_dir.split('/')[-1].split('.')[0]}.png"
    image.save(save_dir, "png")


    torch.manual_seed(SEED)
    image = pipeline(prompt=prompt, image=adv_image, mask_image=mask_image).images[0]

    save_dir = args.output_dir+f"/AdvPaint_{args.mask_dir.split('/')[-1].split('.')[0]}.png"
    image.save(save_dir, "png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_dir', required=True,
                        help='Clean input image dir')
    parser.add_argument('--adv_dir', required=True,
                        help='Protected input image dir')
    
    parser.add_argument('--output_dir', required=True,
                        help='Inpainted image dir')
    
    parser.add_argument('--mask_dir', required=True,
                        help='Mask used ')
    
    
    parser.add_argument('--prompt', required=True,
                        help='prompt')
    parser.add_argument('--seed', default=2025, type=float)

    
    args = parser.parse_args()

    main()
    