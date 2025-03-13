import os
import torch
import numpy as np
import torchvision.transforms as T

import glob
from tqdm import tqdm

from diffusers.utils.torch_utils import randn_tensor
from pipeline_stable_diffusion_inpaint_pgd import *

import random
import argparse

from utils import preprocess
from utils_UNet import (
    self_maps,
    cross_query,
    self_query,
    self_key,
    self_value,
    cross_attn_init,
    register_cross_attention_hook,
    set_layer_with_name_and_path,
    detach_cross_attention_hook,
    save_by_timesteps_and_path,
    save_by_timesteps
)



to_pil = T.ToPILImage()


pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    safety_checker = None,
    requires_safety_checker = False
)

pipeline = pipeline.to("cuda")
##### 1. Init modules #####
cross_attn_init(pipeline)

##### 2. Replace modules and Register hook #####
pipeline.unet = set_layer_with_name_and_path(pipeline.unet)


def collate_fn(batch):
    return tuple(zip(*batch))



def pgd_SelfQKV_And_Cross_Xadv(img_dir, X, model, eps=0.06, step_size=0.03, iters=100, clamp_min=0, clamp_max=1, mask_num=1):
    ## val mask
    ori_img = img_dir
    init_image = Image.open(ori_img).convert("RGB")
    
    prompt = args.prompt

    ## inp_mask: perturbation이 들어갈 위치 설정용 mask ==> black인곳에 perturb
    ## mask512: inpainter에 들어가는 mask
    
    # inp_mask = "/mnt/nas3/joonsung/AdvPaint/Attn/Mask/person_bench/person_mask_black.png"
    # mask_image = Image.open(inp_mask).convert("RGB")
    # inp_mask_512 = T.ToTensor()(mask_image).unsqueeze(0)
    # inp_mask_512 = inp_mask_512.to(device="cuda", dtype=torch.float32)
    
    X_ori = X.detach().clone()
    
    mask_dir = glob.glob(args.mask_dir+"/*.png")
    
    X_adv = X.clone().detach() + ((torch.rand(*X.shape)*2*eps-eps).to("cuda"))
    
    
    
    
    
    for mask_num, m in enumerate(mask_dir):
        
        # inp_mask = "/mnt/nas3/joonsung/AdvPaint/Attn/Mask/person_bench/person_mask2.png"
        mask_image = Image.open(m).convert("RGB")
        mask_512 = T.ToTensor()(mask_image).unsqueeze(0)
        mask_512 = mask_512.to(device="cuda", dtype=torch.float32)
        


        m = m.replace(".", "_")
        
        
        
        model.unet = register_cross_attention_hook(model.unet, ATTN="attn1")
        model.unet = register_cross_attention_hook(model.unet, ATTN="attn2")
     
        
        with torch.no_grad():
            model(prompt=prompt, image=init_image, mask_image=mask_image, guidance_scale=7.5)
            
            
            
        ## Params from pipeline ##
        timesteps = model.timesteps
        mask = model.mask
        timestep_cond = model.timestep_cond
        prompt_embeds = model.prompt_embeds
        added_cond_kwargs = model.added_cond_kwargs
            
            
        

        GT_self_query = {}
        GT_self_key = {}
        GT_self_value = {}
        GT_cross_query = {}
        
        X_ori_masked = X_ori * (mask_512 < 0.5)



        with torch.no_grad():
            image_latents = model.vae.encode(X_ori).latent_dist.sample(model.generator)
            image_latents = model.vae.config.scaling_factor * image_latents
            image_latents = image_latents.repeat(1 // image_latents.shape[0], 1, 1, 1)
    
            noise = randn_tensor(image_latents.shape, generator=model.generator, device=model.device, dtype=prompt_embeds.dtype)
            zT = model.scheduler.add_noise(image_latents, noise, model.latent_timestep)
        

            ori_masked_latents = model.vae.encode(X_ori_masked).latent_dist.sample(model.generator)
            ori_masked_latents = model.vae.config.scaling_factor * ori_masked_latents
            ori_masked_latents = ori_masked_latents.repeat(1 // ori_masked_latents.shape[0], 1, 1, 1)
            ori_masked_latents = torch.cat([ori_masked_latents] * 2) 
            ori_masked_latents = ori_masked_latents.to(device=model.device, dtype=prompt_embeds.dtype)
            
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([zT] * 2) if model.do_classifier_free_guidance else latents

            # concat latents, mask, masked_image_latents in the channel dimension
            latent_model_input = model.scheduler.scale_model_input(latent_model_input, timesteps[0])

                    
            latent_model_input = torch.cat([latent_model_input, mask, ori_masked_latents], dim=1)

                            
            # predict the noise residual
            _ = model.unet(
                        latent_model_input,
                        timesteps[0],
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=model.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
            )[0]
            
            # if "mask1" in m.split("_"):
            #     for timestep in self_maps.keys():
            #         for path in self_maps[timestep].keys():
            #             GT_attn[timestep] = GT_attn.get(timestep, dict())
            #             GT_attn[timestep][path] = self_maps[timestep][path].detach().clone() # torch.Size([16, 77, h, w])
             
            ## Self - Query ##            
            for timestep in self_query.keys():
                    for path in self_query[timestep].keys():
                        GT_self_query[timestep] = GT_self_query.get(timestep, dict())
                        GT_self_query[timestep][path] = self_query[timestep][path].detach() # torch.Size([16, 77, h, w])
                        
            ## Self - Key ##
            for timestep in self_key.keys():
                    for path in self_key[timestep].keys():
                        GT_self_key[timestep] = GT_self_key.get(timestep, dict())
                        GT_self_key[timestep][path] = self_key[timestep][path].detach() # torch.Size([16, 77, h, w])
                        
            ## Self - Value ##
            for timestep in self_value.keys():
                    for path in self_value[timestep].keys():
                        GT_self_value[timestep] = GT_self_value.get(timestep, dict())
                        GT_self_value[timestep][path] = self_value[timestep][path].detach() # torch.Size([16, 77, h, w])
                
            ## Cross - Query ##        
            for timestep in cross_query.keys():
                for path in cross_query[timestep].keys():
                    GT_cross_query[timestep] = GT_cross_query.get(timestep, dict())
                    GT_cross_query[timestep][path] = cross_query[timestep][path].detach() # torch.Size([16, 77, h, w])

            

        pbar = tqdm(range(iters))
        X_ori_masked = X_ori * (mask_512 < 0.5)
        
        for iter in pbar:
            
            
            actual_step_size = step_size - (step_size - step_size / 100) / iters * iter
            
        
            X_adv.requires_grad_(True)
            
            
            
            
            ## Plain image ##  
            with torch.no_grad():
                image_latents = model.vae.encode(X_adv).latent_dist.sample(model.generator)
                image_latents = model.vae.config.scaling_factor * image_latents
                image_latents = image_latents.repeat(1 // image_latents.shape[0], 1, 1, 1)
                    

                    
                noise = randn_tensor(image_latents.shape, generator=model.generator, device=model.device, dtype=prompt_embeds.dtype)
                latents = model.scheduler.add_noise(image_latents, noise, model.latent_timestep)
                
            

            ## Masked image ## 
            masked_image = X_adv * (mask_512 < 0.5)
            

            masked_image_latents = model.vae.encode(masked_image).latent_dist.sample(model.generator)
            masked_image_latents = model.vae.config.scaling_factor * masked_image_latents
            masked_image_latents = masked_image_latents.repeat(1 // masked_image_latents.shape[0], 1, 1, 1)
            masked_image_latents = torch.cat([masked_image_latents] * 2) 
            masked_image_latents = masked_image_latents.to(device=model.device, dtype=prompt_embeds.dtype)
            
            
            
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if model.do_classifier_free_guidance else latents

            # concat latents, mask, masked_image_latents in the channel dimension
            latent_model_input = model.scheduler.scale_model_input(latent_model_input, timesteps[0])

                    
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                            
            # predict the noise residual
            _ = model.unet(
                        latent_model_input,
                        timesteps[0],
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=model.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
            )[0]
                       
            
            
           
            
            
            
            loss_cross_q = 0
            loss_query = 0
            loss_key = 0 
            loss_value = 0

       
            length = len(self_query[timestep].keys())
            for timestep in self_query.keys():
                for path in self_query[timestep].keys():
                        

                    loss_query -= (GT_self_query[timestep][path]-self_query[timestep][path]).norm(p=2)

            for timestep in self_key.keys():
                for path in self_key[timestep].keys():
                        
                        

                    loss_key -= (GT_self_key[timestep][path]-self_key[timestep][path]).norm(p=2)

            for timestep in self_value.keys():
                for path in self_value[timestep].keys():
                    loss_value -= (GT_self_value[timestep][path]-self_value[timestep][path]).norm(p=2)       
                    
            for timestep in cross_query.keys():
                for path in cross_query[timestep].keys():
                    loss_cross_q -= (GT_cross_query[timestep][path]-cross_query[timestep][path]).norm(p=2)
                 
           
           
            
            loss = (loss_query + loss_key + loss_value + loss_cross_q) / length
            

            grad, = torch.autograd.grad(loss, [X_adv])
            
        
            pbar.set_description(f"[Running attack for {m.split('_')[-2]}]: loss_cross_q {(loss_cross_q/ length).item():.5f} | loss_query {(loss_query/length).item():.5f} |  loss_key {(loss_key/length).item():.5f} | loss_value {(loss_value/ length).item():.5f} | step size: {actual_step_size:.4}")
           
            X_adv = X_adv - grad.detach().sign() * actual_step_size
            X_adv = torch.minimum(torch.maximum(X_adv, X_ori - eps), X_ori + eps)
            X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
            X_adv.grad = None
            
            del grad, loss_query, loss_key, loss_value, loss_cross_q
            torch.cuda.empty_cache()
            

    return X_adv





def main():
    
    img_dir = args.input_dir

    init_image = Image.open(img_dir).convert("RGB")
    

    
    seed = 9999

    
    deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    

    
    with torch.autocast('cuda'):
        X = preprocess(init_image).half().to("cuda")

        
        adv_X = pgd_SelfQKV_And_Cross_Xadv(img_dir, 
                    X, 
                    model=pipeline,
                    eps=args.eps, 
                    step_size=args.step_size,
                    iters=args.iters,
                    clamp_min=-1,
                    clamp_max=1,
                   )

        adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
    
        adv_image = to_pil(adv_X[0]).convert("RGB")

    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    adv_image.save(args.output_dir+f"/{args.input_dir.split('/')[-1].split('.')[0]}_AdvPaint_eps{args.eps}_step{args.step_size}_iter{args.iters}.png", 'png')
    

   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True,
                        help='Input image dir')
    parser.add_argument('--output_dir', required=True,
                        help='Output Image dir')
    parser.add_argument('--mask_dir', required=True,
                        help='Enlarged bounding-box mask dir')
    parser.add_argument('--prompt', required=True,
                        help='prompt')
    parser.add_argument('--eps', default=0.1, type=float)
    parser.add_argument('--step_size', default=0.05, type=float)

    parser.add_argument('--iters', default=100, type=int)
    
    args = parser.parse_args()

    main()
    
