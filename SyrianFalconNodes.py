import torch
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
import PIL
import numpy as np
import torch.nn.functional as F
import cv2
import os
import torch
import sys
import json
import hashlib
import traceback
import time
import re
import glob
from PIL.PngImagePlugin import PngInfo
import numpy as np
import safetensors.torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils

import comfy.clip_vision

import comfy.model_management
import importlib

import folder_paths
import latent_preview

import math
import struct

dir = ""
fntslist = []
try:
    fntsdir = os.listdir("C:\\Windows\\Fonts\\")
    dir = "C:\\Windows\\Fonts\\"

except:
    fntsdir = os.listdir("/usr/share/fonts/truetype/freefont/")
    dir = "/usr/share/fonts/truetype/freefont/"

os.chdir(dir)
for file in glob.glob("*.ttf"):
    fntslist.append(file)



class WordAsImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": '', "multiline": False}),
                "add_depth": (["enable", "disable"],),
                "font": (fntslist,),
                "rotate": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 360,
                    "step": 15,
                }),
                "add_border":(["enable","disable"],)            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "word_as_image"

    CATEGORY = "SyrianFalcon/nodes"

    def word_as_image(self, text: str, add_depth: str,font: str,rotate: int,add_border:str):
        unicode_text = text
        tp = font
        try:
        	font = ImageFont.truetype("C:\\Windows\\Fonts\\"+tp, 60, encoding="unic")
        except:
        	font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/"+tp, 60, encoding="unic")

        # get the line size
        text_width, text_height = font.getsize(unicode_text)

        # create a blank canvas with extra space between lines
        canvas = Image.new('L', (text_width+20 + 10, text_height+20 + 10))
        # draw the text onto the text canvas, and use blue as the text color
        draw = ImageDraw.Draw(canvas)
        if add_border == "enable":
        	draw.rectangle([0, 0, text_width+30, text_height+30],fill="white")
        	draw.rectangle([10, 10, text_width+20, text_height+20],fill="black")
        if add_depth == "enable":
        	for i in range(0,20):
            		draw.text((5+i*1, 5+i*1),text,(100+i*8),font=font)
        else:
        	draw.text((15, 12),text,(255),font=font)
        if rotate > 0:
        	canvas = canvas.rotate(rotate, PIL.Image.NEAREST, expand = 1)
        
        image = np.array(canvas).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image,)

def promptscreator(text,steps):
    prompt = text
    checkpoints = []
    replaces = {}
    additions = {}
    removes = {}
    initials = []
    numloop = []
    total = steps
    prompts = []
    occs = re.findall("\[[^\[]*]",prompt)
    #for i in occs:
        #checkpoints.append(re.search(occs[1][1:-1],prompt).span())
    for idx,i in enumerate(occs):
        #c = int(re.findall("[\d][^\]]*",i)[0][:]) if float(re.findall("[\d][^\]]*",i)[0][:])>=1 else int(float(re.findall("[\d][^\]]*",i)[0][:])*total)
        if len(re.findall(":",i)) > 1:
            if re.search("::",i):
                c = int(i[re.search("::",i).span()[1]:-1]) if float(i[re.search("::",i).span()[1]:-1])>=1 else int(float(i[re.search("::",i).span()[1]:-1])*total)
                if c not in numloop:
                    numloop.append(c)
                removes[c] = [i,re.findall(".*:",i)[0][1:-2],re.findall(".*:",i)[0][1:-2]]
                initials.append([i,re.findall(".*:",i)[0][1:-2]])
            else:
                i2 = i[re.search(":",i).span()[0]+1:]   
                     
                c = int(i2[re.search(":",i2).span()[0]+1:-1]) if float(i2[re.search(":",i2).span()[0]+1:-1])>=1 else int(float(i2[re.search(":",i2).span()[0]+1:-1])*total)
                if c not in numloop:
                    numloop.append(c)
                   
                replaces[c] = [i,re.findall(":.*:",i)[0][1:-1],re.findall("[^:]*:",i)[0][1:-1]]
                initials.append([i,re.findall("[^:]*:",i)[0][1:-1]])
        else:
            c = int(i[re.search(":",i).span()[0]+1:-1]) if float(i[re.search(":",i).span()[0]+1:-1])>=1 else int(float(i[re.search(":",i).span()[0]+1:-1])*total)
            if c not in numloop:
                numloop.append(c)
       
            additions[c] = [i,re.findall(".*:",i)[0][1:-1],re.findall(".*:",i)[0][1:-1],re.search(i[1:-1],prompt).span()[0]]
            initials.append([i,""])


  
    numloop.sort()

    for i in initials:
        prompt = re.sub("."+i[0][1:-1]+".",i[1],prompt,1)

    prompts.append(prompt)
    for x in numloop:
        if x in replaces:
            prompt = re.sub(replaces[x][2],replaces[x][1],prompt,1)
            
        if x in removes:
            prompt = re.sub(removes[x][2],"",prompt,1)

        if x in additions:
            prompt = prompt +", "+ additions[x][2]
        prompts.append(prompt)
    if len(prompts)==0:
    	prompts.append(prompt)
    	#print(prompts)
    	numloop.append(steps-1)
    return prompts,numloop

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False,flag=False):
    
    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = latent_preview.get_previewer(device)

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback)
    out = latent.copy()
    out["samples"] = samples
    if not flag:
    	return out
    else: 
    	return (out, )
    	

        
        
class KSamplerAdvancedCustom:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "return_with_leftover_noise": (["disable", "enable"], ),"clip": ("CLIP", ),"text": ("STRING", {"multiline": True})
                    
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "SyrianFalcon/nodes"
    def sample(self, clip,model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, text, return_with_leftover_noise, denoise=1.0):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        
        prompts,numloop = promptscreator(text,steps)
        if len(prompts) ==0:
        	prompts.append(text)
       	if len(numloop) < 1:
       		numloop.append(steps-1)
        positive = ([[clip.encode(prompts[0]), {}]] )
        c1 = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=0, last_step=numloop[0], force_full_denoise=force_full_denoise)

        for i in range(0,len(numloop)-1):
       		positive = ([[clip.encode(prompts[i]), {}]] )
        	common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, c1, denoise=denoise, disable_noise=disable_noise, start_step=numloop[i], last_step=numloop[i+1], force_full_denoise=force_full_denoise)
        positive = ([[clip.encode(prompts[len(prompts)-1]), {}]] )
        return common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, c1, denoise=denoise, disable_noise=disable_noise, start_step=numloop[len(numloop)-1], last_step=steps, force_full_denoise=force_full_denoise,flag=True)
        
        


NODE_CLASS_MAPPINGS = {
    "WordAsImage": WordAsImage,
    "KSamplerAdvancedCustom":KSamplerAdvancedCustom
}
