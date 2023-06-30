from tkinter import Y
import torch
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
import PIL
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
fntsdir = os.listdir("/usr/share/fonts/truetype/")
for f in fntsdir:
    try:
        if os.path.isdir("/usr/share/fonts/truetype/"+f): 
            os.chdir("/usr/share/fonts/truetype/"+f)
            for file in glob.glob("*.ttf"):
                fntslist.append(file)
    except:
        dir = "/usr/share/fonts/truetype/freefont/"
        os.chdir("/usr/share/fonts/truetype/"+f)
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
                "depth_direction": (["q1", "q2","q3","q4"],),
                "depth_factor": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 360,
                    "step": 1,
                }),
                "letter_thickness": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 360,
                    "step": 1,
                }),
                "font": (fntslist,),
                "rotate": ("INT", {
                    "default": 0,
                    "min": -360,
                    "max": 360,
                    "step": 1,
                }),
                "size": ("INT", {
                    "default": 60,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                }),
                "rotate_3d": (["enable", "disable"],),
                "w": ("FLOAT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                }),
                "h": ("FLOAT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                }),
                "distance": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                }),
                "fade": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                }),
                "add_border":(["enable","disable"],),
                "border_thickness": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 360,
                    "step": 1,
                })          }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "word_as_image"

    CATEGORY = "SyrianFalcon/nodes"

    def word_as_image(self, text: str, add_depth: str,font: str,rotate: int,add_border:str,size: int,depth_factor:int,border_thickness:int,letter_thickness:int,rotate_3d:str,w:float,h:float,distance:int,fade:int,depth_direction:str):
        unicode_text = text
        tp = font
        try:
            font = ImageFont.truetype("C:\\Windows\\Fonts\\"+tp, size, encoding="unic")
        except:
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/"+tp, size, encoding="unic")

        # get the line size
        text_width, text_height = font.getsize(unicode_text)
        # create a blank canvas with extra space between lines
        newcanvas = Image.new('RGBA', (text_width+20 + border_thickness+depth_factor,text_height+20 + border_thickness+depth_factor),(0,0,0,0))
        canvas = Image.new('L', (text_width+20 + border_thickness+depth_factor, text_height+20 + border_thickness+depth_factor),"black")
        # draw the text onto the text canvas, and use blue as the text color
        mask_img = Image.new('L', canvas.size, 0)
        draw = ImageDraw.Draw(canvas)
        if add_border == "enable":
            draw.rectangle([0, 0, text_width+30, text_height+30],fill="white")
            draw.rectangle([border_thickness, border_thickness, text_width+30-border_thickness, text_height+30-border_thickness],fill="black")
        
        if add_depth == "enable":
            for i in range(0,depth_factor):
                clr = int(distance/depth_factor)
                x = distance - int(distance/depth_factor)*depth_factor
                if depth_direction == "q1":
                    draw.text((1*i, -i*1),unicode_text ,(clr*i+x),font=font,stroke_width = letter_thickness,stroke_fill=(clr*i+x))
                elif depth_direction =="q2":
                    draw.text((-1*i, -i*1),unicode_text ,(clr*i+x),font=font,stroke_width = letter_thickness,stroke_fill=(clr*i+x))
                elif depth_direction == "q3":
                    draw.text((-1*i, i*1),unicode_text ,(clr*i+x),font=font,stroke_width = letter_thickness,stroke_fill=(clr*i+x))
                elif depth_direction == "q4":
                    draw.text((1*i, i*1),unicode_text ,(clr*i+x),font=font,stroke_width = letter_thickness,stroke_fill=(clr*i+x))
                
        else:
            draw.text((0, 0),text,(distance),font=font)
        newcanvas.paste(canvas,(0,0),canvas)
        canvas = newcanvas
        if rotate > 0:
            canvas = canvas.rotate(rotate, PIL.Image.NEAREST, expand = 1)

        if rotate_3d == "enable":
            newcanvas = Image.new('RGBA', (text_width*2,text_height*2),(0,0,0,0))
            clrs = []
            for idx,i in enumerate(unicode_text):
                clr = int(255/(len(unicode_text)+fade))
                x = 255 - int(255/(len(unicode_text)+fade))*(len(unicode_text)+fade)
                newclr = clr*(len(unicode_text)+fade-idx)+x
                clrs.append(newclr)
            print(clrs)
            for i in range(0,depth_factor):
                canvas = Image.new('L', (text_width*2,text_height*2),"black")
                def find_coeffs(pa, pb):
                    matrix = []
                    for p1, p2 in zip(pa, pb):
                        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
                        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

                    A = np.matrix(matrix, dtype=float)
                    B = np.array(pb).reshape(8)

                    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
                    return np.array(res).reshape(8)
                width,height = canvas.size
                h1 = height*h
                l = 1-h1
                w1 = width*w
                coeffs = find_coeffs(
                        [(0+w1, 0), (width-w1, 0+h1), (width-w1, height-h1), (0+w1, height)],
                        [(0, 0), (width, 0), (width, height), (0, height)])
                mask_img = Image.new('L', canvas.size, 0)
                draw = ImageDraw.Draw(canvas)
                f = 0

                for idx,j in enumerate(unicode_text):
                    clr = int(clrs[idx]/depth_factor)
                    x = clrs[idx] - int(clrs[idx]/depth_factor)*depth_factor
                    draw.text((f, 0),j ,(clr*(i+1)+x),font=font,stroke_width = letter_thickness,stroke_fill=(clr*i+x))
                    f = font.getsize(unicode_text[0:idx+1])[0]
                if rotate > 0:
                    canvas = canvas.rotate(rotate, PIL.Image.NEAREST, expand = 1)
                canvas = canvas.transform((width, height), Image.PERSPECTIVE, coeffs,
                        Image.BICUBIC)
                if depth_direction == "q1":
                    newcanvas.paste(canvas,(0+i,0-i+depth_factor),canvas)
                elif depth_direction == "q2":
                    newcanvas.paste(canvas,(0-i+depth_factor,0-i+depth_factor),canvas)
                elif depth_direction == "q3":
                    newcanvas.paste(canvas,(0-i+depth_factor,0+i),canvas)
                elif depth_direction == "q4":
                    newcanvas.paste(canvas,(0+i,0+i),canvas)
            canvas = newcanvas
        image = np.array(canvas).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image,)
        
class QRGenerate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data_url": ("STRING", {"default": '', "multiline": False}),
                "add_depth": (["enable", "disable"],),
                "depth_factor": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 360,
                    "step": 1,
                }),
                "version": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 360,
                    "step": 1,
                }),
                "rotate": ("INT", {
                    "default": 0,
                    "min": -360,
                    "max": 360,
                    "step": 1,
                })      }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "word_as_image"

    CATEGORY = "SyrianFalcon/nodes"

    def word_as_image(self, data_url: str, add_depth: str,rotate: int,version:int,depth_factor:int):
        try:
            import qrcode,sys,subprocess

        except:
            import pip

            def install(package):
                if hasattr(pip, 'main'):
                    pip.main(['install', package])
                else:
                    pip._internal.main(['install', package])
            install("qrcode")
            import qrcode,sys,subprocess
        from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
        import time
        data = data_url

        # Creating an instance of QRCode class
        qr = qrcode.QRCode(version = version,
                        box_size = 20,
                        border = 5)

        # Adding data to the instance 'qr'
        qr.add_data(data)

        qr.make(fit = True)
        img = qr.make_image(fill_color = (125,125,125),
                            back_color = 'transparent')
        img = img.resize((512,512))

        bckgrnd = Image.new("RGB",(512,512),"white")
        clr = int(-255/depth_factor)
        x = 1 - int(-255/depth_factor)*depth_factor
        data = img.getdata()
        if add_depth == "enable":
            for i in range(0,depth_factor):
                new_image_data = []
                for item in data:
                    if item[0] in range(1,255):
                        new_image_data.append((clr*i+x, clr*i+x, clr*i+x,199))
                    else:
                        new_image_data.append(item)
                img.putdata(new_image_data)
                bckgrnd.paste(img,(0+i*1,0+i*1),img)
        #bckgrnd.paste(img,(0,0),img)
        if rotate >0:
            bckgrnd = bckgrnd.rotate(rotate, Image.NEAREST, expand = 1)
        data = bckgrnd.getdata()
        new_image_data = []
        for item in data:
            if item[0]==0:
                new_image_data.append((255, 255, 255,255))
            else:
                new_image_data.append(item)
        bckgrnd.putdata(new_image_data)
        image = np.array(bckgrnd).astype(np.float32) / 255.0
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
        
        


class CompositeImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"bg_image":("IMAGE",),
                     "inp_img":("IMAGE",),
                     "x": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                     "y": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                    
                     }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"

    CATEGORY = "SyrianFalcon/nodes"
    def sample(self, bg_image:torch.Tensor,inp_img:torch.Tensor,x:int,y:int):
        img = bg_image[0].numpy()
        img = img*255.0
        inp = inp_img[0].numpy()
        inp = inp*255.0
        inp = Image.fromarray(np.uint8(inp))
        bgimg = Image.fromarray(np.uint8(img)).convert("RGBA")
        bgimg.paste(inp,(x,y),inp)
        bgimg = bgimg.convert("RGB")
        img = np.array(bgimg).astype(np.float32) / 255.0
        img = torch.from_numpy(img)[None,]
        return (img,)


NODE_CLASS_MAPPINGS = {
    "WordAsImage": WordAsImage,
    "QRGenerate":QRGenerate,
    "KSamplerAdvancedCustom":KSamplerAdvancedCustom,
    "CompositeImage": CompositeImage
}
