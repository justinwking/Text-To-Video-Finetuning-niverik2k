import argparse
import datetime
import logging
import inspect
import math
import os
import random
import gc
import copy
import json
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import cv2
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import diffusers
import transformers
from torchsummary import summary

from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from models.unet_3d_condition import UNet3DConditionModel
from diffusers.models import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, TextToVideoSDPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, export_to_video
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock

from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPEncoder
from utils.dataset import VideoJsonDataset, SingleVideoDataset, \
    ImageDataset, VideoFolderDataset, CachedDataset
from einops import rearrange, repeat

from utils.lora import (
    extract_lora_ups_down,
    inject_trainable_lora,
    inject_trainable_lora_extended,
    save_lora_weight,
    train_patch_pipe,
    monkeypatch_or_replace_lora,
    monkeypatch_or_replace_lora_extended
)

path = 'F:\\AI\\Text-to-Video-Finetuning\\models\\model_scope_diffusers'

def load_primary_models(pretrained_model_path):
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    return {"noise_scheduler":noise_scheduler, "tokenizer":tokenizer, "text_encoder":text_encoder, "vae":vae, "unet":unet}

def handle_trainable_modules(model_input):
    summary = str(model_input)
    # for name, module in model_input.named_modules():
    #         print(f"named_module =>{name}")
    return summary


def close_model(model): 
    try:
        if hasattr(model,"close"):
            model.close()
            return "closed"
        else:
            return "Nothing to close here."
    except Exception as e:
        return e
    
if __name__ == "__main__":
    json_filepath = 'D:\\Dropbox\\DreamBoothTraining\\ModelscopeData\\modelscope_structure-'
    print("Loading Model")
    model_input_dict = load_primary_models(path)
    model_output_dict = {}
    print("Shuffling Through Models")
    for key,lmodel in model_input_dict.items():
        model_output_dict.update({key:handle_trainable_modules(lmodel)})
        print(f"closing {key} models")
        print(close_model(lmodel))
    

    
        # Write the dictionary to the file
    for key, value in model_output_dict.items():
        with open(f"{json_filepath}{key}.txt", "w") as f:
            print(f"writing {key}")
            f.write(f"{value}\n")
        with open(f"{json_filepath}{key}.json", "w") as j:
            json.dump({key:" ".join(value.split())},j)
    exit()
    