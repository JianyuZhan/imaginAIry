import argparse
import copy
import os
import sys
import time

import diffusers
from packaging import version
diffusers_version = version.parse(diffusers.__version__)
use_new_diffusers = diffusers_version >= version.parse('0.18.0')
if use_new_diffusers:
    from diffusers.models.attention_processor import Attention
else:
    from diffusers.models.cross_attention import CrossAttention

from diffusers import StableDiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np

import torch
import torch.nn as nn
import torch_neuronx


# Constants
COMPILER_WORKDIR_ROOT = 'sd_1_5_fp32_512_compile_workdir'
model_name_or_path = "runwayml/stable-diffusion-v1-5"
PROMPTS = [
    "a photo of an astronaut riding a horse on mars",
    "sonic on the moon",
]

# Set environment variables for the session
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "True"

def get_attention_scores(self, query, key, attn_mask):    
    dtype = query.dtype

    if self.upcast_attention:
        query = query.float()
        key = key.float()

    if(query.size() == key.size()):
        attention_scores = cust_badbmm(
            key,
            query.transpose(-1, -2),
            self.scale
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=1).permute(0,2,1)
        attention_probs = attention_probs.to(dtype)

    else:
        attention_scores = cust_badbmm(
            query,
            key.transpose(-1, -2),
            self.scale
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.to(dtype)
        
    return attention_probs

def cust_badbmm(a, b, scale):
    bmm = torch.bmm(a, b)
    scaled = bmm * scale
    return scaled


class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        out_tuple = self.unet(sample, timestep, encoder_hidden_states, return_dict=False)
        return out_tuple

class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None, return_dict=False):
        sample = self.unetwrap(sample, timestep.float().expand((sample.shape[0],)), encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)

class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = torch.float32
        self.device = text_encoder.device

    def forward(self, emb, attention_mask = None):
        return [self.neuron_text_encoder(emb)['last_hidden_state']]

class NeuronSafetyModelWrap(nn.Module):
    def __init__(self, safety_model):
        super().__init__()
        self.safety_model = safety_model

    def forward(self, clip_inputs):
        return list(self.safety_model(clip_inputs).values())

def compile_text_encoder(model_name_or_path):
    # Compile and save the text encoder model
    # --- Compile CLIP text encoder and save ---
    # Only keep the model being compiled in RAM to minimze memory pressure
    pipe = StableDiffusionPipeline.from_pretrained(model_name_or_path, torch_dtype=torch.float32)
    text_encoder = copy.deepcopy(pipe.text_encoder)
    del pipe

    # Apply the wrapper to deal with custom return type
    text_encoder = NeuronTextEncoder(text_encoder)

    # Compile text encoder
    # This is used for indexing a lookup table in torch.nn.Embedding,
    # so using random numbers may give errors (out of range).
    emb = torch.tensor([[49406, 18376,   525,  7496, 49407,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0]])

    with torch.no_grad():
        text_encoder_neuron = torch_neuronx.trace(
                text_encoder.neuron_text_encoder,
                emb,
                compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
                compiler_args=["--enable-fast-loading-neuron-binaries"]
                )

    # Save the compiled text encoder
    text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
    torch_neuronx.async_load(text_encoder_neuron)
    torch.jit.save(text_encoder_neuron, text_encoder_filename)

    # delete unused objects
    del text_encoder
    del text_encoder_neuron
    del emb
    print("Compiled text encoder")

def compile_vae_decoder(model_name_or_path):
    # Compile and save the VAE decoder model
    # Only keep the model being compiled in RAM to minimze memory pressure
    pipe = StableDiffusionPipeline.from_pretrained(model_name_or_path, torch_dtype=torch.float32)
    decoder = copy.deepcopy(pipe.vae.decoder)
    del pipe

    # # Compile vae decoder
    decoder_in = torch.randn([1, 4, 64, 64])
    with torch.no_grad():
        decoder_neuron = torch_neuronx.trace(
            decoder, 
            decoder_in, 
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
            compiler_args=["--enable-fast-loading-neuron-binaries"]
        )

    # Save the compiled vae decoder
    decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
    torch_neuronx.async_load(decoder_neuron)
    torch.jit.save(decoder_neuron, decoder_filename)

    # delete unused objects
    del decoder
    del decoder_in
    del decoder_neuron

    print("Compiled vae decoder")


def compile_unet(model_name_or_path):
    # Compile and save the UNet model
    pipe = StableDiffusionPipeline.from_pretrained(model_name_or_path, torch_dtype=torch.float32)

    # Replace original cross-attention module with custom cross-attention module for better performance
    if use_new_diffusers:
        Attention.get_attention_scores = get_attention_scores
    else:
        CrossAttention.get_attention_scores = get_attention_scores

    # Apply double wrapper to deal with custom return type
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))

    # Only keep the model being compiled in RAM to minimze memory pressure
    unet = copy.deepcopy(pipe.unet.unetwrap)
    del pipe

    # Compile unet - FP32
    sample_1b = torch.randn([1, 4, 64, 64])
    timestep_1b = torch.tensor(999).float().expand((1,))
    encoder_hidden_states_1b = torch.randn([1, 77, 768])
    example_inputs = sample_1b, timestep_1b, encoder_hidden_states_1b

    with torch.no_grad():
        unet_neuron = torch_neuronx.trace(
            unet,
            example_inputs,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
            compiler_args=["--model-type=unet-inference", "--enable-fast-loading-neuron-binaries"]
        )

    # save compiled unet
    unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
    torch_neuronx.async_load(unet_neuron)
    torch_neuronx.lazy_load(unet_neuron)
    torch.jit.save(unet_neuron, unet_filename)

    # delete unused objects
    del unet
    del unet_neuron
    del sample_1b
    del timestep_1b
    del encoder_hidden_states_1b

    print("Compiled unet")


def compile_post_quant_conv(model_name_or_path):
    # Compile and save the VAE post_quant_conv model
    # Only keep the model being compiled in RAM to minimze memory pressure
    pipe = StableDiffusionPipeline.from_pretrained(model_name_or_path, torch_dtype=torch.float32)
    post_quant_conv = copy.deepcopy(pipe.vae.post_quant_conv)
    del pipe

    # # # Compile vae post_quant_conv
    post_quant_conv_in = torch.randn([1, 4, 64, 64])
    with torch.no_grad():
        post_quant_conv_neuron = torch_neuronx.trace(
            post_quant_conv, 
            post_quant_conv_in,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
            compiler_args=["--enable-fast-loading-neuron-binaries"]
        )


    # # Save the compiled vae post_quant_conv
    post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')
    torch_neuronx.async_load(post_quant_conv_neuron)
    torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)

    # delete unused objects
    del post_quant_conv

    print("Compiled vae post_quant_conv")

def compile_safety_model(model_name_or_path):
    # Compile and save the safety checker model
    # Only keep the model being compiled in RAM to minimze memory pressure
    pipe = StableDiffusionPipeline.from_pretrained(model_name_or_path, torch_dtype=torch.float32)
    safety_model = copy.deepcopy(pipe.safety_checker.vision_model)
    del pipe

    clip_input = torch.randn([1, 3, 224, 224])
    with torch.no_grad():
        safety_model_neuron = torch_neuronx.trace(
            safety_model, 
            clip_input,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'safety_model_neuron'),
            compiler_args=["--enable-fast-loading-neuron-binaries"]
        )

    # # Save the compiled vae post_quant_conv
    safety_model_neuron_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'safety_model_neuron/model.pt')
    torch_neuronx.async_load(safety_model_neuron)
    torch.jit.save(safety_model_neuron, safety_model_neuron_filename)

    # delete unused objects
    del safety_model_neuron

    print("Compiled safety model")

def compile_models(model_name_or_path):
    compile_text_encoder(model_name_or_path)
    compile_vae_decoder(model_name_or_path)
    compile_unet(model_name_or_path)
    compile_post_quant_conv(model_name_or_path)
    compile_safety_model(model_name_or_path)

def load_compiled_models(model_name_or_path):
    # Your code to load compiled models
    pipe = StableDiffusionPipeline.from_pretrained(model_name_or_path, torch_dtype=torch.float32)

    text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
    unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
    decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
    post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')
    safety_model_neuron_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'safety_model_neuron/model.pt')


    # Load the compiled UNet onto two neuron cores.
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
    device_ids = [0,1]
    pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)

    # Load other compiled models onto a single neuron core.
    pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
    pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
    pipe.vae.decoder = torch.jit.load(decoder_filename)
    pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)
    pipe.safety_checker.vision_model = NeuronSafetyModelWrap(torch.jit.load(safety_model_neuron_filename))

    print("Loaded compiled models")
    return pipe

def generate_images(pipe):
    # Run pipeline
    prompt = ["a photo of an astronaut riding a horse on mars",
              "sonic on the moon",
              "elvis playing guitar while eating a hotdog",
              "saved by the bell",
              "engineers eating lunch at the opera",
              "panda eating bamboo on a plane",
              "A digital illustration of a steampunk flying machine in the sky with cogs and mechanisms, 4k, detailed, trending in artstation, fantasy vivid colors",
              "kids playing soccer at the FIFA World Cup"
            ]

    plt.title("Image")
    plt.xlabel("X pixel scaling")
    plt.ylabel("Y pixels scaling")

    total_time = 0
    for i, x in enumerate(prompt):
        start_time = time.time()
        image = pipe(x).images[0]
        total_time = total_time + (time.time()-start_time)
        image.save(f"image_{i}.png")
        image = mpimg.imread(f"image_{i}.png")
        plt.imshow(image)
        plt.show()
    print("Average time: ", np.round((total_time/len(prompt)), 2), "seconds")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="Hugginface model id")
    parser.add_argument('--model_dir', type=str, default="Local model folder, already complied for diffusers") 
    args = parser.parse_args()

    if args.model_name_or_path == '' and args.model_dir == '':
        sys.exit("Please provide either a model id or a model directory")
    model_name_or_path = args.model_name_or_path if args.model_name_or_path != '' else args.model_dir

    compile_models(model_name_or_path)
    pipe = load_compiled_models(model_name_or_path)
    generate_images(pipe)

if __name__ == "__main__":
    main()
