import argparse
import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import inspect

import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch_neuronx
import numpy as np

from diffusers import StableDiffusionControlNetPipeline,StableDiffusionPipeline, ControlNetModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.utils import load_image
import diffusers
from packaging import version
diffusers_version = version.parse(diffusers.__version__)
use_new_diffusers = diffusers_version >= version.parse('0.18.0')
if use_new_diffusers:
    from diffusers.models.attention_processor import Attention
else:
    from diffusers.models.cross_attention import CrossAttention

COMPILER_WORKDIR_ROOT = 'controlenet_compile_workdir'

def get_canny_image():
  image = load_image(
      "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
  )

  image = np.array(image)

  low_threshold = 100
  high_threshold = 200

  image = cv2.Canny(image, low_threshold, high_threshold)
  image = image[:, :, None]
  image = np.concatenate([image, image, image], axis=2)
  canny_image = Image.fromarray(image)

  return canny_image

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

class ControlNetUNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        signature = inspect.signature(self.forward)
        print('signature of ControlNetUNetWrap.forward: ', signature)

    def forward(self, sample, timestep, encoder_hidden_states, #timestep_cond=None, 
                down_block_additional_residuals: List[torch.Tensor] = None, 
                mid_block_additional_residual: List[torch.Tensor] = None,
                cross_attention_kwargs=None):
        sig = inspect.signature(self.forward)
        params = sig.parameters
        print('######################## ControlNetUNetWrap')
        for name, param in params.items():
          if name == 'args':
              for i, arg in enumerate(args):
                  print(f"args[{i}]: type={type(arg)}")
          elif name == 'kwargs':
              for key, value in kwargs.items():
                  print(f"kwargs['{key}']: type={type(value)}")
          else:
              value = locals()[name]
              print(f"{name}: type={type(value)}")
        dbar: List[torch.Tenso] = down_block_additional_residuals
        out_tuple = self.unet(sample, timestep, encoder_hidden_states, down_block_additional_residuals,
                              down_block_additional_residuals=down_block_additional_residuals,
                              mid_block_additional_residual=mid_block_additional_residual,
                              return_dict=False)
        return out_tuple
    
class ControlNetNeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device
        signature = inspect.signature(self.unetwrap)
        print('signature of ControlNetNeuronUNet.unetwrap: ', signature)
        signature = inspect.signature(self.unetwrap.forward)
        print('signature of ControlNetNeuronUNet.unetwrap.forward: ', signature)

    def forward(self, sample, timestep, encoder_hidden_states, 
                down_block_additional_residuals: List[torch.Tensor] = None,
                mid_block_additional_residual: List[torch.Tensor] = None, 
                cross_attention_kwargs=None, return_dict=False):
        sig = inspect.signature(self.forward)
        params = sig.parameters
        print('######################## ControlNetNeuronUNet')
        for name, param in params.items():
          if name == 'args':
              for i, arg in enumerate(args):
                  print(f"args[{i}]: type={type(arg)}")
          elif name == 'kwargs':
              for key, value in kwargs.items():
                  print(f"kwargs['{key}']: type={type(value)}")
          else:
              value = locals()[name]
              print(f"{name}: type={type(value)}")
        sample = self.unetwrap(sample, timestep.float().expand((sample.shape[0],)), 
                               encoder_hidden_states, 
                               down_block_additional_residuals=down_block_additional_residuals,
                               mid_block_additional_residual=mid_block_additional_residual)[0]
        return UNet2DConditionOutput(sample=sample)

class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = torch.float32
        self.device = text_encoder.device

    def forward(self, emb, attention_mask=None):
        return [self.neuron_text_encoder(emb)['last_hidden_state']]


class NeuronSafetyModelWrap(nn.Module):
    def __init__(self, safety_model):
        super().__init__()
        self.safety_model = safety_model

    def forward(self, clip_inputs):
        return list(self.safety_model(clip_inputs).values())
    

def get_example_inputs():
    sample = torch.randn(2, 4, 64, 64) # input image latent
    timestep = torch.tensor(1) # timestep
    encoder_hidden_states = torch.randn(2, 77, 768) # prompt emb
    controlnet_cond = torch.randn(2, 3, 512, 512) # generated image
    conditioning_scale = torch.tensor([1.0])

    print("Example inputs:")
    print(f"sample: {sample.shape}")
    print(f"timestep: {timestep}")
    print(f"encoder_hidden_states: {encoder_hidden_states.shape}")
    print(f"controlnet_cond: {controlnet_cond.shape}")
    #print(f"conditioning_scale: {conditioning_scale}")
    #print(f"guess_mode: {guess_mode}")
    #print(f"return_dict: {return_dict}")
    return (sample, timestep, encoder_hidden_states, controlnet_cond)
            #conditioning_scale, class_labels, timestep_cond,attention_mask, added_cond_kwargs, cross_attention_kwargs, 
            #guess_mode,return_dict)


class ControlNetWrapper(torch.nn.Module):
    def __init__(self, compiled_model):
        super(ControlNetWrapper, self).__init__()
        self.compiled_model = compiled_model

    def forward(self, sample, timestep, encoder_hidden_states, controlnet_cond,  **kwargs):
        #print(**kwargs)
        return self.compiled_model(sample, timestep, encoder_hidden_states, controlnet_cond)
                                   #conditioning_scale=conditioning_scale, guess_mode=guess_mode, return_dict=return_dict)

class NeuronControlNet(nn.Module):
    def __init__(self, cn_model):
        super().__init__()
        self.cn_model = cn_model

    def forward(self, sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.FloatTensor,
        conditioning_scale = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode=False,
        return_dict = False):
        return self.cn_model(sample, timestep, 
                             encoder_hidden_states=encoder_hidden_states, 
                             controlnet_cond=controlnet_cond,
                             conditioning_scale=conditioning_scale,
                             class_labels=class_labels,
                             timestep_cond=timestep_cond,
                             attention_mask=attention_mask,
                             added_cond_kwargs=added_cond_kwargs,
                             cross_attention_kwargs=cross_attention_kwargs,
                             guess_mode=guess_mode,
                             return_dict=return_dict)
    
    
def compile_unet_for_controlnet(model_name_or_path):
    # Compile and save the UNet model
    pipe = StableDiffusionPipeline.from_pretrained(model_name_or_path, torch_dtype=torch.float32)

    # Replace original cross-attention module with custom cross-attention module for better performance
    if use_new_diffusers:
        Attention.get_attention_scores = get_attention_scores
    else:
        CrossAttention.get_attention_scores = get_attention_scores

    # Apply double wrapper to deal with custom return type
    pipe.unet = ControlNetNeuronUNet(ControlNetUNetWrap(pipe.unet))

    # Only keep the model being compiled in RAM to minimze memory pressure
    unet = copy.deepcopy(pipe.unet.unetwrap)
    del pipe

    # Compile unet - FP32
    sample_1b = torch.randn([2, 4, 64, 64])
    timestep_1b = torch.tensor(999).float().expand((1,))
    encoder_hidden_states_1b = torch.randn([2, 77, 768])
    down_block_additional_residuals = [torch.randn([2, 320, 64, 64]),
        torch.randn([2, 320, 64, 64]), torch.randn([2, 320, 64, 64]), torch.randn([2, 320, 32, 32]),
        torch.randn([2, 640, 32, 32]), torch.randn([2, 640, 32, 32]), torch.randn([2, 640, 16, 16]),
        torch.randn([2, 1280, 16, 16]), torch.randn([2, 1280, 16, 16]), torch.randn([2, 1280, 8, 8]),
        torch.randn([2, 1280, 8, 8]), torch.randn([2, 1280, 8, 8])]
        
    mid_block_additional_residual = torch.randn([2, 1280, 8, 8])
    
    example_inputs = (sample_1b, timestep_1b, encoder_hidden_states_1b, 
                    down_block_additional_residuals, mid_block_additional_residual)

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

def compile_controlnet_model(controlnet_model_id):
    cn_model = ControlNetModel.from_pretrained(controlnet_model_id, cache_dir="./controlnet", torch_dtype=torch.float32)
    wrapped_cn_model = NeuronControlNet(cn_model)

    compile_path = controlnet_model_id.replace('/', '-')
    example_inputs = get_example_inputs()
    with torch.no_grad():
        compiled_cn_model = torch_neuronx.trace(
            wrapped_cn_model, 
            example_inputs,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, compile_path),
            compiler_args=["--enable-fast-loading-neuron-binaries"]
        )

    # # Save the compiled vae compiled_cn_model_filename
    compiled_cn_model_filename = os.path.join(COMPILER_WORKDIR_ROOT, f"complied_{compile_path}.pt")
    if not os.path.exists(COMPILER_WORKDIR_ROOT):
      os.makedirs(COMPILER_WORKDIR_ROOT)

    torch_neuronx.async_load(compiled_cn_model)
    torch.jit.save(compiled_cn_model, compiled_cn_model_filename)

    # delete unused objects
    del cn_model

    print(f"Compiled vae {controlnet_model_id}")

def load_compiled_inf_models(pipe, compiled_model_dir, unet_model_file):
    text_encoder_filename = os.path.join(
        compiled_model_dir, 'text_encoder/model.pt')
    unet_filename = unet_model_file
    print('Using unet model file: ', unet_model_file)
    decoder_filename = os.path.join(compiled_model_dir, 'vae_decoder/model.pt')
    post_quant_conv_filename = os.path.join(
        compiled_model_dir, 'vae_post_quant_conv/model.pt')
    safety_model_neuron_filename = os.path.join(
        compiled_model_dir, 'safety_model_neuron/model.pt')

    # Load the compiled UNet onto two neuron cores.
    pipe.unet = ControlNetNeuronUNet(ControlNetUNetWrap(pipe.unet))

    device_ids = [0, 1]
    pipe.unet.unetwrap = torch.jit.load(unet_filename)
    #pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(
    #    unet_filename), device_ids, set_dynamic_batching=False)
    #pipe.unet = ControlNetNeuronUNet(ControlNetUNetWrap(pipe.unet))
    signature = inspect.signature(pipe.unet.unetwrap)
    print('signature of pipe.unet.unetwrap: ', signature)

    # Load other compiled models onto a single neuron core.
    pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
    pipe.text_encoder.neuron_text_encoder = torch.jit.load(
        text_encoder_filename)
    pipe.vae.decoder = torch.jit.load(decoder_filename)
    pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)
    pipe.safety_checker.vision_model = NeuronSafetyModelWrap(
        torch.jit.load(safety_model_neuron_filename))

    return pipe

def load_compiled_txt2img_pipeline(original_model_dir, compiled_model_dir, unet_model_file):
    pipe = StableDiffusionPipeline.from_pretrained(
        original_model_dir, torch_dtype=torch.float32)

    pipe = load_compiled_inf_models(pipe, compiled_model_dir, unet_model_file)

    print("Loaded compiled txt2img models")
    return pipe

def load_compiled_controlnet_model(txt2img_pipe, controlnet_model_id, compiled_cn_model_path):
    controlnet = ControlNetModel.from_pretrained(controlnet_model_id, cache_dir="./controlnet", torch_dtype=torch.float32)

    compiled_controlnet = torch.jit.load(compiled_cn_model_path)
    controlnet_wrapper = ControlNetWrapper(compiled_controlnet)
    controlnet.forward = controlnet_wrapper.forward

    # Note the unet should use the version compiled specifically for controlnet
    controlnet_pipe = StableDiffusionControlNetPipeline(controlnet=controlnet,
                                                      vae=txt2img_pipe.vae,
                                                      text_encoder=txt2img_pipe.text_encoder,
                                                      tokenizer=txt2img_pipe.tokenizer,
                                                      unet=txt2img_pipe.unet,
                                                      scheduler=txt2img_pipe.scheduler,
                                                      safety_checker=txt2img_pipe.safety_checker,
                                                      feature_extractor=txt2img_pipe.feature_extractor)
    
    print('Loaded compiled controlnet model')
    return controlnet_pipe


parser = argparse.ArgumentParser(description="Compile controlnet model")
parser.add_argument("--models-dir", type=str,
                    default="/tmp/models/stable-diffusion", help="Directory for models")
parser.add_argument("--model-name", type=str, help="Name of model for use")
parser.add_argument("--controlnet-dir", type=str,
                    default="/tmp/models/controlnet", help="Directory for controlnet models")
parser.add_argument("--controlnet-model-id", type=str, required=True, 
                    help="huggingface controlnet model id, e.g. 'lllyasviel/sd-controlnet-canny'")
parser.add_argument("--compiled-controlnet-model-path", type=str, 
                    help="The compiled controlnet model path, usually a .pt file")
parser.add_argument("--no-compile", default=False, action="store_true", help="Whether to do compilation")
# parser.add_argument("--lora-dir", type=str, default="/tmp/models/lora", help="Directory for Lora models")

args = parser.parse_args()

original_model_dir = f"{args.models_dir}/{args.model_name}_original"

if not args.no_compile:
  # The base model unet also needs to compile for controlnet
  compile_unet_for_controlnet(original_model_dir)
  # Compile the controlnet model, given the controlnet huggingface model id, e.g.: "lllyasviel/sd-controlnet-canny"
  compile_controlnet_model(args.controlnet_model_id)

# After compilation, try to load and test
compiled_model_dir = f"{args.models_dir}/{args.model_name}_compiled"
unet_model_file = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
txt2img_pipe = load_compiled_txt2img_pipeline(original_model_dir, compiled_model_dir, unet_model_file)

controlnet_pipe = load_compiled_controlnet_model(txt2img_pipe, args.controlnet_model_id, args.compiled_controlnet_model_path)

prompt = 'taylor swift, best quality, extremely detailed'
canny_image = get_canny_image()
output = controlnet_pipe(
    prompt,
    canny_image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    num_inference_steps=20,
)
image = output.images[0]
image = image.cpu().detach()
image = Image.fromarray(image.numpy().astype("uint8"))
image.save('cn_gen.png')
