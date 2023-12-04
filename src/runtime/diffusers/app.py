from diffusers.utils import load_image
from PIL import Image
import cv2
import diffusers
from packaging import version
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
import numpy as np
import torch_neuronx
import torch.nn as nn
import torch
import requests
import argparse
import os
from io import BytesIO
import base64
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError
import datetime
import uuid
import json
import jsonify

os.environ["NEURON_FUSE_SOFTMAX"] = "1"


# Compatibility for diffusers<0.18.0
diffusers_version = version.parse(diffusers.__version__)
use_new_diffusers = diffusers_version >= version.parse('0.18.0')
if use_new_diffusers:
    from diffusers.models.attention_processor import Attention
else:
    from diffusers.models.cross_attention import CrossAttention


s3_client = boto3.client('s3')
s3_bucket = "test-aimaginairy-us-east-1"

class ImageRequest(BaseModel):
    req_id: str
    prompt: str
    negative_prompt: str = None
    controlnet_model: str = None

def generate_image_uri(req_id, req_key, file_extension="png"):
    return f"{req_id}/{req_key}.{file_extension}"

def generate_req_key():
    unique_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{timestamp}-{unique_id}"

def write_req_handle(request, req_key, status=1, error_msg=""):
    path = f"{request.req_id}/{req_key}.json"
    s3_uri = f"s3://{s3_bucket}/{path}"
    image_url = generate_image_uri(request.req_id, req_key)
    handle_data = {
      "parameters": {
          "req_id": request.req_id,
          "status": status,
          "image_url": image_url,
          "error_msg": error_msg,
      }
    }

    try:
        s3_client.put_object(Bucket=s3_bucket, Key=path, Body=json.dumps(handle_data))
        print(f"JSON data successfully uploaded to {s3_uri}")
    except Exception as e:
        print(f"Error uploading JSON to S3: {e}")

    return s3_uri

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

def get_img2img_img():
    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((512, 512))

    return init_image 

def write_image_to_s3(image, req_id, req_key):
    path = generate_image_uri(req_id, req_key)
    s3_uri = f"s3://{s3_bucket}/{path}"
    buf = BytesIO()
    image.save(buf, format='PNG')
    image_data = buf.getvalue()
    try:
        s3_client.put_object(Bucket=s3_bucket, Key=path, Body=image_data)
        print(f"Image successfully uploaded to {s3_uri}")
    except Exception as e:
        print(f"Error uploading image to S3: {e}")

    return s3_uri

async def gen_txt2image_task(request: ImageRequest, req_key):
  try:
    prompt = request.prompt
    image = txt2img_pipe(prompt).images[0]
    write_image_to_s3(image, request.req_id, req_key)
  except Exception as e:
      # Any errors will be written to the request handle file
      write_req_handle(request, req_key, status=1, error_msg=str(e))
   
async def get_img2img_task(request: ImageRequest, req_key):
  try:
    prompt = request.prompt
    controlent_model = request.controlnet_model
    if controlent_model:
        pipe = load_compiled_controlnet_pipeline(txt2img_pipe, controlnet_dir, controlent_model)
        image = pipe(prompt, get_canny_image(), num_inference_steps=20).images[0]
    else:
        pipe = img2img_pipe
        image = pipe(prompt=prompt, image=get_img2img_img(), strength=0.75, guidance_scale=7.5).images[0]

    write_image_to_s3(image, request.req_id, req_key)
  except Exception as e:
      # Any errors will be written to the request handle file
      write_req_handle(request, req_key, status=1, error_msg=str(e))
   

# Start a FastAPI app to listen for requests
app = FastAPI()

is_ready = False

@app.get("/ready")
def readiness_check():
    return jsonify({"ready": is_ready})

@app.post("/txt2img")
def txt2img(request: ImageRequest, background_tasks: BackgroundTasks):
    try:
        req_key = generate_req_key(request.req_id)
        # async generate image
        background_tasks.add_task(gen_txt2image_task, request, req_key)

        # Write request handle file to S3
        s3_uri = write_req_handle(request, req_key)

        # Return to user for polling
        return {"handle": s3_uri}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.route("/img2img")
def img2img(request: ImageRequest, background_tasks: BackgroundTasks):
    try:
       
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None, return_dict=False):
        sample = self.unetwrap(sample, timestep.float().expand(
            (sample.shape[0],)), encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)


class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        out_tuple = self.unet(
            sample, timestep, encoder_hidden_states, return_dict=False)
        return out_tuple


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

def parse_request(request):
    content = request.json
    prompt = content["prompt"]
    controlent_model = content.get("controlnet_model", "")
    return prompt, controlent_model

def load_compiled_inf_models(pipe, compiled_model_dir):
    text_encoder_filename = os.path.join(
        compiled_model_dir, 'text_encoder/model.pt')
    unet_filename = os.path.join(compiled_model_dir, 'unet/model.pt')
    decoder_filename = os.path.join(compiled_model_dir, 'vae_decoder/model.pt')
    post_quant_conv_filename = os.path.join(
        compiled_model_dir, 'vae_post_quant_conv/model.pt')
    safety_model_neuron_filename = os.path.join(
        compiled_model_dir, 'safety_model_neuron/model.pt')

    # Load the compiled UNet onto two neuron cores.
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
    device_ids = [0, 1]
    pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(
        unet_filename), device_ids, set_dynamic_batching=False)

    # Load other compiled models onto a single neuron core.
    pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
    pipe.text_encoder.neuron_text_encoder = torch.jit.load(
        text_encoder_filename)
    pipe.vae.decoder = torch.jit.load(decoder_filename)
    pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)
    pipe.safety_checker.vision_model = NeuronSafetyModelWrap(
        torch.jit.load(safety_model_neuron_filename))

    return pipe


def load_compiled_txt2img_pipeline(original_model, compiled_model_dir):
    pipe = StableDiffusionPipeline.from_pretrained(
        original_model, torch_dtype=torch.float32)

    pipe = load_compiled_inf_models(pipe, compiled_model_dir)

    print("Loaded compiled txt2img models")
    return pipe


def load_complied_img2img_pipeline(txt2img_pipe):
    # We reuse components of the txt2img pipeline to build the img2img pipeline for memory efficiency.
    img2img_pipeline = StableDiffusionImg2ImgPipeline(**txt2img_pipe.components)

    print("Loaded compiled img2img models")
    return img2img_pipeline

def load_compiled_controlnet_pipeline(txt2img_pipe, compiled_model_dir, controlnet_model):
    controlnet = ControlNetModel.from_pretrained(controlnet_model, cache_dir=compiled_model_dir, torch_dtype=torch.float32)

    controlnet_pipe = StableDiffusionControlNetPipeline(controlnet=controlnet,
                                                      vae=txt2img_pipe.vae,
                                                      text_encoder=txt2img_pipe.text_encoder,
                                                      tokenizer=txt2img_pipe.tokenizer,
                                                      unet=txt2img_pipe.unet,
                                                      scheduler=txt2img_pipe.scheduler,
                                                      safety_checker=txt2img_pipe.safety_checker,
                                                      feature_extractor=txt2img_pipe.feature_extractor)


    #pipe.enable_model_cpu_offload()
    #pipe.enable_xformers_memory_efficient_attention()

    print(f"Loaded compiled controlnet models: {controlnet_model}")
    return controlnet_pipe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch API for model serving")
    parser.add_argument("--port", type=int, default=9527,
                        help="Port to listen on")
    parser.add_argument("--xformers", action='store_true',
                        help="Enable xformers")
    parser.add_argument("--models-dir", type=str,
                        default="/tmp/models/stable-diffusion", help="Directory for models")
    parser.add_argument("--model-name", type=str, help="Name of model for use")
    parser.add_argument("--controlnet-dir", type=str,
                        default="/tmp/models/controlnet", help="Directory for controlnet models")
    # parser.add_argument("--lora-dir", type=str, default="/tmp/models/lora", help="Directory for Lora models")
    # parser.add_argument("--vae-dir", type=str, default="/tmp/models/vae", help="Directory for VAE models")

    args = parser.parse_args()

    try:
        original_model_dir = f"{args.models_dir}/{args.model_name}_original"
        compiled_model_dir = f"{args.models_dir}/{args.model_name}_compiled"
        txt2img_pipe = load_compiled_txt2img_pipeline(original_model_dir, compiled_model_dir)
        img2img_pipe = load_complied_img2img_pipeline(txt2img_pipe)
        controlnet_dir = args.controlnet_dir
    except Exception as e:
        print(f"Error loading models: {e}")
        exit(1)

    is_ready = True
    app.run(debug=False, host='0.0.0.0', port=args.port)
