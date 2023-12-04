#!/bin/bash

uvicorn app:app  \
   --port 9527 --xformers \
    --models-dir /tmp/models/stable-diffusion \
    --model-name darksushi225D \
    --controlnet-dir /tmp/models/controlnet 