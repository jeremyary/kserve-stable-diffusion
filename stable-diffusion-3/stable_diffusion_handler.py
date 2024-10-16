import logging
import orjson
from abc import ABC

import diffusers
import torch
from diffusers import StableDiffusion3Pipeline
import numpy as np

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Diffusers version %s", diffusers.__version__)


class DiffusersHandler(BaseHandler, ABC):
    """
    Diffusers handler class for text to image generation.
    """

    def __init__(self):
        self.initialized = False


    def initialize(self, ctx):
        """In this initialize function, the Stable Diffusion model is loaded and
        initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        model_name = "stabilityai/stable-diffusion-3-medium-diffusers"

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        hf_token = "[your-token-here]"
        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_name, token=hf_token)
        self.pipe.enable_model_cpu_offload()
        self.pipe.load_lora_weights(model_dir, weight_name="pytorch_lora_weights.safetensors")
        logger.info("Diffusion model loaded successfully")
        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, of the user's prompt.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of prompts.
        """
        inputs = []

        for _, data in enumerate(requests):
            input_text = data['body']['inputs'][0]['data'][0]
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            logger.info("Received text: '%s'", input_text)
            inputs.append(input_text)
        return inputs

    def inference(self, inputs):
        """Generates the image relevant to the received text.
        Args:
            input_batch (list): List of Text from the pre-process function is passed here
        Returns:
            list : It returns a list of the generate images for the input text
        """
        # Handling inference for sequence_classification.
        logger.info("start model")
        image = self.pipe(
            inputs, guidance_scale=7.5, num_inference_steps=50, height=768, width=768
        ).images
        logger.info("done model")
        return image

    def postprocess(self, inference_output):
        """Post Process Function converts the generated image into Torchserve readable format.
        Args:
            inference_output (list): It contains the generated image of the input text.
        Returns:
            (list): Returns a list of the images.
        """
        img_nparray = np.array(inference_output[0])
        response = {
            "id": "42",
            "outputs": [
                {
                    "name": "output0",
                    "datatype": "fp16",
                    "shape": img_nparray.shape,
                    "data": img_nparray.tolist()
                }
            ]
        }
        logger.info("returning nparray of shape %s", img_nparray.shape)
        return [orjson.dumps(response)]