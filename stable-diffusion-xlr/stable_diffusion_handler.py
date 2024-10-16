import logging
import zipfile
import concurrent.futures
import os
import queue
import orjson
from abc import ABC

import diffusers
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
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

    def unzip_file(self, zip_path, extract_to):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            self.output_queue.put(f'Unzipped {zip_path} to {extract_to}')
        except Exception as e:
            self.output_queue.put(f'Error unzipping {zip_path}: {str(e)}')

    def unzip_files_concurrently(self, zip_files, extract_paths):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.unzip_file, zip_file, extract_to)
                for zip_file, extract_to in zip(zip_files, extract_paths)
            ]
            concurrent.futures.wait(futures)

    def show_threads_queue(self):
        while not self.output_queue.empty():
            print(self.output_queue.get())


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

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        logger.info("starting new zip threads")
        zip_files = [model_dir + "/model.zip", model_dir + "/refiner.zip"]
        extract_paths = [model_dir + "/model", model_dir + "/refiner"]
        self.output_queue = queue.Queue()

        for path in extract_paths:
            os.makedirs(path, exist_ok=True)

        self.unzip_files_concurrently(zip_files, extract_paths)
        self.show_threads_queue()

        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.
        self.pipe = StableDiffusionXLPipeline.from_pretrained(model_dir + "/model")
        self.pipe.to(self.device)
        logger.info("Diffusion model from path %s loaded successfully", model_dir)

        self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_dir + "/refiner")
        logger.info("Refiner model from path %s loaded successfully", model_dir)

        self.n_steps = 40
        self.high_noise_frac = 0.8
        self.negative_prompt = ["worst quality, normal quality, low quality, low res, blurry"]

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

        logger.info("start refiner")
        image = self.refiner(
            prompt=inputs,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.n_steps,
            denoising_start=self.high_noise_frac,
            image=image
        ).images
        logger.info("done refiner")

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