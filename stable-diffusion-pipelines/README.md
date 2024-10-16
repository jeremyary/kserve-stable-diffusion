# Loading & Querying Stable Diffusion 2 via HuggingFace `diffusers` Pipeline

Example images generated using this scenario:

![sd2-robos.png](..%2Fdoc-assets%2Fsd2-robos.png)

## HuggingFace Pipelines

> HuggingFace pipelines are a great and easy way to use models for inference. These pipelines are objects 
> that abstract most of the complex code from the library, offering a simple API dedicated to several tasks, including 
> Named Entity Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering.

<sup>[source](https://huggingface.co/docs/transformers/en/main_classes/pipelines)</sup>

## Notebook-Only Example of Stable Diffusion Use

Thanks to some excellent HuggingFace python libraries, loading & using the Stable Diffusion model can be done with just a few lines of code in a [notebook](sd2-diffusers-pipelines.ipynb).

```python
from diffusers import StableDiffusionPipeline

# pull a snapshot of the model from HuggingFace
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")

# utilize the GPU accelerator we've set up
pipe.to("cuda")

prompt = "a toy robot wearing a red fedora"

pipe(prompt).images[0]
```