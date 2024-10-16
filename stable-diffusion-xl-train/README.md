# Serving Stable Diffusion XL with Tuned LoRA Weights

Example images generated using this scenario:

![sdxl-weighted-dinos.png](..%2Fdoc-assets%2Fsdxl-weighted-dinos.png)

## Prior Context

If you haven't already, you may wish to review the contents of `stable-diffusion-2` & `stable-diffusion-xlr` directories as they give more 
extensive looks at configuration & setup. The rest of this README assumes prior familiarity and builds upon the ideas found within those scenarios.

## Tuning with Provided Data

Within our [notebook](nb-archive-trained-sdxl.ipynb), before we create the `.mar` archive, we'll tune the SDXL base model with a dozen pictures 
of a stylized T-Rex striking various poses.

```python
!rm -rf tuning-data/.ipynb_checkpoints

!accelerate launch scripts/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0  \
  --instance_data_dir=tuning-data \
  --output_dir=tuned-weights \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of a Tyrannosaurus Rex" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100 \
  --seed="0" 
```

## Handler Specifics

For this exercise, we'll return to pulling the base model from HuggingFace, but we'll then apply our tuning prior to archival & load accordingly.

### initialization

```python
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"

    ...

    self.pipe = StableDiffusionXLPipeline.from_pretrained(model_name)
    self.pipe.load_lora_weights(model_dir, weight_name="pytorch_lora_weights.safetensors")
    logger.info("Diffusion model loaded successfully")
```

### inference

```python
    logger.info("start model")
    image = self.pipe(
        inputs, guidance_scale=7.5, num_inference_steps=50, height=768, width=768
    ).images
    logger.info("done model")
```

## Querying the model

The [notebook](nb-query-server-trained-sdxl.ipynb) code for hitting the inference endpoint remains similar to previous scenarios.
