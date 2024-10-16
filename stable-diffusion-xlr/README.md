# Serving Stable Diffusion XL with Refiner

Example images generated using this scenario:

![xlr-doggos.png](..%2Fdoc-assets%2Fxlr-doggos.png)

## Prior Context

If you haven't already, you may wish to review the [README](../stable-diffusion-2/README.md) 
included in the `stable-diffusion-2` directory as it gives a more extensive look at configuration & setup. The rest of this README assumes
prior familiarity and builds upon the ideas found there.

## Additional Cluster Configuration

In moving from [Stable Diffusion 2](https://huggingface.co/stabilityai/stable-diffusion-2) 
to [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) [[white paper](https://arxiv.org/abs/2307.01952)], 
we can expect an increase in model file sizes, required resources & loading timeframes. To accommodate, we need to adjust a few KServe wrapper 
parameters to be a bit more patient with our model load & processing times. To do so, you'll need permissions to edit the `config-defaults` 
ConfigMap within the `knative-serving` namespace. 

Prior to deploying your model, edit the ConfigMap's YAML `data` section so that the section resembles this (values added above `_example`):
```yaml
data:
  max-revision-timeout-seconds: '1800'
  revision-response-start-timeout-seconds: '1600'
  revision-timeout-seconds: '1800'
  _example: |
    [rest of the Example Configuration _example block, as-is]
```

## Model Archival

Similarly to how we bundled the base model for the previous Stable Diffusion 2 [example](../stable-diffusion-2) into our `.mar` archive, we'll bundle the SDXL [base 
model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) for this scenario, however, we'll also & include the [refiner model](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)
which will utilizing a [StableDiffusionXLImg2ImgPipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLImg2ImgPipeline) 
to further refine results.

The [included notebook](nb-archive-sdxl.ipynb) can be executed in-full to prepare the `stable-diffusion.mar` file (~12GB) for download and transfer to bucket.

## `config.properties`

A few changes to this config are needed for easing timeframe allowances.
```text
inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8085
metrics_address=http://0.0.0.0:8082
grpc_inference_port=7070
grpc_management_port=7071
enable_envvars_config=true
install_py_dep_per_model=true
enable_metrics_api=true
metrics_mode=prometheus
NUM_WORKERS=1
number_of_netty_threads=4
job_queue_size=10
max_response_size=30000000
default_response_timeout=2400
default_startup_timeout=2400
model_store=/mnt/models/model-store
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"stable-diffusion":{"1.0":{"defaultVersion":true,"marName":"stable-diffusion.mar","minWorkers":1,"maxWorkers":5,"batchSize":1,"maxBatchDelay":5000,"startupTimeout":2400,"responseTimeout":2400}}}}
```

## Handler Specifics

Pre/Post-Processing remains the same as previous example handler, but our initialization & inference need some updates.

### initialization

For starters, we need to deal with two zip archives that have been included in our `.mar` archive file. Since these are both sizeable, let's use async threads to 
make the timeframe a bit more tolerable:

```python
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
    ...
    
    def initialize(self, ctx):
        ...
        
        logger.info("starting new zip threads")
        zip_files = [model_dir + "/model.zip", model_dir + "/refiner.zip"]
        extract_paths = [model_dir + "/model", model_dir + "/refiner"]
        self.output_queue = queue.Queue()

        for path in extract_paths:
            os.makedirs(path, exist_ok=True)

        self.unzip_files_concurrently(zip_files, extract_paths)
        self.show_threads_queue()

        ...

        self.pipe = StableDiffusionXLPipeline.from_pretrained(model_dir + "/model")
        self.pipe.to(self.device)
        logger.info("Diffusion model from path %s loaded successfully", model_dir)

        self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_dir + "/refiner")
        #self.pipe.to(self.device) #if we try to send both to cuda, 24gb isn't enough for GPU to handle it
        logger.info("Refiner model from path %s loaded successfully", model_dir)
```

### inference

In order to take advantage of our refiner model, we need to chain the output from our base model pipeline into the refiner model
via StableDiffusionXLImg2ImgPipeline. After refiner's done, we can return our image the same as if it came from base model:

```python
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
```

## Querying the model

The [notebook](nb-query-server-sdxlr.ipynb) code for hitting the inference endpoint remains the same as before. Given that we're using an XL model, meaning higher resolution
than SD2, and also chaining into a refiner, expect the query to take some time to respond.

### Inference Service Logs

Similar to before, querying the inference endpoint will reflect the model query in logs. This time, however, notice the transition
into refiner model once the 50 steps have been completed for the base model.

```text
2024-10-15T20:28:21,063 [WARN ] W-9000-stable-diffusion_1.0-stderr MODEL_LOG - 100%|██████████| 50/50 [00:24<00:00, 2.23it/s]
2024-10-15T20:28:21,064 [WARN ] W-9000-stable-diffusion_1.0-stderr MODEL_LOG - 100%|██████████| 50/50 [00:24<00:00, 2.02it/s]
2024-10-15T20:28:21,430 [INFO ] W-9000-stable-diffusion_1.0-stdout MODEL_LOG - done model
2024-10-15T20:28:21,430 [INFO ] W-9000-stable-diffusion_1.0-stdout MODEL_LOG - start refiner
2024-10-15T20:28:27,920 [WARN ] W-9000-stable-diffusion_1.0-stderr MODEL_LOG -
2024-10-15T20:28:42,499 [WARN ] W-9000-stable-diffusion_1.0-stderr MODEL_LOG - 0%| | 0/8 [00:00<?, ?it/s]
2024-10-15T20:28:57,092 [WARN ] W-9000-stable-diffusion_1.0-stderr MODEL_LOG - 12%|█▎ | 1/8 [00:14<01:42, 14.58s/it]
```

![xlr-nb.png](..%2Fdoc-assets%2Fxlr-nb.png)