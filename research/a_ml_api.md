# A Machine Learning API - Inference and Bench Marking

Accessing inference via rest api has certain benefits with regard to the evaluation and bechmark of models. Variation of models can be access through different url addresses, e.g:

```cmd
url_access_point = http://localhost:8080/run_inference/[dataset]/[model_name]
```

This way it could be possible to test several models easier without the need to initialize and configure them in the inference script:

```python
model_1 = url_access_point
model_2 = url_access_point

for input in inputs:
    result_1 = engine(model_1, input)
    result_2 = engine(model_2, input)

    // POST results to benchmark database
```

Different models are then to be added like a post in a blog?

### refs:

* [hugging-face pipelines](https://huggingface.co/docs/transformers/pipeline_tutorial)
* [hugging-face pipeline_webserver](https://huggingface.co/docs/transformers/pipeline_webserver)
