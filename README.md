# Generative Artificial Intelligence

## Introduction
Large Language Models (LLMs) are types of machine learning models that are built based on large text datasets to generate various responses.

This example shows how to build an Android application with TensorFlow Lite to run a Keras LLM model and provides suggestions for model optimization using techniques such as quantization.

This example has been open-sourced and provides an application framework for Android in which LLMs compatible with TFLite can be integrated. Here are two demonstrations:
* In Figure 1, a Keras GPT-2 model was used to perform text completion tasks on the device.

<p align="center">
  <img src="figures/fig1.gif" width="300">
</p>
Figure 1: Example of running the Keras GPT-2 model (converted from this Codelab) on the device to perform text completion on a Pixel 7. The demo shows real latency without hardware acceleration.

## Guides
### Step 1. Train a language model using Keras

For this demonstration, we will use KerasNLP to obtain the GPT-2 model. KerasNLP is a library that contains state-of-the-art pretrained models for natural language processing tasks and can support users throughout their development cycle. You can view the list of available models in the [KerasNLP](https://github.com/keras-team/keras-nlp/tree/master/keras_nlp/models) repository. Workflows are built from modular components that have state-of-the-art pretrained weights and architectures when used directly, and are easily customizable when more control is needed. The GPT-2 model can be created with the following steps:

```python
gpt2_tokenizer = keras_nlp.models.GPT2Tokenizer.from_preset("gpt2_base_en")

gpt2_preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
  "gpt2_base_en",
  sequence_length=256,
  add_end_token=True,
)

gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
  "gpt2_base_en", 
  preprocessor=gpt2_preprocessor,
)
```

You can check the complete implementation of the GPT-2 model [en GitHub](https://github.com/keras-team/keras-nlp/tree/master/keras_nlp/models/gpt2).


### Step 2. Convert a Keras model to a TFLite model

Start with the generate() function of GPT2CausalLM that performs the conversion. Wrap the generate() function to create a concrete TensorFlow function:

```python
@tf.function
def generate(prompt, max_length):
  # prompt: input to the LLM in string format
  # max_length: maximum length of the generated tokens
  return gpt2_lm.generate(prompt, max_length)
concrete_func = generate.get_concrete_function(tf.TensorSpec([], tf.string), 100)
```

Now define a helper function that will run inference with an input and a TFLite model. TensorFlow text operations are not built-in operations in TFLite, so you’ll need to add these custom operations for the interpreter to be able to perform inference on this model. This helper function accepts an input and a function that performs the conversion, i.e., the generator() function defined earlier.

```python
def run_inference(input, generate_tflite):
  interp = interpreter.InterpreterWithCustomOps(
    model_content=generate_tflite,
    custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS)
  interp.get_signature_list()

  generator = interp.get_signature_runner('serving_default')
  output = generator(prompt=np.array([input]))
```

Now you can convert the model:

```python
gpt2_lm.jit_compile = False
converter = tf.l

ite.TFLiteConverter.from_concrete_functions(
  [concrete_func],
  gpt2_lm)

converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TFLite operations
  tf.lite.OpsSet.SELECT_TF_OPS, # enable TF operations
]
converter.allow_custom_ops = True
converter.target_spec.experimental_select_user_tf_ops = [
  "UnsortedSegmentJoin",
  "UpperBound"
]
converter._experimental_guarantee_all_funcs_one_use = True
generate_tflite = converter.convert()
run_inference("I am enjoying", generate_tflite)
```

### Step 3. Quantization

TensorFlow Lite has implemented an optimization technique called quantization that can reduce model size and speed up inference. Through the quantization process, 32-bit floating point numbers are mapped to smaller 8-bit integers, reducing the model size by a factor of 4 for more efficient execution on modern hardware. 

There are several ways to perform quantization in TensorFlow. You can visit the [TFLite Model Optimization](https://www.tensorflow.org/lite/performance/model_optimization) and [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization) pages for more information.

Below is a brief explanation of the types of quantization.

Here, you will use post-training dynamic quantization on the GPT-2 model by setting the converter’s optimization flag to `tf.lite.Optimize.DEFAULT`, and the rest of the conversion process is the same as previously described. 

We observed that with this quantization technique, the latency is approximately 6.7 seconds on a Pixel 7 with a maximum output length set to 100.

```python
gpt2_lm.jit_compile = False
converter = tf.lite.TFLiteConverter.from_concrete_functions(
  [concrete_func],
  gpt2_lm)

converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enables TFLite operations
  tf.lite.OpsSet.SELECT_TF_OPS, # enables TF operations
]
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.experimental_select_user_tf_ops = [
  "UnsortedSegmentJoin",
  "UpperBound"
]
converter._experimental_guarantee_all_funcs_one_use = True
quant_generate_tflite = converter.convert()
run_inference("I'm enjoying", quant_generate_tflite)

with open('gpt2_cuantificado.tflite', 'wb') as f:
  f.write(quant_generate_tflite)
```



### Step 4. Integration with the Android App

You can clone this repository and replace `android/app/src/main/assets/autocomplete.tflite` with your converted `quant_generate_tflite` file.

## Safety and Responsible AI

As mentioned in the original [OpenAI GPT-2](https://openai.com/research/better-language-models) announcement, there are [notable caveats and limitations](https://github.com/openai/gpt-2#some-caveats) with the GPT-2 model. In fact, large language models (LLMs) in general face known challenges such as hallucinations, fairness, and bias. This is because these models are trained on real-world data, which causes them to reflect real-world issues.

This codelab is created solely to demonstrate how to build an application powered by LLMs using TensorFlow tools. The model produced in this codelab is for educational purposes only and is not intended for production use. Production use of LLMs requires careful selection of training datasets and comprehensive safety mitigations.

One of the features offered in this Android application is offensive language filtering, which rejects inappropriate model inputs or outputs. If any inappropriate language is detected, the app will block the action.

For more information on responsible AI in the context of LLMs, be sure to watch the technical session Safe and Responsible Development with Generative Language Models from Google I/O 2023 and explore the [Responsible AI Toolkit](https://www.tensorflow.org/responsible_ai).
