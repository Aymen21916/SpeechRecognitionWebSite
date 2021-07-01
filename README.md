---
language: ar
datasets:
- common_voice 
metrics:
- wer
tags:
- audio
- automatic-speech-recognition
- speech
- xlsr-fine-tuning-week
license: apache-2.0
model-index:
- name: XLSR Wav2Vec2 Arabic by Othmane Rifki
  results:
  - task: 
      name: Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Common Voice ar
      type: common_voice
      args: ar
    metrics:
       - name: Test WER
         type: wer
         value: 46.77
---

# Wav2Vec2-Large-XLSR-53-Arabic

Fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on Arabic using the [Common Voice](https://huggingface.co/datasets/common_voice). 
When using this model, make sure that your speech input is sampled at 16kHz.

## Usage

The model can be used directly (without a language model) as follows:

```python
import librosa
import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

test_dataset = load_dataset("common_voice", "ar", split="test[:2%]")

processor = Wav2Vec2Processor.from_pretrained("kmfoda/wav2vec2-large-xlsr-arabic")
model = Wav2Vec2ForCTC.from_pretrained("kmfoda/wav2vec2-large-xlsr-arabic")

resamplers = {  # all three sampling rates exist in test split
    48000: torchaudio.transforms.Resample(48000, 16000),
    44100: torchaudio.transforms.Resample(44100, 16000),
    32000: torchaudio.transforms.Resample(32000, 16000),
}

def prepare_example(example):
    speech, sampling_rate = torchaudio.load(example["path"])
    example["speech"] = resamplers[sampling_rate](speech).squeeze().numpy()
    return example

test_dataset = test_dataset.map(prepare_example)

inputs = processor(test_dataset["speech"][:2], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)

print("Prediction:", processor.batch_decode(predicted_ids))
print("Reference:", test_dataset["sentence"][:2])
```

## Evaluation

The model can be evaluated as follows on the Arabic test data of Common Voice. 

```python
import librosa
import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re

test_dataset = load_dataset("common_voice", "ar", split="test") 
wer = load_metric("wer")
processor = Wav2Vec2Processor.from_pretrained("kmfoda/wav2vec2-large-xlsr-arabic") 
model = Wav2Vec2ForCTC.from_pretrained("kmfoda/wav2vec2-large-xlsr-arabic")
model.to("cuda")

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\؟\_\؛\ـ\—]'

resamplers = {  # all three sampling rates exist in test split
    48000: torchaudio.transforms.Resample(48000, 16000),
    44100: torchaudio.transforms.Resample(44100, 16000),
    32000: torchaudio.transforms.Resample(32000, 16000),
}

def prepare_example(example):
    speech, sampling_rate = torchaudio.load(example["path"])
    example["speech"] = resamplers[sampling_rate](speech).squeeze().numpy()
    return example

test_dataset = test_dataset.map(prepare_example)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

        pred_ids = torch.argmax(logits, dim=-1)
        batch["pred_strings"] = processor.batch_decode(pred_ids)
        return batch

result = test_dataset.map(evaluate, batched=True, batch_size=8)

print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"])))

```

**Test Result**: 52.53


## Training

The Common Voice `train`, `validation` datasets were used for training.

The script used for training can be found [here](https://huggingface.co/kmfoda/wav2vec2-large-xlsr-arabic/tree/main) 