# mtp_NUMERICAL_CLAIM_VERIFICATION

## INTRODUCTION
We started with literature review and this are the helpful links. 
Quantemp Repository: https://github.com/factiverse/QuanTemp

Quantemp Trained models: https://drive.google.com/drive/folders/1FmaelDhJ7QwsRTs8H0B4vYliw_qjL7P-

Quantemp Dataset: https://github.com/factiverse/QuanTemp/tree/main/data/raw_data

## STANDARD PIPELINE
<img width="1660" height="186" alt="image" src="https://github.com/user-attachments/assets/7d3ac29d-24d9-448b-8642-2c2de3b42575" />

## CLAIM DECOMPOSITION
It is a method where we decompose a complex claim into sub-questions. Quantmep has decomposed claims into YES-NO type questions. Entity mentioned in question so when we do retrieval as the retrieval is keywords matching based retrieval it will retrieve the evidences which have keyword match from question. This does not bring contrastive evidences. Example: "Did 11.3% work in informal economy?" → snippets with "11.3%" regardless of context


### WH-QUESTION GENERATION
We proposed the strategy of decomposing claims into Wh-Questions. 
```python
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
tokenizer = T5TokenizerFast.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base").to('cuda')
model.load_state_dict(torch.load("/content/drive/MyDrive/NUMERICAL CLAIM VERIFICATION/Finetune_Question_Generation_22kData.pth"))

def run_model(input_string, **generator_args):
  generator_args = {
  "max_length": 512, #256,
  "num_beams": 20,
  "length_penalty": 1.5,
  "no_repeat_ngram_size": 3,
  "early_stopping": True,
  "num_return_sequences" : 20
  }
  input_string = "generate questions: " + input_string + " </s>"
  input_ids = tokenizer.encode(input_string, return_tensors="pt").to('cuda')
  res = model.generate(input_ids, **generator_args)
  output = tokenizer.batch_decode(res, skip_special_tokens=True)
  output = [item.split("<sep>") for item in output]
  return output
sentence = 'Twitter owner Elon Musk tweeted that Democrats paid former Twitter CEO Jack Dorsey "millions of dollars" to block negative information about the Bidens in 2020.'
questions = run_model(sentence)

```

## MMR BASED RERANKING
## EVIDENCE RETRIEVAL WITH APIS
## QWEN MODEL TRAINING AND TESTING
## ROBERTA MODEL TRAINING AND TESTING
## TEMPORAL QUERY REFINEMENT
## PAGE CONTENT RETRIEVAL
## 3 ENCODER STRATEGY TRAINING AND RESTING
## DATASETS
