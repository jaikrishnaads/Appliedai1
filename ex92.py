# Experiment 9.2: Neural Machine Translation with Transformer

import warnings
import os
import torch
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
import sacrebleu

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# Load MarianMT model
model_name = 'Helsinki-NLP/opus-mt-en-fr'
print("Loading Tokenizer and Model...")
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print('English to French Translator loaded.')

# Load OPUS100 dataset
print("Loading OPUS100 Dataset...")
dataset = load_dataset('opus100', 'en-fr')
test_data = dataset['test']
print('Dataset loaded successfully.')

# Translation function
def translate_batch(text_list, num_beams=5, max_length=128):
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True).to(device)
    translated_tokens = model.generate(**inputs, num_beams=num_beams, max_length=max_length)
    outputs = [tokenizer.decode(t, skip_special_tokens=True).replace('_', '').strip() for t in translated_tokens]
    return outputs

# Example translation
example_text = 'Artificial intelligence is changing the world'
example_translation = translate_batch([example_text])[0]
print('English:', example_text)
print('French:', example_translation)

# BLEU evaluation on 500 sentences
print('Calculating BLEU score on first 500 test sentences...')
batch_size = 32
predictions = []
references = []

for i in range(0, 500, batch_size):
    batch_en = [test_data[j]['translation']['en'] for j in range(i, min(i + batch_size, 500))]
    batch_fr = [[test_data[j]['translation']['fr']] for j in range(i, min(i + batch_size, 500))]
    batch_pred = translate_batch(batch_en)
    predictions.extend(batch_pred)
    references.extend(batch_fr)

predictions_clean = [pred.replace('_', '').strip() for pred in predictions]
references_clean = [[ref[0].replace('_', '').strip()] for ref in references]

bleu = sacrebleu.corpus_bleu(predictions_clean, references_clean)
print('BLEU Score on 500 sentences:', bleu.score)

# Interactive translation
print("\n--- Interactive Mode (Type 'exit' to quit) ---")
while True:
    user_input = input('Enter English sentence: ')
    if user_input.lower() == 'exit':
        print('Exiting translator...')
        break
    french_output = translate_batch([user_input])[0]
    print('French translation:', french_output)
