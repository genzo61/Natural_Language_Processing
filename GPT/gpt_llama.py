# import libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
# model tanımlanması
model_name = "gpt2"
# tokenization and model creating
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# metin üretimi için gerekli olan başlangıç text'i

text = "Afternoon,"

# tokenization
inputs = tokenizer.encode(text, return_tensors="pt")

# metin üretimi gerçekleştirilmesi

outputes = model.generate(inputs,max_length = 50)   

# modelin ürettiği tokenları okunabilir hale getirme

generated_text = tokenizer.decode(outputes[0], skip_special_tokens=True)

# üretilen metni print ettirme

print("generated text is : ", generated_text)   

"""
Afternoon, the police arrived at the scene and found the body of a man who had been shot in the head.

The man was taken to the hospital where he was pronounced dead.

The man's family said he had been a
"""
