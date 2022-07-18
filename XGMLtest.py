from transformers import XGLMTokenizer, XGLMModel

tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
model = XGLMModel.from_pretrained("facebook/xglm-564M")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

print(outputs[0])
