from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2b"
model_path = "./model_/gemma"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)
