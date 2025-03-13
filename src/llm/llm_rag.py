from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="mistralai/Mistral-7B")

def ask_ai(question, context):
    return qa_pipeline({"question": question, "context": context})

context = "Irrigation is optimal when soil moisture is between 30% and 60%."
response = ask_ai("What is the best irrigation schedule?", context)
print(response)
