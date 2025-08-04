import os
import re
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

# To handle warnings from transformers
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from transformers.utils import logging
logging.set_verbosity_error()

# load Model
generator = pipeline("text-generation", model="openai-community/gpt2", max_new_tokens=30, do_sample=False)
llm = HuggingFacePipeline(pipeline=generator)

# Few-shot examples
examples = [
    {"review": "This movie was fantastic and full of emotion.", "sentiment": "Positive"},
    {"review": "I loved the acting and the story was great.", "sentiment": "Positive"},
    {"review": "The film had good visuals but the plot was weak.", "sentiment": "Negative"},
]

example_prompt = PromptTemplate(
    input_variables=["review", "sentiment"],
    template="Review: {review}\nSentiment: {sentiment}"
)

def build_prompt(review, shots=1):
    fewshot_prompt = FewShotPromptTemplate(
        examples=examples[:shots],
        example_prompt=example_prompt,
        prefix="You are a sentiment classifier.\n\n",
        suffix="i want you say this review '{input_review}' is Sentiment:",
        input_variables=["input_review"],
    )
    return fewshot_prompt.format(input_review=review)

def extract_last_sentiment(text):
    matches = re.findall(r"Sentiment:\s*(Positive|Negative)", text, re.IGNORECASE)
    print(f"** These is All Matches: {matches}")
    return matches[-1].capitalize() if matches else "Unknown"

def classify_review(review, shots=1):
    prompt = build_prompt(review, shots)
    raw_output = llm(prompt)
    print(f"This is full output: {raw_output}")
    return extract_last_sentiment(raw_output)

# Examples
print("1-shot:", classify_review("The movie was excellent from the story to the acting.", shots=1))
print("2-shot:", classify_review("Amazing animation and a great story, One Piece is epic!", shots=2))
print("3-shot:", classify_review("This was boring and poorly written.", shots=3))
