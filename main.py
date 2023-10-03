import os
import sys
from dotenv import find_dotenv, load_dotenv # for api key
from transformers import pipeline # this will allow us to download hunginface models to our machine
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

# this puts our keys etc into the namespace
load_dotenv(find_dotenv())

# image to text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base",  max_new_tokens=64)
    text = image_to_text(url)[0]['generated_text']
    print("Salesforce's (variable size!) image-to-text model run on our image: ", text)
    return text

# llm
def generate_story(scenario):
    # a template for the prompt
    # this is like an f-string for inserting a string variable into another string

    # template = """
    # You tell fantasy stories;
    # You generate a story based on some context;

    # CONTEXT: {scenario}
    # STORY:
    # """

    template = "Here's some context: {scenario}. And here's a really funny story. Story: "

    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    # hub_llm = HuggingFaceHub(repo_id="google/flan-T5-base")
    # hub_llm = HuggingFaceHub(repo_id="gpt2")
    # hub_llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.55, "max_length": 256})
    # repo_id = "databricks/dolly-v2-3b" # killed before response
    # repo_id = "tiiuae/falcon-40b" # killed before response
    # repo_id = "facebook/opt-125m"
    # repo_id = "facebook/opt-2.7b"
    # repo_id="google/flan-T5-base"
    # repo_id="google/flan-t5-xxl"
    repo_id="google/flan-T5-large"
    hub_llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.8, "max_length": 128})

    story_llm = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
    # story_llm = LLMChain(prompt=prompt, llm=hub_llm)

    # story = story_llm.predict(scenario=scenario)
    story = story_llm.run(scenario=scenario)

    print("the story generated:", story)
    return story


# run it
scenario = img2text("digital_physics_still_3.png")
story = generate_story(scenario)

# text to speech (to do later)



