import os
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain, SimpleSequentialChain

HUGGINGFACEHUB_API_TOKEN = ""
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

template = "You are a naming consultant for new companies. What is a good name for a company that makes {product}?"

first_prompt = PromptTemplate.from_template(template)

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.9, token=HUGGINGFACEHUB_API_TOKEN
)

first_chain = LLMChain(prompt=first_prompt, llm=llm)

second_template = "Write a catch phrase for the following company:{company_name}"
second_prompt = PromptTemplate.from_template(second_template)
second_chain = LLMChain(prompt=second_prompt, llm=llm)

overall_chain = SimpleSequentialChain(chains=[first_chain,second_chain], verbose=True)

catchphrase = overall_chain("Colourful Socks")
print(catchphrase)