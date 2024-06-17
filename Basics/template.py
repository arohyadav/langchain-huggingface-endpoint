import os
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain

HUGGINGFACEHUB_API_TOKEN = ""

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

template = "You are a naming consultant for new companies. What is a good name for a company that makes {product}?"

prompt = PromptTemplate.from_template(template)
print(prompt.format(product="colourful socks"))


repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.9, token=HUGGINGFACEHUB_API_TOKEN
)
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run(template))