import os
# from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

HUGGINGFACEHUB_API_TOKEN = ""

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

question = "Who won the FIFA World Cup in the year 1994? "

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.1, token=HUGGINGFACEHUB_API_TOKEN
)
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run(question))