import pprint
from langchain.agents import load_tools, initialize_agent
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain, SimpleSequentialChain
import os

HUGGINGFACEHUB_API_TOKEN = ""

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

prompt = "When was the 3rd president of the united states born? What is that year raised to the power of 3?"

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.1, token=HUGGINGFACEHUB_API_TOKEN
)

tools = load_tools(["wikipedia","llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run(prompt)