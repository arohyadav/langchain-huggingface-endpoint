from langchain.agents import load_tools, initialize_agent, AgentType
from langchain_huggingface import HuggingFaceEndpoint
import os

HUGGINGFACEHUB_API_TOKEN = ""

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.1, token=HUGGINGFACEHUB_API_TOKEN
)

tools = load_tools(["human"])

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent_chain.run("What is my brothers name")