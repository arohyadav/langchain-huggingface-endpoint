import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMMathChain
from langchain_community.agent_toolkits import load_tools
from langchain_community.agent_toolkits.experimental import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain_community.tools import SerpAPIWrapper, WikipediaAPIWrapper
from langchain.agents.tools import Tool

# Set your HuggingFace API token
HUGGINGFACEHUB_API_TOKEN = ""

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# Initialize LLM
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.1, token=HUGGINGFACEHUB_API_TOKEN
)

# Initialize tools
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
search = SerpAPIWrapper()
wikipedia = WikipediaAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for when you need to lookup facts and statistics"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="Useful for when you need to answer questions about math"
    ),
]

# Define the prompt
prompt = "Where are the next summer olympics going to be hosted? What is the population of that country raised to the 0.43 power?"

# Initialize planner and executor
planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)

# Assuming you want to run this as a script, add the execution part
if __name__ == "__main__":
    plan_and_execute = PlanAndExecute(planner=planner, executor=executor)
    result = plan_and_execute(prompt)
    print(result)
