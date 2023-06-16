import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.memory import ConversationBufferMemory
#from IPython.display import Markdown

# Set environment variables
os.environ['GOOGLE_API_KEY'] = '...'
os.environ['OPENAI_API_KEY'] = '...'
os.environ['GOOGLE_CSE_ID'] = '...'

# A conversation buffer (memory) & import llm of choice
memory = ConversationBufferMemory()
llm = ChatOpenAI()

# Provide access to a list of tools that the agents will use
tools = load_tools(['wikipedia', 'google-search', 'llm-math'], llm=llm)

# initialise the agents & make all the tools and llm available to it
agent = initialize_agent(tools, llm, agent='zero-shot-react-description', verbose=True, memory=memory)

# provide a prompt and you are done!
agent.run("Find the number of IPL titles won by MS Dhoni & find it's cube root")
