import os
import gradio as gr
from google.colab import userdata
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.agents import AgentType, initialize_agent, load_tools

# Install necessary packages
!pip install google-search-results
!pip install -U langchain-community

# Set up Google API keys
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize embeddings and language model
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Set up Serper API key and tools
SERPER_API_KEY = userdata.get('SERPER_API_KEY')
tools = load_tools(["serpapi", "llm-math"], llm=llm, serpapi_api_key=SERPER_API_KEY)

# Initialize the agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Function to run the agent
def search(query):
    output = agent.run(query)
    return output

# Create the Gradio interface
iface = gr.Interface(fn=search, 
                     inputs=gr.inputs.Textbox(label="Enter your search query", placeholder="What is the hometown of the reigning men's U.S. Open champion?"), 
                     outputs="text",
                     title="Custom Search Engine",
                     description="A search engine powered by LangChain and Google Generative AI. Enter your query to get started!",
                     theme="default")

# Launch the interface
iface.launch()