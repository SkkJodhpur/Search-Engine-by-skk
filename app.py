import os
import gradio as gr
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate  # Import PromptTemplate

# Set up Google API keys from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')

# Check if the keys are loaded correctly
if GOOGLE_API_KEY is None or SERPER_API_KEY is None:
    raise ValueError("Please set the GOOGLE_API_KEY and SERPER_API_KEY environment variables.")

# Initialize embeddings and language model
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Load tools
tools = load_tools(["serpapi", "llm-math"], llm=llm, serpapi_api_key=SERPER_API_KEY)

# Define a prompt using PromptTemplate
prompt = PromptTemplate(
    input_variables=["query"],
    template="You are a helpful assistant that answers questions based on the provided tools. Question: {query}"
)

# Initialize the agent with the prompt
agent = create_react_agent(tools, llm, prompt)

# Function to run the agent
def search(query):
    output = agent.run(query)
    return output

# Create the Gradio interface
iface = gr.Interface(
    fn=search,
    inputs=gr.Textbox(label="Enter your search query", placeholder="What is the hometown of the reigning men's U.S. Open champion?"),
    outputs="text",
    title="Custom Search Engine",
    description="A search engine powered by LangChain and Google Generative AI. Enter your query to get started!",
    theme="default"
)

# Launch the interface
iface.launch()