import os
import gradio as gr
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
import requests

# Set up Google API keys from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')

# Initialize embeddings and language model
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Load tools
tools = load_tools(["serpapi", "llm-math"], llm=llm, serpapi_api_key=SERPER_API_KEY)

# Define a prompt using PromptTemplate with required variables
prompt = PromptTemplate(
    input_variables=["query", "agent_scratchpad", "tools", "tool_names"],
    template=(
        "You are a helpful assistant that answers questions based on the provided tools.\n"
        "Tools available: {tool_names}\n"
        "Current tools: {tools}\n"
        "Scratchpad: {agent_scratchpad}\n"
        "Question: {query}"
    )
)

# Initialize the agent with the prompt
agent = create_react_agent(tools=tools, llm=llm, prompt=prompt)

# Function to search Google using SERP API
def google_search(query):
    response = requests.get(
        f"https://api.serpapi.com/search.json?q={query}&api_key={SERPER_API_KEY}"
    )
    results = response.json()
    # Extract relevant information from the results
    if "organic_results" in results:
        return results["organic_results"][0]["snippet"]  # Return the snippet of the first result
    return "No results found."

# Function to run the agent
def search(query):
    inputs = {
        "query": query,
        "agent_scratchpad": "",  # Initial empty scratchpad
        "tools": tools,
        "tool_names": ", ".join([tool.name for tool in tools]),
        "intermediate_steps": []  # Initial empty intermediate steps
    }
    
    try:
        # Debug: Print inputs
        print("Inputs to the agent:", inputs)
        
        # Attempt to get the answer from the LLM
        output = agent.invoke(inputs)
        
        # Debug: Print output
        print("Output from the agent:", output)
        
        # Check if output is empty or not generated
        if not output or "error" in output.lower():
            print("LLM did not generate a valid output, searching Google...")
            output = google_search(query)
        
        return output

    except Exception as e:
        # Print the exception and the inputs for debugging
        print(f"Error: {e}")
        print("Inputs:", inputs)
        return str(e)

# Create the Gradio interface
iface = gr.Interface(
    fn=search,
    inputs=gr.Textbox(label="Enter your search query", placeholder="What is the hometown of the reigning men's U.S. Open champion?"),
    outputs="text",
    title="Custom Search Engine",
    description="A search engine powered by LangChain and Google Generative AI. Enter your query to get started!",
    theme="default"
)

# Launch the interface with share=True for a public link
iface.launch(share=True)
