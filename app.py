import os
import gradio as gr
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

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

# Custom output parsing function
def custom_output_parser(text):
    if "```tool_code" in text:
        tool_code_start = text.find("```tool_code") + len("```tool_code")
        tool_code_end = text.find("```", tool_code_start)
        tool_code = text[tool_code_start:tool_code_end].strip()
        return {"tool_code": tool_code}
    return {"text": text}

# Function to execute tool code
def execute_tool_code(tool_code):
    local_vars = {}
    try:
        # Extract the function call from the tool code
        exec(tool_code, {}, local_vars)
    except Exception as e:
        return f"Error executing tool code: {e}"
    return local_vars.get('result', 'No result from tool execution.')

# Function to run the agent and fallback to search tool if needed
def search(query):
    inputs = {
        "query": query,
        "agent_scratchpad": "",  # Initial empty scratchpad
        "tools": tools,
        "tool_names": ", ".join([tool.name for tool in tools]),
        "intermediate_steps": []  # Initial empty intermediate steps
    }
    try:
        # Attempt to get the answer from the LLM
        output = agent.invoke(inputs)
        parsed_output = custom_output_parser(output)
        
        # Check if the output is a valid answer
        if parsed_output.get("text") and parsed_output["text"].strip():
            return parsed_output["text"]

        # If not a valid answer, proceed with tool code execution
        if "tool_code" in parsed_output:
            tool_code = parsed_output["tool_code"]
            result = execute_tool_code(tool_code)
            return result

        return "No valid answer found."

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
