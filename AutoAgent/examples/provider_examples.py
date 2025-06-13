"""
Example script demonstrating how to use different LLM providers with AutoAgent.
Make sure to set up your environment variables in .env before running.
"""

import os
from dotenv import load_dotenv
from autoagent import MetaChain
from autoagent.types import Agent, Result

# Load environment variables
load_dotenv()

def create_example_agent(provider: str, model: str) -> Agent:
    """Create an example agent with basic tools."""
    def echo(message: str) -> Result:
        """Simple echo function for demonstration."""
        return Result(value=f"Echo: {message}")

    return Agent(
        name=f"{provider}_agent",
        model=model,
        instructions="You are a helpful assistant that can echo messages.",
        functions=[echo],
        tool_choice="auto"
    )

def run_example(provider: str, model: str, query: str):
    """Run an example query using the specified provider and model."""
    print(f"\n=== Testing {provider.title()} Provider ===")
    print(f"Model: {model}")
    print(f"Query: {query}")
    
    try:
        agent = create_example_agent(provider, model)
        chain = MetaChain()
        
        response = chain.run(
            agent=agent,
            messages=[{"role": "user", "content": query}],
            debug=True
        )
        
        print("\nResponse:")
        for message in response.messages:
            if message.get("role") == "assistant":
                print(f"Assistant: {message.get('content', '')}")
            elif message.get("role") == "tool":
                print(f"Tool ({message.get('name')}): {message.get('content', '')}")
                
    except Exception as e:
        print(f"\nError: {str(e)}")

def main():
    """Run examples for each provider."""
    # Test query
    query = "Hello! Please echo this message: 'Testing different providers'"
    
    # Test Ollama
    if os.getenv("OLLAMA_API_KEY"):
        run_example("ollama", "ollama/llama2", query)
    else:
        print("\nSkipping Ollama example - API key not found")
    
    # Test Perplexity
    if os.getenv("PERPLEXITY_API_KEY"):
        run_example("perplexity", "perplexity/pplx-7b-chat", query)
    else:
        print("\nSkipping Perplexity example - API key not found")
    
    # Test Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        run_example("anthropic", "anthropic/claude-3-opus-20240229", query)
    else:
        print("\nSkipping Anthropic example - API key not found")

if __name__ == "__main__":
    main()
