"""
Token Tracking and Tokenization Module

This module demonstrates how to track and count tokens when working with LLMs,
specifically using OpenAI's GPT models.

Overview:
- Tokenization basics with tiktoken
- Token counting for API requests and responses
- Practical examples with OpenAI's Chat API
"""

import os
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv


def basic_tokenization_example():
    """
    Basic Tokenization Example
    
    Demonstrates basic tokenization using tiktoken. Encodes a simple string
    and examines how it's broken down into tokens.
    """
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = encoding.encode("Testing Tokenomies")
    
    print("Number of tokens:", len(tokens))
    for i, token in enumerate(tokens):
        print(f"Token {i}: {encoding.decode([token])}")


def calculate_input_tokens(messages, encoding):
    """
    Calculate input tokens from a list of messages.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        encoding: tiktoken encoding object
        
    Returns:
        int: Number of input tokens
    """
    input_text = ""
    for message in messages:
        input_text += message["role"] + ": " + message["content"] + "\n"
    input_tokens = encoding.encode(input_text)
    return len(input_tokens)


def calculate_output_tokens(text, encoding):
    """
    Calculate output tokens from response text.
    
    Args:
        text: Response text string
        encoding: tiktoken encoding object
        
    Returns:
        int: Number of output tokens
    """
    output_tokens = encoding.encode(text)
    return len(output_tokens)


def openai_api_token_tracking_example():
    """
    OpenAI API Token Tracking Example
    
    This function demonstrates how to:
    1. Set up OpenAI API client with environment variables
    2. Create a conversation with multiple messages
    3. Calculate input tokens before making the API call
    4. Make a chat completion request
    5. Calculate output tokens from the response
    
    Note: The token counting here is approximate. For exact token counts,
    you can use the 'usage' field from the API response.
    """
    # Load environment variables
    load_dotenv(override=True)
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("No API key was found!")
        return
    else:
        print("API key found!")
    
    # Initialize OpenAI client
    openai = OpenAI()
    
    # Create conversation messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hi! I'm Mkay!"},
        {"role": "assistant", "content": "Hi Mkay! How can I assist you today?"},
        {"role": "user", "content": "What's my name?"}
    ]
    
    # Get encoding for token counting
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")  # Note: notebook uses "gpt-4.1-mini"
    
    # Calculate input tokens
    input_tokens = calculate_input_tokens(messages, encoding)
    print("Input tokens:", input_tokens)
    
    # Make API call
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Note: notebook uses "gpt-4.1-mini"
        messages=messages
    )
    
    # Get response content
    response_content = response.choices[0].message.content
    print(f"Response: {response_content}")
    
    # Calculate output tokens
    output_tokens = calculate_output_tokens(response_content, encoding)
    print("Output tokens:", output_tokens)
    
    # Note: For exact token counts, use response.usage
    if hasattr(response, 'usage') and response.usage:
        print(f"\nExact token usage from API:")
        print(f"  Prompt tokens: {response.usage.prompt_tokens}")
        print(f"  Completion tokens: {response.usage.completion_tokens}")
        print(f"  Total tokens: {response.usage.total_tokens}")
    
    return response_content


if __name__ == "__main__":
    """
    Main execution block.
    Run examples when script is executed directly.
    """
    print("=" * 60)
    print("Basic Tokenization Example")
    print("=" * 60)
    basic_tokenization_example()
    
    print("\n" + "=" * 60)
    print("OpenAI API Token Tracking Example")
    print("=" * 60)
    openai_api_token_tracking_example()

