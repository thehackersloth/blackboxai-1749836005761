import os
import pytest
from unittest.mock import patch, MagicMock
from autoagent.connectors import OllamaConnector, PerplexityConnector, AnthropicConnector

# Test messages that will be used across tests
TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
]

def test_ollama_connector_initialization():
    """Test Ollama connector initialization with default and custom settings."""
    # Test with default settings
    connector = OllamaConnector()
    assert connector.endpoint == "http://localhost:11434"

    # Test with custom settings
    custom_connector = OllamaConnector(endpoint="http://custom:11434")
    assert custom_connector.endpoint == "http://custom:11434"

@patch('requests.get')
def test_ollama_list_models(mock_get):
    """Test Ollama model listing functionality."""
    # Mock successful response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "models": [
            {"name": "phuzzy/darknemo:latest"},
            {"name": "codellama:7b"},
            {"name": "llama2"}
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    connector = OllamaConnector()
    models = connector.list_models()

    assert len(models) == 3
    assert "phuzzy/darknemo:latest" in models
    assert "codellama:7b" in models
    assert "llama2" in models

    # Test error handling
    mock_get.side_effect = requests.exceptions.RequestException("Connection error")
    models = connector.list_models()
    assert models == []

def test_perplexity_connector_initialization():
    """Test Perplexity connector initialization."""
    # Test with default settings
    connector = PerplexityConnector()
    assert connector.endpoint == "https://api.perplexity.ai"
    assert connector.api_key is None

    # Test with custom settings
    custom_connector = PerplexityConnector(
        api_key="test_key",
        endpoint="https://custom.perplexity.ai"
    )
    assert custom_connector.endpoint == "https://custom.perplexity.ai"
    assert custom_connector.api_key == "test_key"

def test_anthropic_connector_initialization():
    """Test Anthropic connector initialization."""
    # Test with default settings
    connector = AnthropicConnector()
    assert connector.endpoint == "https://api.anthropic.com"
    assert connector.api_key is None

    # Test with custom settings
    custom_connector = AnthropicConnector(
        api_key="test_key",
        endpoint="https://custom.anthropic.com"
    )
    assert custom_connector.endpoint == "https://custom.anthropic.com"
    assert custom_connector.api_key == "test_key"

@patch('requests.post')
def test_ollama_generate(mock_post):
    """Test Ollama generate method with mocked response."""
    # Mock successful response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "response": "Hello! I'm doing well. How can I help you today?"
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    connector = OllamaConnector()
    response = connector.generate(TEST_MESSAGES, model="phuzzy/darknemo:latest")

    assert "choices" in response
    assert response["choices"][0]["message"]["content"] == "Hello! I'm doing well. How can I help you today?"

    # Test with different model
    response = connector.generate(TEST_MESSAGES, model="codellama:7b")
    assert "choices" in response

@patch('requests.post')
def test_perplexity_generate(mock_post):
    """Test Perplexity generate method with mocked response."""
    # Mock successful response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Hello! I'm doing well. How can I help you today?"
            }
        }]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    connector = PerplexityConnector(api_key="test_key")
    response = connector.generate(TEST_MESSAGES, model="pplx-7b-chat")

    assert "choices" in response
    assert response["choices"][0]["message"]["content"] == "Hello! I'm doing well. How can I help you today?"

@patch('requests.post')
def test_anthropic_generate(mock_post):
    """Test Anthropic generate method with mocked response."""
    # Mock successful response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "content": [{"text": "Hello! I'm doing well. How can I help you today?"}]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    connector = AnthropicConnector(api_key="test_key")
    response = connector.generate(TEST_MESSAGES, model="claude-3-opus-20240229")

    assert "choices" in response
    assert response["choices"][0]["message"]["content"] == "Hello! I'm doing well. How can I help you today?"

def test_error_handling():
    """Test error handling in connectors."""
    # Test Ollama without API key
    connector = OllamaConnector()
    response = connector.generate(TEST_MESSAGES, model="llama2")
    assert "error" not in response  # Ollama can work without API key

    # Test Perplexity without API key
    connector = PerplexityConnector()
    response = connector.generate(TEST_MESSAGES, model="pplx-7b-chat")
    assert "error" in response
    assert "Perplexity API key not found" in response["error"]

    # Test Anthropic without API key
    connector = AnthropicConnector()
    response = connector.generate(TEST_MESSAGES, model="claude-3-opus-20240229")
    assert "error" in response
    assert "Anthropic API key not found" in response["error"]

if __name__ == "__main__":
    pytest.main([__file__])
