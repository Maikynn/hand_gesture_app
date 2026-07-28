#!/usr/bin/env python3
"""
OpenRouter client for the Hand Gesture Application.
Handles communication with OpenRouter API for AI responses.
"""

import requests
from typing import Optional

class OpenRouterClient:
    """
    Client for OpenRouter API.
    """
    def __init__(self, api_key: str = "", model: str = "free"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        
    def set_api_key(self, api_key: str):
        """Set the API key."""
        self.api_key = api_key
        
    def set_model(self, model: str):
        """Set the model to use."""
        self.model = model
        
    def send_message(self, message: str, system_prompt: str = "") -> Optional[str]:
        """
        Send a message to OpenRouter and get response.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            
        Returns:
            Response text or None on error
        """
        if not self.api_key:
            return "Error: No API key configured"
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Network error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
            
    def get_available_models(self) -> list:
        """
        Get list of available models from OpenRouter.
        """
        if not self.api_key:
            return []
            
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return [model["id"] for model in data.get("data", [])]
            else:
                return []
                
        except Exception:
            return []


if __name__ == "__main__":
    # Example usage
    client = OpenRouterClient(api_key="your-api-key-here")
    response = client.send_message("Hello, how are you?")
    print(response)