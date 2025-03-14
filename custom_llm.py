import requests
import json
from typing import Any, Dict, List, Optional, Mapping, Iterator
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

class AnythingLLMProvider(BaseChatModel):
    """Custom Chat Model provider for AnythingLLM API endpoint."""
    
    api_url: str
    api_key: str
    model: str = "demo"
    temperature: float = 0.7
    system_prompt: str = "You are an expert SQL agent that helps users analyze their database by translating their natural language questions into SQL queries. Always include the SQL query you generated to answer the question."
    
    @property
    def _llm_type(self) -> str:
        return "custom_anythingllm_chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completions using the AnythingLLM API endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Convert LangChain messages to format expected by AnythingLLM API
        api_messages = []
        system_message = None
        last_user_message = None
        assistant_messages = []
        
        # First pass: categorize messages
        for message in messages:
            if isinstance(message, SystemMessage):
                # Keep only the last system message
                system_message = {"role": "system", "content": message.content}
            elif isinstance(message, HumanMessage):
                # Keep track of the last user message
                last_user_message = {"role": "user", "content": message.content}
            elif isinstance(message, AIMessage):
                assistant_messages.append({"role": "assistant", "content": message.content})
            else:
                # Treat unknown message types as user messages
                last_user_message = {"role": "user", "content": str(message.content)}
        
        # Build the final message array
        # Always include a system message (either from messages or default)
        api_messages.append(system_message or {"role": "system", "content": self.system_prompt})
        
        # Add assistant messages
        api_messages.extend(assistant_messages)
        
        # Always add the user message last
        if last_user_message:
            api_messages.append(last_user_message)
        else:
            # If no user message found, create a default one
            api_messages.append({"role": "user", "content": "Please help me with my database query."})
        
        data = {
            "messages": api_messages,
            "model": self.model,
            "stream": False,
            "temperature": self.temperature
        }
        
        try:
            print(f"Sending request to AnythingLLM API: {json.dumps(data, indent=2)}")  # Debug print
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            print(f"Received response from AnythingLLM API: {json.dumps(result, indent=2)}")  # Debug print
            
            # Extract the content from the response
            # The format should match OpenAI's API response structure
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                # Create a valid ChatGeneration
                chat_generation = ChatGeneration(
                    message=AIMessage(content=content),
                    generation_info={"finish_reason": result["choices"][0].get("finish_reason", "stop")}
                )
                # Return a valid ChatResult
                return ChatResult(generations=[chat_generation])
            else:
                raise ValueError(f"Unexpected response format: {result}")
            
        except Exception as e:
            print(f"Error calling AnythingLLM API: {str(e)}")  # Debug print
            if isinstance(e, requests.exceptions.HTTPError):
                print(f"Response content: {e.response.text}")  # Debug print
            # Even in error case, return valid ChatResult with error message
            error_message = f"Error: {str(e)}"
            chat_generation = ChatGeneration(
                message=AIMessage(content=error_message),
                generation_info={"finish_reason": "error"}
            )
            return ChatResult(generations=[chat_generation])
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate asynchronous chat completions."""
        # For simplicity, we'll use the synchronous version in async context
        # In a production environment, you'd want to use aiohttp or similar
        return self._generate(messages, stop, run_manager, **kwargs)
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "api_url": self.api_url
        } 