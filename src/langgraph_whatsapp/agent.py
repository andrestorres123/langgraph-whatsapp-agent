import logging
from langgraph_sdk import get_client
from langgraph_whatsapp import config
import json
import uuid

LOGGER = logging.getLogger(__name__)


class Agent:
    def __init__(self):
        self.client = get_client(url=config.LANGGRAPH_URL)
        try:
            self.graph_config = (
                json.loads(config.CONFIG) if isinstance(config.CONFIG, str) else config.CONFIG
            )
        except json.JSONDecodeError as e:
            LOGGER.error(f"Failed to parse CONFIG as JSON: {e}")
            raise

    async def invoke(self, id: str, user_message: str, images: list = None) -> dict:
        """
        Process a user message through the LangGraph client.
        
        Args:
            id: The unique identifier for the conversation
            user_message: The message content from the user
            images: List of dictionaries with image data
            
        Returns:
            dict: The result from the LangGraph run
        """
        LOGGER.info(f"Invoking agent with thread_id: {id}")

        try:
            # Build message content - always use a list for consistent format
            message_content = []
            if user_message:
                message_content.append({
                    "type": "text",
                    "text": user_message
                })

            if images:
                for img in images:
                    if isinstance(img, dict) and "image_url" in img:
                        message_content.append({
                            "type": "image_url",
                            "image_url": img["image_url"]
                        })
            
            request_payload = {
                "thread_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, id)),
                "assistant_id": config.ASSISTANT_ID,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": message_content
                        }
                    ]
                },
                "config": self.graph_config,
                "metadata": {"event": "api_call"},
                "multitask_strategy": "interrupt",
                "if_not_exists": "create",
                "stream_mode": "values",
            }
            
            final_response = None
            all_chunks = []
            async for chunk in self.client.runs.stream(**request_payload):
                all_chunks.append(chunk)
                final_response = chunk
            
            # Debug logging to understand the response structure
            LOGGER.info(f"Total chunks received: {len(all_chunks)}")
            LOGGER.info(f"Final response type: {type(final_response)}")
            LOGGER.info(f"Final response data keys: {list(final_response.data.keys()) if hasattr(final_response, 'data') and final_response.data else 'No data or data is None'}")
            LOGGER.info(f"Final response data content: {final_response.data}")
            
            # Handle different possible response structures
            if hasattr(final_response, 'data') and final_response.data:
                if "messages" in final_response.data:
                    # Get the last message from the assistant
                    messages = final_response.data["messages"]
                    for message in reversed(messages):
                        if message.get("role") == "assistant" or (hasattr(message, "type") and message.type == "ai"):
                            return message.get("content", str(message))
                    # If no assistant message found, return the last message
                    return messages[-1].get("content", str(messages[-1])) if messages else "No messages found"
                else:
                    # If there's no messages key, try to extract content from other possible keys
                    LOGGER.warning(f"No 'messages' key found in response data. Available keys: {list(final_response.data.keys())}")
                    # Try to return the entire data structure for debugging
                    return str(final_response.data)
            else:
                LOGGER.error("Final response has no data attribute or data is None")
                return "No response data received"
                
        except Exception as e:
            LOGGER.error(f"Error during invoke: {str(e)}", exc_info=True)
            raise
    