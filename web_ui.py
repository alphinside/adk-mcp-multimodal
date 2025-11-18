"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import gradio as gr
import mimetypes
from pathlib import Path
from typing import List, Dict, Any
from product_photo_editor.agent import root_agent as photo_editor_agent
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.events import Event
from typing import AsyncIterator
from google.genai import types
from pprint import pformat

APP_NAME = "photo_editor_app"   
USER_ID = "default_user"
SESSION_ID = "default_session"
SESSION_SERVICE = InMemorySessionService()
ARTIFACT_SERVICE = InMemoryArtifactService()
PHOTO_EDITOR_AGENT_RUNNER = Runner(
    agent=photo_editor_agent,  # The agent we want to run
    app_name=APP_NAME,  # Associates runs with our app
    session_service=SESSION_SERVICE,  # Uses our session manager
    artifact_service= ARTIFACT_SERVICE,  # Uses our artifact manager
)

async def initialize_session_if_not_exists():
    """Ensure session exists before agent runs."""
    if await SESSION_SERVICE.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    ) is None:
        await SESSION_SERVICE.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
        )



async def get_response_from_agent(
    message: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> str:
    """Send the message to the backend and get a response.

    Args:
        message: Text an uploaded file content of the message.
        history: List of previous message dictionaries in the conversation.

    Returns:
        Text response from the backend service.
    """
    await initialize_session_if_not_exists()
    
    try:
        # Build parts list from message
        parts = []
        text_content = message.get("text", "")
        files = message.get("files", [])
        
        if text_content:
            parts.append(types.Part(text=text_content))
        
        for file_path in files:
            mime_type, _ = mimetypes.guess_type(file_path)
            
            if mime_type and mime_type.startswith("image/"):
                with open(file_path, "rb") as f:
                    image_bytes = f.read()
                
                parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type,
                            data=image_bytes
                        )
                    )
                )
        
        events_iterator: AsyncIterator[Event] = PHOTO_EDITOR_AGENT_RUNNER.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=types.Content(role="user", parts=parts),
        )

        responses = []
        async for event in events_iterator:  # event has type Event
            if event.content.parts:
                for part in event.content.parts:
                    if part.function_call:
                        formatted_call = f"```python\n{pformat(part.function_call.model_dump(), indent=2, width=80)}\n```"
                        responses.append(
                            gr.ChatMessage(
                                role="assistant",
                                content=f"{part.function_call.name}:\n{formatted_call}",
                                metadata={"title": "🛠️ Tool Call"},
                            )
                        )
                    elif part.function_response:
                        formatted_response = f"```python\n{pformat(part.function_response.model_dump(), indent=2, width=80)}\n```"

                        responses.append(
                            gr.ChatMessage(
                                role="assistant",
                                content=formatted_response,
                                metadata={"title": "⚡ Tool Response"},
                            )
                        )

            # Key Concept: is_final_response() marks the concluding message for the turn
            if event.is_final_response():
                if event.content and event.content.parts:
                    # Extract text from the first part
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    # Handle potential errors/escalations
                    final_response_text = (
                        f"Agent escalated: {event.error_message or 'No specific message.'}"
                    )
                responses.append(
                    gr.ChatMessage(role="assistant", content=final_response_text)
                )
                yield responses
                break  # Stop processing events once the final response is found

            yield responses
    except Exception as e:
        yield [
            gr.ChatMessage(
                role="assistant",
                content=f"Error communicating with agent: {str(e)}",
            )
        ]


if __name__ == "__main__":
    demo = gr.ChatInterface(
        get_response_from_agent,
        title="Photo Editor",
        description="This assistant can help you to edit and enhance your product photos.",
        type="messages",
        multimodal=True,
        textbox=gr.MultimodalTextbox(file_count="multiple", file_types=["image"]),
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
    )