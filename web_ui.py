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
from gradio.data_classes import FileData
import shutil

APP_NAME = "photo_editor_app"
USER_ID = "default_user"
SESSION_ID = "default_session"
SESSION_SERVICE = InMemorySessionService()
ARTIFACT_SERVICE = InMemoryArtifactService()
gr.Progress()

# Create fresh artifacts directory (delete if exists)
GRADIO_ARTIFACT_DIR = Path("gradio_artifacts")
if GRADIO_ARTIFACT_DIR.exists():
    shutil.rmtree(GRADIO_ARTIFACT_DIR)
    print(f"🗑️  Deleted existing artifacts directory: {GRADIO_ARTIFACT_DIR}")
GRADIO_ARTIFACT_DIR.mkdir(exist_ok=True)
print(f"📁 Created fresh artifacts directory: {GRADIO_ARTIFACT_DIR}")

PHOTO_EDITOR_AGENT_RUNNER = Runner(
    agent=photo_editor_agent,  # The agent we want to run
    app_name=APP_NAME,  # Associates runs with our app
    session_service=SESSION_SERVICE,  # Uses our session manager
    artifact_service=ARTIFACT_SERVICE,  # Uses our artifact manager
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

    # Build message parts from text and files
    parts = build_message_parts(message)

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
                    yield responses
                elif part.function_response:
                    formatted_response = f"```python\n{pformat(part.function_response.model_dump(), indent=2, width=80)}\n```"

                    # Capture artifact delta from function response
                    artifact_delta = event.actions.artifact_delta

                    for artifact_name in artifact_delta.keys():
                        artifact = await ARTIFACT_SERVICE.load_artifact(
                            app_name=APP_NAME,
                            user_id=USER_ID,
                            session_id=SESSION_ID,
                            filename=artifact_name,
                        )

                        artifact_bytes = artifact.inline_data.data
                        mime_type = artifact.inline_data.mime_type

                        # Write artifact to Gradio directory
                        artifact_file_path = write_artifact_to_gradio_dir(
                            artifact_bytes, mime_type, artifact_name
                        )

                        responses.append(
                            gr.ChatMessage(
                                role="assistant",
                                content=FileData(
                                    path=artifact_file_path, mime_type=mime_type
                                ),
                                metadata={"title": "📥 Generated Artifact"},
                            )
                        )

                        yield responses

                    responses.append(
                        gr.ChatMessage(
                            role="assistant",
                            content=formatted_response,
                            metadata={"title": "⚡ Tool Response"},
                        )
                    )

                    yield responses
                else:
                    responses.append(
                        gr.ChatMessage(
                            role="assistant",
                            content=part.text,
                        )
                    )

                    yield responses


async def initialize_session_if_not_exists():
    """Ensure session exists before agent runs."""
    if (
        await SESSION_SERVICE.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
        )
        is None
    ):
        await SESSION_SERVICE.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
        )


def load_image_as_part(file_path: str) -> types.Part:
    """Load an image file and convert it to a Part with inline data.

    Args:
        file_path: Path to the image file.

    Returns:
        A Part object with the image data.
    """
    mime_type, _ = mimetypes.guess_type(file_path)

    with open(file_path, "rb") as f:
        image_bytes = f.read()

    return types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_bytes))


def build_message_parts(message: Dict[str, Any]) -> List[types.Part]:
    """Build a list of Parts from a Gradio message dictionary.

    Args:
        message: Dictionary containing 'text' and 'files' keys.

    Returns:
        List of Part objects for the message content.
    """
    parts = []

    # Add text part if present
    text_content = message.get("text", "")
    if text_content:
        parts.append(types.Part(text=text_content))

    # Add image parts from uploaded files
    files = message.get("files", [])
    for file_path in files:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type.startswith("image/"):
            parts.append(load_image_as_part(file_path))

    return parts


def write_artifact_to_gradio_dir(
    artifact_bytes: bytes, mime_type: str, artifact_name: str
) -> str:
    """Write artifact bytes to the Gradio artifacts directory.

    Args:
        artifact_bytes: Binary data of the artifact.
        mime_type: MIME type of the artifact.
        artifact_name: Name of the artifact file.

    Returns:
        Path to the created file.
    """
    # Ensure artifact has correct extension based on mime_type
    artifact_path = Path(artifact_name)
    expected_ext = mimetypes.guess_extension(mime_type) or ""

    # Add extension if missing
    if not artifact_path.suffix and expected_ext:
        artifact_name = f"{artifact_name}{expected_ext}"

    # Write to Gradio artifacts directory
    file_path = GRADIO_ARTIFACT_DIR / artifact_name
    with open(file_path, "wb") as f:
        f.write(artifact_bytes)

    return str(file_path)


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
