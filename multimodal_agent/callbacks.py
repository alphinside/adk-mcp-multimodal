from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai.types import Part
from typing import Optional


async def before_agent_callback(
    callback_context: CallbackContext,
) -> Optional[LlmResponse]:
    """Saves uploaded files as artifacts and provides artifact information to the agent.

    This function is called before the agent processes the user's request. It
    intercepts any uploaded files, saves them as artifacts, and then appends the
    artifact information to the user's message. This ensures that the agent is
    aware of the uploaded files and can process them accordingly.

    Args:
        callback_context: The callback context containing the user's request.

    Returns:
        An optional LLM response.
    """
    # Modify uploaded user file as artifact and set data string identifier with
    # artifact information.
    modified_parts = []
    for part in callback_context.user_content.parts:
        modified_parts.append(part)
        if part.inline_data is not None:
            artifact_filename = part.inline_data.display_name
            await callback_context.save_artifact(
                filename=artifact_filename, artifact=part
            )
            modified_parts.append(
                Part(text=str({"artifact_filename": artifact_filename}))
            )

    callback_context.user_content.parts = modified_parts
