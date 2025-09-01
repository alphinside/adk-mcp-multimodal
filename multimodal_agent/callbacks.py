from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai.types import Part
from typing import Optional, Any
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool
import base64
import logging
import json
from mcp.types import CallToolResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

async def before_tool_callback(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
):
    # Identify which tool input should be modified
    if tool.name == "flip_image":
        logger.info("Modify tool args for artifact: %s", args["file_data"])
        # Get the artifact filename from the tool input argument
        artifact_filename = args["file_data"]
        artifact = await tool_context.load_artifact(filename=artifact_filename)
        file_data = artifact.inline_data.data

        # Convert byte data to base64 string
        base64_data = base64.b64encode(file_data).decode('utf-8')

        # Then modify the tool input argument
        args["file_data"] = base64_data

async def after_tool_callback(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: dict | CallToolResult
):

    if tool.name == "flip_image":
        tool_result = json.loads(tool_response.content[0].text)

        # Get the expected response field which contains the flipped image base64 string
        flipped_image_base64 = tool_result["flipped_image"]
        artifact_filename = f"flipped_image_{tool_context.function_call_id}.png"
        
        # Convert base64 string to byte data
        flipped_image_bytes = base64.b64decode(flipped_image_base64)
        
        # Save the flipped image as artifact
        await tool_context.save_artifact(
            filename=artifact_filename, artifact=Part(inline_data={"mime_type": "image/png", "data": flipped_image_bytes})
        )

        # Then modify the tool response to include the artifact filename and remove the base64 string
        tool_result["flipped_image"] = artifact_filename
        logger.info("Modify tool response for artifact: %s", tool_result["flipped_image"])
        
        return tool_result