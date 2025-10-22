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

from google.genai.types import Part
from typing import Any
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool.mcp_tool import McpTool
import base64
import logging
import json
from mcp.types import CallToolResult


async def before_tool_modifier(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
):
    # Identify which tool input should be modified
    if isinstance(tool, McpTool) and tool.name == "generate_video_with_image":
        logging.info("Modify tool args for artifact: %s", args["image_data"])
        # Get the artifact filename from the tool input argument
        artifact_filename = args["image_data"]
        artifact = await tool_context.load_artifact(filename=artifact_filename)
        file_data = artifact.inline_data.data

        # Convert byte data to base64 string
        base64_data = base64.b64encode(file_data).decode("utf-8")

        # Then modify the tool input argument
        args["image_data"] = base64_data


async def after_tool_modifier(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict | CallToolResult,
):
    if isinstance(tool, McpTool) and tool.name == "generate_video_with_image":
        tool_result = json.loads(tool_response.content[0].text)

        # Get the expected response field which contains the video data
        video_data = tool_result["video_data"]
        artifact_filename = f"video_{tool_context.function_call_id}.mp4"

        # Convert base64 string to byte data
        video_bytes = base64.b64decode(video_data)

        # Save the video as artifact
        await tool_context.save_artifact(
            filename=artifact_filename,
            artifact=Part(inline_data={"mime_type": "video/mp4", "data": video_bytes}),
        )

        # Remove the video data from the tool response
        tool_result.pop("video_data")

        # Then modify the tool response to include the artifact filename and remove the base64 string
        tool_result["video_artifact_id"] = artifact_filename
        logging.info(
            "Modify tool response for artifact: %s", tool_result["video_artifact_id"]
        )

        return tool_result
