from google.adk.agents.llm_agent import Agent

from multimodal_agent.callbacks import (
    before_agent_callback,
    before_tool_callback,
    after_tool_callback,
)
from multimodal_agent.tools import convert_image_to_grayscale, mcp_tools


root_agent = Agent(
    name="MultimodalAgent",
    model="gemini-2.5-flash",
    instruction="""Assist user with their multimodal requests
If tools that have input argument with base-64 encoded data type,
ALWAYS provide the artifact filename string value to that input argument field.

E.g.:
    "flip_image" tool has "file_data" input arguments with base64 data type.
    Then you must provide {"file_data":"<artifact_filename>"} when you want to execute
    this tool
    
DO NOT make assumption or modification on the artifact filename. 
ONLY use the artifact filename provided to you.   
""",
    description="""You are expert at handling multimodal requests like file or images.
When receiving file from the user, each uploaded file will be saved as artifact and 
you will be provided with the artifact filename information directly after the file.
E.g.

[Uploaded File 1]
{"artifact_filename": "uploaded-file-1-filename"}
[Uploaded File 2]
{"artifact_filename": "uploaded-file-2-filename"}
""",
    before_agent_callback=before_agent_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    tools=[mcp_tools, convert_image_to_grayscale],
)
