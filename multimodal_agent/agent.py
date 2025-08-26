from google.adk.agents.llm_agent import Agent

from multimodal_agent.callbacks import before_agent_callback
from multimodal_agent.tools import convert_image_to_grayscale


root_agent = Agent(
    name="MultimodalAgent",
    model="gemini-2.5-flash-lite",
    instruction="Assist user with their multimodal requests",
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
    tools=[convert_image_to_grayscale],
)
