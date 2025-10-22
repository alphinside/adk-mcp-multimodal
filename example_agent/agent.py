from google.adk.agents.llm_agent import Agent
from product_photo_editor.custom_tools import edit_product_asset
from product_photo_editor.mcp_tools import mcp_toolset
from product_photo_editor.model_callbacks import before_model_modifier
from product_photo_editor.tool_callbacks import (
    before_tool_modifier,
    after_tool_modifier,
)
from product_photo_editor.prompt import AGENT_INSTRUCTION

root_agent = Agent(
    model="gemini-2.5-flash",
    name="product_photo_editor",
    description="""A friendly product photo editor assistant that helps small business 
owners edit and enhance their product photos. Perfect for improving photos of handmade 
goods, food products, crafts, and small retail items""",
    instruction=AGENT_INSTRUCTION
    + """
**IMPORTANT: Base64 Argument Rule on Tool Call**

If you found any tool call arguments that requires base64 data,
ALWAYS provide the artifact_id of the referenced file to 
the tool call. NEVER ask user to provide base64 data. 
Base64 data encoding process is out of your 
responsibility and will be handled in another part of the system.
""",
    tools=[
        edit_product_asset,
        mcp_toolset,
    ],
    before_model_callback=before_model_modifier,
    before_tool_callback=before_tool_modifier,
    after_tool_callback=after_tool_modifier,
)
