from google.adk.tools import ToolContext
from google.genai.types import Part
import numpy as np
import cv2
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StreamableHTTPConnectionParams,
)

mcp_tools = MCPToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="http://0.0.0.0:8000/mcp",
        timeout=30,
    )
)


async def convert_image_to_grayscale(
    tool_context: ToolContext, artifact_filename: str
) -> dict:
    """Converts an image artifact to black and white (grayscale).

    This tool loads an image artifact from the specified artifact_filename,
    converts it to grayscale using OpenCV, and returns the processed result.
    Useful for image preprocessing or creating monochrome versions of uploaded
    images.

    Args:
        artifact_filename: The name of the image artifact file to convert.

    Returns:
        A dictionary containing the conversion result and metadata.
        Typically includes status information and processed image
        details.
    """
    try:
        artifact = await tool_context.load_artifact(filename=artifact_filename)
        file_data = artifact.inline_data.data

        # Convert bytes to numpy array for cv2.imdecode
        nparr = np.frombuffer(file_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {
                "status": "error",
                "message": "Invalid image format or corrupted file data",
            }

        # Convert image to grayscale (black and white)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Encode the grayscale image back to bytes
        success, encoded_image = cv2.imencode(".png", gray_image)

        if not success:
            return {"status": "error", "message": "Failed to encode processed image"}

        # Save the processed image as a new artifact
        processed_filename = f"bw_{artifact_filename}"
        processed_image_part = Part(
            inline_data={"mime_type": "image/png", "data": encoded_image.tobytes()}
        )

        await tool_context.save_artifact(
            filename=processed_filename, artifact=processed_image_part
        )

        return {
            "status": "success",
            "message": "Image successfully converted to black and white",
            "original_filename": artifact_filename,
            "processed_filename": processed_filename,
        }

    except Exception as e:
        return {"status": "error", "message": f"Error processing image: {str(e)}"}
