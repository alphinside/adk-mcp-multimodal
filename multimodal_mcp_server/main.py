from fastmcp import FastMCP
from typing import Annotated
from pydantic import Field
import base64
from PIL import Image
import io

mcp = FastMCP("My MCP Server")


@mcp.tool
def flip_image(
    file_data: Annotated[str, Field(description="Base64-encoded image data")],
    direction: Annotated[
        str, Field(description="Flip direction: 'horizontal' or 'vertical'")
    ] = "horizontal",
) -> dict:
    """Flips an image horizontally or vertically and returns the flipped image as base64.

    This function takes a base64-encoded image, decodes it, flips it either horizontally
    or vertically using PIL's transpose method, and then re-encodes it to base64 before returning.

    Args:
        file_data: Base64-encoded string representing image data
        direction: Direction to flip the image ('horizontal' or 'vertical'), defaults to 'horizontal'

    Returns:
        dict: A dictionary containing:
            - status: 'success' or 'error'
            - message: Description of the result
            - flipped_image: Base64-encoded string of the flipped image (on success only)

    Raises:
        Exception: If there's an error in decoding, flipping, or encoding the image
    """
    try:
        image_bytes = base64.b64decode(file_data)
        print(f"Successfully decoded image data: {len(image_bytes)} bytes")

        # Flip the image based on direction
        image = Image.open(io.BytesIO(image_bytes))

        if direction.lower() == "vertical":
            flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
            flip_direction = "vertically"
        else:  # default to horizontal
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flip_direction = "horizontally"

        # Convert flipped image back to bytes
        buffer = io.BytesIO()
        flipped_image.save(buffer, format=image.format or "PNG")
        flipped_bytes = buffer.getvalue()

        # Convert bytes to base64 string
        flipped_base64 = base64.b64encode(flipped_bytes).decode("utf-8")

        return {
            "status": "success",
            "message": f"Image successfully flipped {flip_direction}: {len(image_bytes)} bytes",
            "flipped_image": flipped_base64,
        }
    except Exception as e:
        return {"status": "error", "message": f"Error flipping image image: {str(e)}"}


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
