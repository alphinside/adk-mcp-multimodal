from fastmcp import FastMCP
from typing import Annotated, Optional
from pydantic import Field
import base64
import io
import time
import asyncio
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

mcp = FastMCP("Veo MCP Server")

@mcp.tool
async def generate_video_with_image(
    prompt: Annotated[
        str, Field(description="Text description of the video to generate")
    ],
    image_data: Annotated[
        str, Field(description="Base64-encoded image data to use as starting frame")
    ],
    model: Annotated[
        str, Field(description="Veo model to use")
    ] = "veo-3.1-fast-generate-preview",
    negative_prompt: Annotated[
        str | None,
        Field(description="Things to avoid in the generated video"),
    ] = None,
    duration_seconds: Annotated[
        int, Field(description="Video duration in seconds (4, 6, or 8)")
    ] = 4,
) -> dict:
    """Generates a video from text prompt and a starting image using Google's Veo API.

    This function uses an image as the first frame of the generated video.

    IMPORTANT: The prompt must be as detailed as possible for best results. Include:
    - Camera movements (e.g., "panning", "tracking shot", "close-up")
    - Lighting and mood (e.g., "cinematic", "soft lighting", "golden hour")
    - Subject appearance and actions (e.g., detailed descriptions of what's happening)
    - Environment and atmosphere details

    Args:
        prompt: VERY DETAILED text description of the video to generate. Be specific about
                camera angles, movements, lighting, subject actions, and environment.
                More detail = better results.
        image_data: Base64-encoded image data to use as the starting frame
        model: Veo model name (default: veo-3.1-fast-generate-preview)
        negative_prompt: Optional prompt describing what to avoid in the video
        duration_seconds: Video length (4, 6, or 8 seconds)

    Returns:
        dict: A dictionary containing:
            - status: 'success' or 'error'
            - message: Description of the result
            - video_data: Base64-encoded video data (on success only)
            - video_path: Path to saved video file for debugging (on success only)
    """
    try:
        # Initialize the Gemini client
        client = genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION"),
        )


        # Decode the image
        image_bytes = base64.b64decode(image_data)
        print(f"Successfully decoded image data: {len(image_bytes)} bytes")
        
        # Create image object
        image = types.Image(image_bytes=image_bytes, mime_type="image/png")
        
        # Prepare the config
        config = types.GenerateVideosConfig(
            duration_seconds=duration_seconds,
            number_of_videos=1,
        )
        
        if negative_prompt:
            config.negative_prompt = negative_prompt
        
        print(f"Starting video generation with image and prompt: {prompt[:100]}...")
        
        # Generate the video (async operation)
        operation = client.models.generate_videos(
            model=model,
            prompt=prompt,
            image=image,
            config=config,
        )
        
        # Poll until the operation is complete
        poll_count = 0
        while not operation.done:
            poll_count += 1
            print(f"Waiting for video generation to complete... (poll {poll_count})")
            await asyncio.sleep(5)
            operation = client.operations.get(operation)
        
        # Download the video and convert to base64
        video = operation.response.generated_videos[0]
        
        # Save video to file for debugging
        output_path = f"generated_video_{int(time.time())}.mp4"
        video.video.save(output_path)
        print(f"Video saved to {output_path} for debugging")
        
        # Get video bytes and encode to base64
        video_bytes = video.video.video_bytes
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        
        print(f"Video generated successfully: {len(video_bytes)} bytes")
        
        return {
            "status": "success",
            "message": f"Video with image generated successfully after {poll_count * 5} seconds",
            "video_data": video_base64,
            "video_path": output_path,
        }
    except Exception as e:
        breakpoint()
        logging.error(e)
        return {"status": "error", "message": f"Error generating video with image: {str(e)}"}

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)