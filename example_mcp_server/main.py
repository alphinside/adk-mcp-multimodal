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

from fastmcp import FastMCP
from typing import Annotated
from pydantic import Field
import base64
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
    negative_prompt: Annotated[
        str | None,
        Field(description="Things to avoid in the generated video"),
    ] = None,
) -> dict:
    """Generates a professional product marketing video from text prompt and starting image using Google's Veo API.

    This function uses an image as the first frame of the generated video and automatically
    enriches your prompt with professional video production quality guidelines to create
    high-quality marketing assets suitable for commercial use.

    AUTOMATIC ENHANCEMENTS APPLIED:
    - 4K cinematic quality with professional color grading
    - Smooth, stabilized camera movements
    - Professional studio lighting setup
    - Shallow depth of field for product focus
    - Commercial-grade production quality
    - Marketing-focused visual style

    PROMPT WRITING TIPS:
    Describe what you want to see in the video. Focus on:
    - Product actions/movements (e.g., "rotating slowly", "zooming into details")
    - Desired camera angles (e.g., "close-up of the product", "wide shot")
    - Background/environment (e.g., "minimalist white backdrop", "lifestyle setting")
    - Any specific details about the product presentation

    The system will automatically enhance your prompt with professional production quality.

    Args:
        prompt: Description of the video to generate. Focus on the core product presentation
                you want. The system will automatically add professional quality enhancements.
        image_data: Base64-encoded image data to use as the starting frame
        negative_prompt: Optional prompt describing what to avoid in the video

    Returns:
        dict: A dictionary containing:
            - status: 'success' or 'error'
            - message: Description of the result
            - video_data: Base64-encoded video data (on success only)
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
            duration_seconds=8,
            number_of_videos=1,
        )

        if negative_prompt:
            config.negative_prompt = negative_prompt

        # Enrich the prompt for professional marketing quality
        enriched_prompt = enrich_prompt_for_marketing(prompt)

        # Generate the video (async operation)
        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=enriched_prompt,
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

        # Get video bytes and encode to base64
        video_bytes = video.video.video_bytes
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")

        print(f"Video generated successfully: {len(video_bytes)} bytes")

        return {
            "status": "success",
            "message": f"Video with image generated successfully after {poll_count * 5} seconds",
            "complete_prompt": enriched_prompt,
            "video_data": video_base64,
        }
    except Exception as e:
        logging.error(e)
        return {
            "status": "error",
            "message": f"Error generating video with image: {str(e)}",
        }


def enrich_prompt_for_marketing(user_prompt: str) -> str:
    """Enriches user prompt with professional video production quality enhancements.

    Adds cinematic quality, professional lighting, smooth camera work, and marketing-focused
    elements to ensure high-quality product marketing videos.
    """
    enhancement_prefix = """Create a high-quality, professional product marketing video with the following characteristics:

TECHNICAL SPECIFICATIONS:
- 4K cinematic quality with professional color grading
- Smooth, stabilized camera movements
- Professional studio lighting setup with soft, even illumination
- Shallow depth of field for product focus
- High dynamic range (HDR) for vibrant colors

VISUAL STYLE:
- Clean, minimalist aesthetic suitable for premium brand marketing
- Elegant and sophisticated presentation
- Commercial-grade production quality
- Attention to detail in product showcase

USER'S SPECIFIC REQUIREMENTS:
"""

    enhancement_suffix = """

ADDITIONAL QUALITY GUIDELINES:
- Ensure smooth transitions and natural motion
- Maintain consistent lighting throughout
- Keep the product as the clear focal point
- Use professional camera techniques (slow pans, tracking shots, or dolly movements)
- Apply subtle motion blur for cinematic feel
- Ensure brand-appropriate tone and style"""

    return f"{enhancement_prefix}{user_prompt}{enhancement_suffix}"


if __name__ == "__main__":
    mcp.run()
