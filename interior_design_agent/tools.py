from typing import Optional
import base64
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from PIL import Image
from io import BytesIO
from google.adk.tools import ToolContext


load_dotenv()

client = genai.Client(
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION"),
)

async def generate_concept_image(
    tool_context: ToolContext,
    description: str,
    style: Optional[str] = None,
    room_type: Optional[str] = None,
    color_scheme: Optional[str] = None,
    additional_details: Optional[str] = None,
) -> dict[str, str | list[str]]:
    """Generate an interior design concept image from a text description.
    
    This tool creates original interior design visualizations based on the provided
    description and optional parameters. Use this when users want to explore new
    design ideas from scratch or visualize a space that doesn't exist yet.
    
    **IMPORTANT: This tool can generate both complete room designs AND individual 
    furniture pieces.** You can use it to visualize specific furniture items, 
    decor elements, or entire room layouts.
    
    Args:
        description: Main description of the interior design concept or furniture piece 
                    (required). Can describe a complete room, a specific furniture item 
                    (e.g., "modern velvet sofa", "minimalist dining table"), or decor 
                    elements. Should be detailed and specific about the desired look and feel.
        style: Design style (e.g., "modern", "minimalist", "scandinavian", 
              "industrial", "bohemian", "traditional", "mid-century modern").
        room_type: Type of room (e.g., "living room", "bedroom", "kitchen", 
                  "bathroom", "home office", "dining room").
        color_scheme: Desired color palette (e.g., "neutral tones", 
                     "warm earth tones", "cool blues and grays", "monochromatic").
        additional_details: Any additional specifications like lighting preferences,
                          furniture styles, materials, or special features.
    
    Returns:
        dict with keys:
            - 'generated_image_artifact_ids': List of artifact IDs for the generated images
            - 'prompt': The full prompt used for generation
            - 'status': Success or error status
            - 'message': Additional information or error details
    
    Examples:
        # Generate a complete room design
        result = await generate_concept_image(
            description="A cozy reading nook with large windows and natural light",
            style="scandinavian",
            room_type="living room",
            color_scheme="warm whites and natural wood tones",
            additional_details="Include built-in bookshelves and a comfortable armchair"
        )
        
        # Generate individual furniture pieces
        result = await generate_concept_image(
            description="A modern sectional sofa with clean lines and tufted cushions",
            style="contemporary",
            color_scheme="charcoal gray with chrome legs",
            additional_details="L-shaped configuration, low profile, modular design"
        )
    """
    # Build comprehensive prompt from parameters
    prompt_parts = [description]
    
    if room_type:
        prompt_parts.append(f"Room type: {room_type}")
    
    if style:
        prompt_parts.append(f"Design style: {style}")
    
    if color_scheme:
        prompt_parts.append(f"Color scheme: {color_scheme}")
    
    if additional_details:
        prompt_parts.append(f"Additional details: {additional_details}")
    
    # Add professional interior design quality markers
    prompt_parts.append(
        "Professional interior design photography, high quality, well-lit, "
        "realistic, detailed, 8k resolution, architectural photography style"
    )
    
    full_prompt = ". ".join(prompt_parts)
    
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[full_prompt],
        )
        
        artifact_count = 0
        artifact_ids = []
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                artifact_id = f"gen_img_{tool_context.function_call_id}_{artifact_count}.png"
                artifact_count += 1
                artifact_ids.append(artifact_id)

                await tool_context.save_artifact(
                    filename=artifact_id,
                    artifact=part
                )

        return {
            "status": "success",
            "prompt": full_prompt,
            "generated_image_artifact_ids": artifact_ids,
            "message": "Image generation not yet implemented. Please integrate with Imagen or similar API.",
        }
    except Exception as e:
        print(e)
        return {
            "status": "error",
            "prompt": full_prompt,
            "generated_image_artifact_ids": [],
            "message": f"Error generating image: {str(e)}",
        }


async def edit_image(
    image_input: str,
    edit_description: str,
    preserve_structure: bool = True,
    intensity: Optional[str] = "medium",
) -> dict[str, str]:
    """Edit an existing interior image based on modification instructions.
    
    This tool modifies existing room photos according to user specifications.
    Use this when users provide a photo of their current space and want to see
    specific changes, such as different furniture, colors, layouts, or styles.
    
    Args:
        image_input: Artifact image identifier. Should be a clear photo of an 
                     interior space.
        edit_description: Detailed description of the changes to make 
                         (e.g., "change wall color to sage green", 
                         "replace the sofa with a modern sectional",
                         "add warm ambient lighting").
        preserve_structure: If True, maintains the room's architectural structure
                          and only modifies specified elements. If False, allows
                          more dramatic transformations. Default: True.
        intensity: How dramatic the changes should be. Options: "subtle", "medium", 
                  "dramatic". Default: "medium".
    
    Returns:
        dict with keys:
            - 'edited_image_url': URL or base64 encoded edited image data
            - 'original_image_url': Reference to the original image
            - 'edit_prompt': The full edit prompt used
            - 'status': Success or error status
            - 'message': Additional information or error details
    
    Example:
        result = await edit_image(
            image_input="image_123.jpg",
            edit_description="Replace the beige walls with a soft blue-gray color and add modern pendant lighting",
            preserve_structure=True,
            intensity="medium"
        )
    """
    try:
        # Load and validate image
        image_data = None
        if Path(image_input).exists():
            with open(image_input, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        elif image_input.startswith("data:image") or len(image_input) > 100:
            # Assume it's already base64 encoded
            image_data = image_input
        else:
            return {
                "status": "error",
                "edited_image_url": "",
                "original_image_url": image_input,
                "edit_prompt": edit_description,
                "message": "Invalid image input. Provide a valid file path or base64 encoded image.",
            }
        
        # Build edit prompt with constraints
        structure_note = (
            "Maintain the room's architectural structure and layout"
            if preserve_structure
            else "Allow significant structural and layout changes"
        )
        
        intensity_guidance = {
            "subtle": "Make minimal, tasteful changes while keeping most elements similar",
            "medium": "Make noticeable improvements while maintaining overall coherence",
            "dramatic": "Transform the space with bold, significant changes",
        }
        
        full_edit_prompt = (
            f"{edit_description}. {structure_note}. "
            f"{intensity_guidance.get(intensity, intensity_guidance['medium'])}. "
            "Professional interior design quality, realistic, photographic style."
        )
        
        # TODO: Implement actual image editing using Imagen or Gemini API
        # This is a placeholder that should be replaced with actual API calls
        # Example with Imagen:
        # response = await imagen_client.edit_image(
        #     base_image=image_data,
        #     mask=None,  # or generate mask based on edit_description
        #     prompt=full_edit_prompt
        # )
        
        return {
            "status": "success",
            "edited_image_url": "",  # Placeholder - should contain actual edited image
            "original_image_url": image_input,
            "edit_prompt": full_edit_prompt,
            "message": "Image editing not yet implemented. Please integrate with Imagen or similar API.",
        }
    except Exception as e:
        return {
            "status": "error",
            "edited_image_url": "",
            "original_image_url": image_input,
            "edit_prompt": edit_description,
            "message": f"Error editing image: {str(e)}",
        }
