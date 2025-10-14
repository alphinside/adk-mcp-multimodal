from typing import Optional, Literal
from google import genai
from dotenv import load_dotenv
import os
from google.adk.tools import ToolContext
import logging


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
) -> dict[str, str]:
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
            - 'generated_image_artifact_id': artifact ID for the generated image
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

        artifact_id = ""
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                artifact_id = f"generated_img_{tool_context.function_call_id}.png"

                await tool_context.save_artifact(filename=artifact_id, artifact=part)

        return {
            "status": "success",
            "prompt": full_prompt,
            "generated_image_artifact_id": artifact_id,
            "message": "Image generated successfully"
            if artifact_id
            else "No image generated",
        }
    except Exception as e:
        logging.error(e)
        return {
            "status": "error",
            "prompt": full_prompt,
            "generated_image_artifact_id": "",
            "message": f"Error generating image: {str(e)}",
        }


async def edit_image(
    tool_context: ToolContext,
    image_artifact_id: str,
    edit_description: str,
    preserve_structure: bool = True,
    intensity: Literal["subtle", "medium", "dramatic"] = "medium",
) -> dict[str, str]:
    """Edit an existing interior image based on modification instructions.

    This tool modifies existing room photos or furniture images according to user
    specifications. Use this when users provide a photo of their current space or
    furniture piece and want to see specific changes, such as different furniture,
    colors, layouts, styles, or modifications to individual furniture items.

    **CRITICAL: ONE FOCUSED EDIT DESCRIPTION PER CALL**: This tool should be invoked with ONE
    focused edit description at a time. It CANNOT handle multiple edit descriptions
    in a single call. If the user wants multiple changes, you must call this tool multiple times
    sequentially, using the output from one call as input to the next.

    **IMPORTANT for tidying/organizing rooms**: If the user wants to make a room
    tidier or more organized, you MUST explicitly specify which objects need to be
    removed and which objects need to be repositioned. Vague requests like "make it
    tidier" will not work - be specific about what changes to make.

    Args:
        image_artifact_id: Artifact image identifier. Should be a clear photo of an
            interior space OR individual furniture piece. **This can be used to edit
            both complete room photos and standalone furniture images** (e.g., to change
            a sofa's color, modify a chair's upholstery, or adjust furniture details).
        edit_description: Single focused description of the change to make.
                         For tidying/organizing: specify which specific objects to remove
                         or reposition in one focused action.
                         Examples of single focused edits:
                         - "change wall color to sage green"
                         - "replace the sofa with a modern sectional"
                         - "add warm ambient lighting"
                         - "remove the stack of magazines from the coffee table"
                         - "reposition the side table against the wall"
        preserve_structure: If True, maintains the room's architectural structure
                          and only modifies specified elements. If False, allows
                          more dramatic transformations. Default: True.
        intensity: How dramatic the changes should be. Options: "subtle", "medium",
                  "dramatic". Default: "medium".

    Returns:
        dict with keys:
            - 'edited_image_artifact_id': Artifact ID for the edited image
            - 'original_image_artifact_id': Artifact ID for the original image
            - 'edit_prompt': The full edit prompt used
            - 'status': Success or error status
            - 'message': Additional information or error details

    Examples:
        # Edit a room photo - change wall color
        result = await edit_image(
            image_artifact_id="room_photo_123.png",
            edit_description="Change the beige walls to a soft blue-gray color",
            preserve_structure=True,
            intensity="medium"
        )

        # Edit individual furniture
        result = await edit_image(
            image_artifact_id="sofa_456.png",
            edit_description="Change the upholstery from beige to deep navy blue velvet",
            preserve_structure=True,
            intensity="subtle"
        )

        # Tidy/organize a room - single focused action (must be explicit)
        result = await edit_image(
            image_artifact_id="messy_room_789.png",
            edit_description="Remove the stack of books and magazines from the coffee table",
            preserve_structure=True,
            intensity="medium"
        )

        # Sequential edits - make multiple changes by chaining calls
        # First edit: change wall color
        result1 = await edit_image(
            image_artifact_id="room_original.png",
            edit_description="Change wall color to sage green",
            preserve_structure=True,
            intensity="medium"
        )
        # Second edit: use output from first edit
        result2 = await edit_image(
            image_artifact_id=result1["edited_image_artifact_id"],
            edit_description="Replace the wooden coffee table with a modern glass one",
            preserve_structure=True,
            intensity="medium"
        )
    """
    try:
        # Load image artifact
        image_artifact = await tool_context.load_artifact(filename=image_artifact_id)

        # Build edit prompt with constraints
        structure_note = (
            "Maintain the original structure and composition (room layout or furniture form)"
            if preserve_structure
            else "Allow significant structural and compositional changes"
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

        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[image_artifact, full_edit_prompt],
        )

        artifact_id = ""
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                artifact_id = f"edited_img_{tool_context.function_call_id}.png"

                await tool_context.save_artifact(filename=artifact_id, artifact=part)

        return {
            "status": "success",
            "edited_image_artifact_id": artifact_id,
            "original_image_artifact_id": image_artifact_id,
            "edit_prompt": full_edit_prompt,
            "message": "Image edited successfully",
        }
    except Exception as e:
        logging.error(e)
        return {
            "status": "error",
            "edited_image_artifact_id": "",
            "original_image_artifact_id": image_artifact_id,
            "edit_prompt": edit_description,
            "message": f"Error editing image: {str(e)}",
        }
