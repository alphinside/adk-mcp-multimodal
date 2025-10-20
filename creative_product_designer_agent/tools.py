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


async def generate_product_asset(
    tool_context: ToolContext,
    product_description: str,
    scene_description: Optional[str] = None,
) -> dict[str, str]:
    """Generate a professional product photo for your business.

    This tool creates beautiful product images perfect for social media, your website,
    or online marketplace listings. Just describe what you want to see!

    Args:
        product_description: Describe your product (required).
                           Tell us what it is, what it looks like, and any important details.
                           Example: "handmade lavender soap bar with dried flowers on top"
                           Example: "ceramic coffee mug in navy blue with gold rim"
                           Example: "organic honey jar with wooden dipper"
        scene_description: Describe the setting or background (optional).
                         Where do you want your product? What's around it?
                         Example: "on a white background"
                         Example: "on a wooden table with morning sunlight"
                         Example: "surrounded by fresh ingredients"
                         Example: "in a cozy kitchen setting"
                         Leave empty for a clean, simple background.

    Returns:
        dict with keys:
            - 'tool_response_artifact_id': artifact ID for the generated image
            - 'prompt': The full prompt used for generation
            - 'status': Success or error status
            - 'message': Additional information or error details

    Examples:
        # Simple product photo with clean background
        result = await generate_product_asset(
            product_description="handmade candle in glass jar with wooden wick",
            scene_description="clean white background"
        )

        # Product in a lifestyle setting
        result = await generate_product_asset(
            product_description="organic coffee beans in kraft paper bag",
            scene_description="on rustic wooden table with morning light, coffee cup nearby"
        )
        
        # Just the product description works too!
        result = await generate_product_asset(
            product_description="handcrafted leather wallet in brown"
        )
    """
    # Build comprehensive prompt from parameters
    prompt_parts = [product_description]

    if scene_description:
        prompt_parts.append(f"Setting: {scene_description}")

    # Add professional quality markers
    prompt_parts.append(
        "Professional product photography, high quality, well-lit, "
        "sharp focus, clean and appealing, suitable for business use"
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
            "tool_response_artifact_id": artifact_id,
            "message": "Image generated successfully"
            if artifact_id
            else "No image generated",
        }
    except Exception as e:
        logging.error(e)
        return {
            "status": "error",
            "prompt": full_prompt,
            "tool_response_artifact_id": "",
            "message": f"Error generating image: {str(e)}",
        }


async def edit_product_asset(
    tool_context: ToolContext,
    change_description: str,
    image_artifact_id: str | list[str] = None,
) -> dict[str, str]:
    """Modify an existing product photo or combine multiple product photos.

    This tool lets you make changes to product photos. You can:
    - Edit a single photo (change background, lighting, colors, etc.)
    - Combine multiple products into one photo (arrange them side by side, create bundles, etc.)

    **IMPORTANT**: Make ONE change at a time. If you want multiple changes, we'll do
    them one by one.

    Args:
        change_description: What do you want to do? Be specific!
                          For single image edits:
                          - "change the background to white"
                          - "add some flowers around the product"
                          - "make the lighting brighter"
                          - "remove the cup from the background"
                          - "change the product color to blue"
                          
                          For multiple images:
                          - "arrange these products side by side on white background"
                          - "create a product bundle with all items together"
                          - "put these products on a wooden table"
                          - "show these items arranged nicely for a gift set"
        image_artifact_id: The ID(s) of the image(s) to edit.
                         - For single image: provide a string (e.g., "product.png")
                         - For multiple images: provide a list of strings (e.g., ["product1.png", "product2.png"])
                         Use multiple images to combine products into one photo.

    Returns:
        dict with keys:
            - 'tool_response_artifact_id': Artifact ID for the edited image
            - 'tool_input_artifact_id': Artifact ID(s) of the original image(s)
            - 'edit_prompt': The full edit prompt used
            - 'status': Success or error status
            - 'message': Additional information or error details

    Examples:
        # Edit a single image - change the background
        result = await edit_product_asset(
            change_description="change background to white",
            image_artifact_id="product_shot_123.png"
        )

        # Add something to the scene
        result = await edit_product_asset(
            change_description="add fresh leaves around the product",
            image_artifact_id="skincare_789.png"
        )

        # Combine multiple products together
        result = await edit_product_asset(
            change_description="arrange these three candles side by side on white background",
            image_artifact_id=["candle1.png", "candle2.png", "candle3.png"]
        )
        
        # Create a product bundle
        result = await edit_product_asset(
            change_description="create a spa gift set with all items arranged nicely",
            image_artifact_id=["lotion.png", "soap.png", "scrub.png"]
        )

        # Make multiple changes by doing them one at a time
        result1 = await edit_product_asset(
            change_description="change background to wooden table",
            image_artifact_id="product_original.png"
        )
        # Use the result from the first edit
        result2 = await edit_product_asset(
            change_description="add soft natural lighting",
            image_artifact_id=result1["tool_response_artifact_id"]
        )
    """
    try:
        # Normalize input to list
        if image_artifact_id is None:
            return {
                "status": "error",
                "tool_response_artifact_id": "",
                "tool_input_artifact_id": "",
                "edit_prompt": change_description,
                "message": "No images provided. Please provide image_artifact_id.",
            }
        
        # Convert single string to list for uniform handling
        if isinstance(image_artifact_id, str):
            image_ids = [image_artifact_id]
        else:
            image_ids = image_artifact_id
        
        # Load all images
        image_artifacts = []
        for img_id in image_ids:
            artifact = await tool_context.load_artifact(filename=img_id)
            image_artifacts.append(artifact)
        
        # Build edit prompt
        if len(image_artifacts) > 1:
            full_edit_prompt = (
                f"{change_description}. "
                f"Combine these {len(image_artifacts)} product images together. "
                "Keep all products looking good and professional. "
                "Make natural, appealing composition suitable for business use."
            )
        else:
            full_edit_prompt = (
                f"{change_description}. "
                "Keep the product looking good and professional. "
                "Make natural, appealing changes suitable for business use."
            )
        
        # Build contents list: all images followed by the prompt
        contents = image_artifacts + [full_edit_prompt]

        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
        )

        artifact_id = ""
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                artifact_id = f"edited_img_{tool_context.function_call_id}.png"
                await tool_context.save_artifact(filename=artifact_id, artifact=part)

        input_ids_str = ", ".join(image_ids)
        return {
            "status": "success",
            "tool_response_artifact_id": artifact_id,
            "tool_input_artifact_ids": input_ids_str,
            "edit_prompt": full_edit_prompt,
            "message": f"Image edited successfully using {len(image_artifacts)} input image(s)",
        }
    except Exception as e:
        logging.error(e)
        input_ids_str = ", ".join(image_ids) if 'image_ids' in locals() else ""
        return {
            "status": "error",
            "tool_response_artifact_id": "",
            "tool_input_artifact_ids": input_ids_str,
            "edit_prompt": change_description,
            "message": f"Error editing image: {str(e)}",
        }
