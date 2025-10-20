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

async def edit_product_asset(
    tool_context: ToolContext,
    change_description: str,
    image_artifact_ids: list = [],
) -> dict[str, str]:
    """Modify an existing product photo or combine multiple product photos.

    This tool lets you make changes to product photos. You can:
    - Edit a single photo (change background, lighting, colors, etc.)
    - Combine multiple products into one photo (arrange them side by side, create bundles, etc.)

    **IMPORTANT**: 
    - Make ONE change at a time. If you want multiple changes, do them one by one.
    - BE AS DETAILED AS POSSIBLE in the change_description for best results!

    Args:
        change_description: What do you want to do? BE VERY DETAILED AND SPECIFIC!
                          
                          **The more details you provide, the better the result.**
                          Include specifics about colors, positions, lighting, style, mood, etc.
                          
                          For single image edits:
                          - GOOD: "change the background to pure white, clean and minimal"
                          - BETTER: "change to soft white background with subtle gradient, studio lighting"
                          - GOOD: "add some flowers around the product"
                          - BETTER: "add fresh pink roses and eucalyptus leaves arranged naturally around the product on the left and right sides"
                          - GOOD: "make the lighting brighter"
                          - BETTER: "increase brightness with soft natural window light from the left, creating gentle shadows"
                          
                          For multiple images:
                          - GOOD: "arrange these products side by side"
                          - BETTER: "arrange these three products in a horizontal line on a white marble surface, evenly spaced, with soft natural lighting from above"
                          - GOOD: "create a product bundle"
                          - BETTER: "create an elegant spa gift set arrangement with products centered, surrounded by fresh eucalyptus leaves and soft white towels, on a light wood surface"
                          
                          Always include: positioning, spacing, lighting direction, background details, mood/style
        image_artifact_ids: List of image IDs to edit or combine.
                          - For single image: provide a list with one item (e.g., ["product.png"])
                          - For multiple images: provide a list with multiple items (e.g., ["product1.png", "product2.png"])
                          Use multiple images to combine products into one photo.

    Returns:
        dict with keys:
            - 'tool_response_artifact_id': Artifact ID for the edited image
            - 'tool_input_artifact_ids': Comma-separated list of input artifact IDs
            - 'edit_prompt': The full edit prompt used
            - 'status': Success or error status
            - 'message': Additional information or error details

    Examples:
        # Edit a single image - change the background (DETAILED!)
        result = await edit_product_asset(
            change_description="change background to soft pure white with subtle gradient, studio lighting from top, clean and minimal aesthetic",
            image_artifact_ids=["product_shot_123.png"]
        )

        # Add something to the scene (DETAILED!)
        result = await edit_product_asset(
            change_description="add fresh green eucalyptus leaves and sprigs arranged naturally around the product on both sides, with water droplets, on a white marble surface",
            image_artifact_ids=["skincare_789.png"]
        )

        # Combine multiple products together (DETAILED!)
        result = await edit_product_asset(
            change_description="arrange these three candles in a perfect horizontal line on a clean white background, evenly spaced with 2 inches between each, soft diffused lighting from above creating subtle shadows",
            image_artifact_ids=["candle1.png", "candle2.png", "candle3.png"]
        )
        
        # Create a product bundle (DETAILED!)
        result = await edit_product_asset(
            change_description="create an elegant spa gift set with items arranged in a semi-circle on light oak wood surface, surrounded by fresh lavender sprigs and white fluffy towels, warm natural window light from the left, relaxing and luxurious mood",
            image_artifact_ids=["lotion.png", "soap.png", "scrub.png"]
        )

        # Make multiple changes by doing them one at a time (DETAILED!)
        result1 = await edit_product_asset(
            change_description="change background to rustic dark wood table with natural grain texture visible, warm brown tones",
            image_artifact_ids=["product_original.png"]
        )
        # Use the result from the first edit
        result2 = await edit_product_asset(
            change_description="add soft warm natural morning light from the left side at 45 degree angle, creating gentle shadows on the right, cozy and inviting atmosphere",
            image_artifact_ids=[result1["tool_response_artifact_id"]]
        )
    """
    try:
        # Validate input
        if not image_artifact_ids:
            return {
                "status": "error",
                "tool_response_artifact_id": "",
                "tool_input_artifact_ids": "",
                "edit_prompt": change_description,
                "message": "No images provided. Please provide image_artifact_ids as a list.",
            }
        
        # Load all images
        image_artifacts = []
        for img_id in image_artifact_ids:
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
        logging.info("Gemini Flash Image: response.candidates: ", response.candidates)
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                artifact_id = f"edited_img_{tool_context.function_call_id}.png"
                await tool_context.save_artifact(filename=artifact_id, artifact=part)

        input_ids_str = ", ".join(image_artifact_ids)
        return {
            "status": "success",
            "tool_response_artifact_id": artifact_id,
            "tool_input_artifact_ids": input_ids_str,
            "edit_prompt": full_edit_prompt,
            "message": f"Image edited successfully using {len(image_artifacts)} input image(s)",
        }
    except Exception as e:
        logging.error(e)
        input_ids_str = ", ".join(image_artifact_ids) if image_artifact_ids else ""
        return {
            "status": "error",
            "tool_response_artifact_id": "",
            "tool_input_artifact_ids": input_ids_str,
            "edit_prompt": change_description,
            "message": f"Error editing image: {str(e)}",
        }
