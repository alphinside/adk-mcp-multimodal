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
    - Make ONE type of change per tool call (background OR lighting OR props OR arrangement)
    - For complex edits, chain multiple tool calls together
    - BE AS DETAILED AS POSSIBLE in the change_description for best results!

    Args:
        change_description: What do you want to do? BE VERY DETAILED AND SPECIFIC!

                          **The more details you provide, the better the result.**
                          Focus on ONE type of change, but describe it thoroughly.

                          For BACKGROUND changes:
                          - "change background to soft pure white with subtle gradient from top to bottom, clean and minimal aesthetic"
                          - "replace background with rustic dark wood table surface with natural grain texture visible, warm brown tones"

                          For ADDING PROPS:
                          - "add fresh pink roses and eucalyptus leaves arranged naturally around the product on the left and right sides, 
                            with some petals scattered in front"
                          - "add fresh basil leaves and cherry tomatoes scattered around the product naturally"

                          For LIGHTING changes:
                          - "add soft natural window light coming from the left side at 45 degree angle, creating gentle shadows on the 
                            right side, warm morning atmosphere"
                          - "increase brightness with soft diffused studio lighting from above, eliminating harsh shadows"

                          For ARRANGEMENT/POSITIONING:
                          - "reposition product to be perfectly centered in frame with equal space on all sides"
                          - "arrange these three products in a horizontal line, evenly spaced with 2 inches between each"

                          Note: When combining multiple products, you can include background/lighting in the initial arrangement since it's 
                                one cohesive setup
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
        # Example 1: BACKGROUND change only
        result = await edit_product_asset(
            change_description="change background to soft pure white with subtle gradient from top to bottom, clean and minimal aesthetic",
            image_artifact_ids=["product_shot_123.png"]
        )

        # Example 2: ADD PROPS only
        result = await edit_product_asset(
            change_description="add fresh green eucalyptus leaves and sprigs arranged naturally around the product on both sides, with water droplets",
            image_artifact_ids=["skincare_789.png"]
        )

        # Example 3: LIGHTING change only
        result = await edit_product_asset(
            change_description="add soft natural window light coming from the left side at 45 degree angle, creating gentle shadows on the right, warm morning atmosphere",
            image_artifact_ids=["candle_123.png"]
        )

        # Example 4: ARRANGEMENT (combining multiple products - can include cohesive setup details)
        result = await edit_product_asset(
            change_description="arrange these three candles in a perfect horizontal line, centered in frame, evenly spaced with 2 inches between each, on a clean white background with soft diffused lighting from above",
            image_artifact_ids=["candle1.png", "candle2.png", "candle3.png"]
        )

        # Example 5: Chaining multiple edits for complex results
        # Step 1: Change background
        result1 = await edit_product_asset(
            change_description="change background to rustic dark wood table with natural grain texture visible, warm brown tones",
            image_artifact_ids=["product_original.png"]
        )

        # Step 2: Add lighting (using result from step 1)
        result2 = await edit_product_asset(
            change_description="add soft warm natural morning light from the left side at 45 degree angle, creating gentle shadows on the right, cozy and inviting atmosphere",
            image_artifact_ids=[result1["tool_response_artifact_id"]]
        )

        # Step 3: Add props (using result from step 2)
        result3 = await edit_product_asset(
            change_description="add fresh lavender sprigs and dried flowers arranged naturally on the left and right sides of the product",
            image_artifact_ids=[result2["tool_response_artifact_id"]]
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
            if artifact is None:
                logging.error(f"Artifact {img_id} not found")
                return {
                    "status": "error",
                    "tool_response_artifact_id": "",
                    "tool_input_artifact_ids": "",
                    "edit_prompt": change_description,
                    "message": f"Artifact {img_id} not found",
                }

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
