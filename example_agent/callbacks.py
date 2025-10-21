from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.genai.types import Part
import hashlib
from typing import List


async def before_model_modifier(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """Modify LLM request to include artifact references for images."""
    for content in llm_request.contents:
        if not content.parts:
            continue

        modified_parts = []
        for idx, part in enumerate(content.parts):
            # Handle user-uploaded inline images
            if part.inline_data:
                processed_parts = await _process_inline_data_part(
                    part, callback_context
                )
            # Handle function response parts for image generation/editing
            elif part.function_response:
                if part.function_response.name in [
                    "edit_product_asset",
                ]:
                    processed_parts = await _process_function_response_part(
                        part, callback_context
                    )
                else:
                    processed_parts = [part]
            # Default: keep part as-is
            else:
                processed_parts = [part]

            modified_parts.extend(processed_parts)

        content.parts = modified_parts


async def _process_inline_data_part(
    part: Part, callback_context: CallbackContext
) -> List[Part]:
    """Process inline data parts (user-uploaded images).

    Returns:
        List of parts including artifact marker and the image.
    """
    artifact_id = _generate_artifact_id(part)

    # Save artifact if it doesn't exist
    if artifact_id not in await callback_context.list_artifacts():
        await callback_context.save_artifact(filename=artifact_id, artifact=part)

    return [
        Part(
            text=f"[User Uploaded Artifact] Below is the content of artifact ID : {artifact_id}"
        ),
        part,
    ]


def _generate_artifact_id(part: Part) -> str:
    """Generate a unique artifact ID for user uploaded image.

    Returns:
        Hash-based artifact ID with proper file extension.
    """
    filename = part.inline_data.display_name or "uploaded_image"
    image_data = part.inline_data.data

    # Combine filename and image data for hash
    hash_input = filename.encode("utf-8") + image_data
    content_hash = hashlib.sha256(hash_input).hexdigest()[:16]

    # Extract file extension from mime type
    mime_type = part.inline_data.mime_type
    extension = mime_type.split("/")[-1]

    return f"usr_upl_img_{content_hash}.{extension}"


async def _process_function_response_part(
    part: Part, callback_context: CallbackContext
) -> List[Part]:
    """Process function response parts and append artifacts.

    Returns:
        List of parts including the original function response and artifact.
    """
    artifact_id = part.function_response.response.get("tool_response_artifact_id")

    if not artifact_id:
        return [part]

    artifact = await callback_context.load_artifact(filename=artifact_id)

    return [
        part,  # Original function response
        Part(
            text=f"[Tool Response Artifact] Below is the content of artifact ID : {artifact_id}"
        ),
        artifact,
    ]
