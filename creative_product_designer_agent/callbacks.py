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
            # Handle function response parts for image generation/editing
            if part.function_response and part.function_response.name in [
                "edit_product_asset",
            ]:
                processed_parts = await _process_function_response_part(
                    part, callback_context
                )
            # Handle user-uploaded inline images
            elif _is_user_uploaded_image(part, idx, content.parts):
                processed_parts = await _process_inline_data_part(
                    part, callback_context
                )
            # Default: keep part as-is
            else:
                processed_parts = [part]

            modified_parts.extend(processed_parts)

        content.parts = modified_parts


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


def _is_user_uploaded_image(part: Part, idx: int, parts: List[Part]) -> bool:
    """Check if part is a user-uploaded image that needs processing.

    Args:
        part: The current part to check.
        idx: Index of the current part.
        parts: List of all parts.

    Returns:
        True if this is a user-uploaded image that should be processed.
    """
    if not part.inline_data:
        return False

    # First inline data is always user-uploaded
    if idx == 0:
        return True

    # Check if previous part is text and not an artifact marker
    previous_part = parts[idx - 1]
    if not previous_part.text:
        return False

    # Inline data preceded by artifact markers should not be processed again
    is_artifact_marker = previous_part.text.startswith(
        "[Tool Response Artifact]"
    ) or previous_part.text.startswith("[User Uploaded Artifact]")

    return not is_artifact_marker


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
