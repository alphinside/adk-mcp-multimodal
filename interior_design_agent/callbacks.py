from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.genai.types import Part
from datetime import datetime
import hashlib


async def before_model_modifier(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    for content in llm_request.contents:
        if not content.parts:
            continue

        modified_parts = []
        for _idx, part in enumerate(content.parts):
            if (
                part.function_response
                and part.function_response.name
                in ["generate_concept_image", "edit_image"]
                and part.function_response.response["generated_image_artifact_id"]
            ):
                artifact_id = part.function_response.response[
                    "generated_image_artifact_id"
                ]
                artifact = await callback_context.load_artifact(filename=artifact_id)

                # Original function response
                modified_parts.append(part)

                # Artifact content appended after function response
                modified_parts.append(
                    Part(
                        text=(
                            f"[Function Response Artifact] Below is the content of artifact ID : {artifact_id}"
                        )
                    )
                )
                modified_parts.append(artifact)
            elif (part.inline_data and _idx == 0) or (
                part.inline_data
                and content.parts[_idx - 1].text
                and not content.parts[_idx - 1].text.startswith(
                    "[Function Response Artifact]"
                )
            ):
                # Create timestamp-based identifier with short hash
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Generate short hash from the image data or timestamp
                hash_input = timestamp.encode("utf-8")
                short_hash = hashlib.md5(hash_input).hexdigest()[:8]

                # Extract file extension from mime type
                mime_type = part.inline_data.mime_type
                extension = mime_type.split("/")[-1]

                artifact_id = f"usr_upl_img_{short_hash}.{extension}"

                await callback_context.save_artifact(
                    filename=artifact_id, artifact=part
                )
                modified_parts.append(
                    Part(
                        text=(
                            f"[User Uploaded Artifact] Below is the content of artifact ID : {artifact_id}"
                        )
                    )
                )
                modified_parts.append(part)
            else:
                modified_parts.append(part)

        content.parts = modified_parts
