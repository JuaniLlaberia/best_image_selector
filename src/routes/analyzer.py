import os
import logging
from flask import Blueprint, request, jsonify

from src.pipeline.steps.pre_processing_step import PreProcessingStep
from src.pipeline.steps.semantic_similarity_step import SemanticSimilarityStep
from src.pipeline.steps.aesthetic_analyzer_step import AestheticAnalyzerStep

images_analyzer_bp = Blueprint("images_analyzer", __name__)

@images_analyzer_bp.route("/analyze", methods=["POST"])
def generate_titles():
    """
    Analyzes a set of images to select the best event image based on
    preprocessing, semantic similarity with a provided summary, and
    aesthetic scoring.

    Responses
    - 200 OK:
      `{ "message": "Successfully selected event image" }`
    - 400 Bad Request:
      Missing fields, empty image list, invalid data, or preprocessing errors.
    - 500 Internal Server Error:
      Unexpected failure during the analysis pipeline.

    Returns
    JSON response with status code indicating the result of the operation.
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        imagesJSON = request.get_json()

        if "summary" not in imagesJSON or "images_paths" not in imagesJSON:
            return jsonify({"error": "Request body must have 'summary' & 'images_paths'"}), 400

        if len(imagesJSON["images_paths"]) == 0:
            return jsonify({"error": "You must provide at least 1 event image"}), 400

        # Initialize steps
        preprocessing_step = PreProcessingStep()
        semantic_similarity_step = SemanticSimilarityStep()
        aesthetic_analyzer_step = AestheticAnalyzerStep()

        # 1. Run the images pre-processing
        valid_images, invalid_images, failed_images = preprocessing_step.run(images_urls=imagesJSON["images_paths"])
        logging.info(f"Successfully pre-processed images: {len(valid_images)} valid imgs, {len(invalid_images)} invalid imgs, {len(failed_images)} failed imgs")

        # 2. Run the semantic similarity step
        images, images_features = semantic_similarity_step.run(images=valid_images,
                                     summary=imagesJSON["summary"])

        # 3. Aesthetic analysis step
        final_image = aesthetic_analyzer_step.run(images=images,
                                                  images_features=images_features)

        logging.info("Storing image in provided path...")
        final_image.save(os.getenv("PATH_TO_SAVE"))

        logging.info(f"Successfully selected event image")
        return jsonify({"message": "Successfully selected event image"}), 200

    except ValueError as e:
        logging.error(f"Validation error: {e}")
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        logging.error(f"Unexpected error in generate_titles endpoint: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
