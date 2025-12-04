import logging
from flask import jsonify, Blueprint

health_bp = Blueprint("health", __name__)

@health_bp.route("/health", methods=["GET"])
def health_check():
    logging.info("Running health check endpoint")
    return jsonify({"status": "healthy"}), 200
