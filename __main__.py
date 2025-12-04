import logging
import sys
from flask import Flask
from dotenv import load_dotenv

from src.routes.analyzer import images_analyzer_bp
from src.routes.health import health_bp

load_dotenv()

# Configure logging to stdout for container environments
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def create_app():
    app = Flask(__name__)
    app.register_blueprint(images_analyzer_bp)
    app.register_blueprint(health_bp)

    return app

def main():
    """
    Main entry point for the Flask events images analyzer API.
    Starts the web server to group articles via HTTP POST requests.
    """
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    main()
