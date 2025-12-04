import logging

import torch
import torch.nn as nn
from PIL import ImageFile

class AestheticModel(nn.Module):
    """
    Neural network model for aesthetic analysis.
    Architecture: Linear layers with ReLU activations, outputting a single score.
    """
    def __init__(self) -> None:
        """
        Initialize the AestheticModel with a predefined architecture.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 768).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        return self.layers(x)

class AestheticAnalyzerStep:
    """
    Step for analyzing image aesthetics using a pretrained neural network model.
    Loads the model, predicts scores for input images, and selects the top image.
    """
    def __init__(self) -> None:
        """
        Initialize the AestheticAnalyzerStep, loading the pretrained model.
        Raises:
            Exception: If the model fails to load.
        """
        self.model_path: str = "D:\Punto_Medio\images-analyzer\models\sac+logos+ava1-l14-linearMSE.pth"
        self.aesthetic_analyzer_model: nn.Module | None = None
        # Initialize model
        self._init_model()

        if not self.aesthetic_analyzer_model:
            raise Exception(f"Failed to load model in `{self.model_path}`")

    def _init_model(self) -> None:
        """
        Internal method to initialize and load the aesthetic analyzer model.
        Loads weights from the specified path and sets the model to evaluation mode.
        """
        # Create model instance
        aesthetic_analyzer_model = AestheticModel()
        # Load model's pretrained weights
        aesthetic_analyzer_model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        # Set model to evaluation mode
        aesthetic_analyzer_model.eval()

        logging.info(f"Success loaded model ({self.model_path})")
        self.aesthetic_analyzer_model = aesthetic_analyzer_model

    def _predict_images(
        self,
        images: list[ImageFile.ImageFile],
        images_features: torch.Tensor
    ) -> ImageFile.ImageFile:
        """
        Predict aesthetic scores for images and select the top image.
        Args:
            images (list[ImageFile.ImageFile]): List of image objects.
            images_features (torch.Tensor): Tensor of image features (shape: [N, 768]).
        Returns:
            ImageFile.ImageFile: The image with the highest predicted aesthetic score.
        """
        with torch.no_grad():
            scores = self.aesthetic_analyzer_model(images_features)

        tok_k_images = torch.topk(input=scores, k=1)
        image_to_keep = images[tok_k_images.indices[0].item()]
        return image_to_keep

    def run(
        self,
        images: list[ImageFile.ImageFile],
        images_features: list[torch.Tensor]
    ) -> ImageFile.ImageFile:
        """
        Run the aesthetic analysis step.
        Args:
            images (list[ImageFile.ImageFile]): List of image objects.
            images_features (list[torch.Tensor]): List of image feature tensors.
        Returns:
            ImageFile.ImageFile: The image with the highest predicted aesthetic score.
        """
        if isinstance(images_features, list):
            images_features = torch.stack(images_features)

        return self._predict_images(images, images_features)