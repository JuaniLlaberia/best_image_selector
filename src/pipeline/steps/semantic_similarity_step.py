import logging
from typing import Callable

import open_clip
import torch
from PIL import Image, ImageFile

class SemanticSimilarityStep:
    """
    A pipeline step for semantic similarity analysis between images and a text summary using OpenCLIP.
    Loads a CLIP model, preprocesses images and text, computes embeddings, and selects the most semantically similar images to a given summary.
    """
    def __init__(self) -> None:
        """
        Initializes the SemanticSimilarityStep by loading the CLIP model, preprocessor, and tokenizer.
        Raises an exception if the model or tokenizer fails to load.
        """
        # Config values
        self.model_name: str = "ViT-L-14"
        self.pretrained: str = "openai"

        self.model: torch.nn.Module | None = None
        self.preprocessor: Callable[[Image.Image], torch.Tensor] | None = None
        self.tokenizer: open_clip.SimpleTokenizer | None = None

        # Initialize model
        self._init_model()

        if not self.model and not self.tokenizer:
            raise Exception(f"Failed to load {self.model_name} model and tokenizer")

        self.model.to("cpu")
        self.model.eval()

    def _init_model(self) -> None:
        """
        Loads the CLIP model, preprocessor, and tokenizer using OpenCLIP.
        """
        logging.info("Initializing image model and tokenizer...")
        try:
            model, _, preprocessor = open_clip.create_model_and_transforms(model_name=self.model_name,
                                                  pretrained=self.pretrained)
            tokenizer = open_clip.get_tokenizer(model_name=self.model_name)

            logging.info(f"Successfully loaded {self.model_name} model & tokenizer")
            self.model = model
            self.preprocessor = preprocessor
            self.tokenizer = tokenizer

        except Exception as e:
            logging.error(f"Failed to load {self.model_name} model: {e}")

    def _embed_images(self, images: list[ImageFile.ImageFile]) -> torch.Tensor:
        """
        Embeds a list of images into feature vectors using the CLIP model.

        Args:
            images (list[ImageFile.ImageFile]): List of PIL ImageFile objects.
        Returns:
            torch.Tensor: Normalized image feature vectors of shape (N, D).
        """
        # Pre-process images
        processed_images = torch.stack([self.preprocessor(img) for img in images])

        with torch.no_grad():
            image_features = self.model.encode_image(processed_images)
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features

    def _embed_text(self, text: str) -> torch.Tensor:
        """
        Embeds a text string into a feature vector using the CLIP model.

        Args:
            text (str): The text to embed.
        Returns:
            torch.Tensor: Normalized text feature vector of shape (1, D).
        """
        # Pre-process text
        processed_text = self.tokenizer(texts=[text])

        with torch.no_grad():
            text_features = self.model.encode_text(processed_text)
            # Normalize features
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def _calculate_similary(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the semantic similarity between text and image features.

        Args:
            text_features (torch.Tensor): Text feature vector of shape (1, D).
            image_features (torch.Tensor): Image feature vectors of shape (N, D).
        Returns:
            torch.Tensor: Similarity scores (softmaxed), shape (1, N).
        """
        similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
        return similarity

    def run(
        self,
        images: list[ImageFile.ImageFile],
        summary: str
    ) -> tuple[list[ImageFile.ImageFile], list[torch.Tensor]]:
        """
        Runs the semantic similarity step: embeds images and summary, computes similarity, and selects top images and their features.

        Args:
            images (list[ImageFile.ImageFile]): List of images to analyze.
            summary (str): Text summary to compare against images.
        Returns:
            tuple[list[ImageFile.ImageFile], list[torch.Tensor]]:
                - List of images most similar to the summary.
                - List of their corresponding feature vectors.
        """
        min_images_to_keep = len(images) // 3

        # Pre-process image and summary
        image_features = self._embed_images(images=images)
        text_features = self._embed_text(text=summary)
        # Calculate similarity
        similarity_scores = self._calculate_similary(text_features=text_features,
                                                     image_features=image_features)
        # Use the N wanted images
        tok_k_images = torch.topk(input=similarity_scores,
                                  k=min_images_to_keep)

        # Return images to keep and their features
        images_to_keep = [images[idx] for idx in tok_k_images.indices.flatten().tolist()]
        images_features_to_keep = [image_features[idx] for idx in tok_k_images.indices.flatten().tolist()]

        return images_to_keep, images_features_to_keep
