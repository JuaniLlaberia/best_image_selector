import logging

import numpy as np
from scipy.ndimage import convolve
import mediapipe as mp
from PIL import Image, ImageFile, ImageStat

class PreProcessingStep:
    """
    Class to pre-process images. Performing cleaning and filtering
    """
    def __init__(self):
        """
        Initializes the PreProcessingStep class
        """
        self.blur_threshold = 50
        self.min_width = 400
        self.min_height = 400
        self.min_brightness = 40
        self.max_brightness = 220
        self.min_contrast = 30
        self.default_width = 1080
        self.default_height = 1920

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

    # Helper functions for the analysis
    def _pil_to_grayscale_array(self, img: Image.Image) -> np.ndarray:
        """
        Convert PIL image to grayscale numpy array
        """
        if img.mode != 'L':
            img = img.convert('L')
        return np.array(img, dtype=np.float64)

    def _create_images(self, images_urls: list[str]) -> tuple[list[ImageFile.ImageFile], list[str]]:
        """
        Generates the image objects from the images urls

        Args:
            images_urls (list[str]): List of images paths (urls) to convert to image objects
        Returns:
            tuple[list[ImageFile.ImageFile], list[str]]: Containing the successful transformations and the paths of the failed images
        """
        images: list[ImageFile.ImageFile] = []
        failed_images: list[str] = []

        for url in images_urls:
            try:
                logging.info(f"Transforming `{url}` to image")
                img = Image.open(fp=url)
                images.append(img)
            except Exception as e:
                logging.warning(f"Failed to transform `{url}` to image: {e}")
                failed_images.append(url)

        logging.info(f"Finished Transforming images: {len(images)} successful & {len(failed_images)} failed")
        return images, failed_images

    def _check_resolution_quality(self, img: ImageFile.ImageFile) -> bool:
        """
        Check if the image meets the minimum resolution requirements.

        Args:
            img (ImageFile.ImageFile): PIL Image object
        Returns:
            bool: True if image meets minimum width and height, False otherwise.
        """
        width, height = img.size
        return width >= self.min_width and height >= self.min_height

    def _check_blurry(self, img: ImageFile.ImageFile) -> bool:
        """
        Detect if image is blurry using the Laplacian variance method.

        Args:
            img (ImageFile.ImageFile): PIL Image object
        Returns:
            bool: True if image is NOT blurry (variance >= threshold), False if blurry.
        """
        # Convert to grayscale array
        gray = self._pil_to_grayscale_array(img)
        # Apply Laplacian filter
        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]])
        # Convolve
        laplacian = convolve(input=gray,
                             weights=laplacian_kernel)
        # Calculate variance
        variance = laplacian.var()

        return variance >= self.blur_threshold

    def _check_brightness(self, img: ImageFile.ImageFile) -> bool:
        """
        Check if image brightness is within the allowed range.

        Args:
            img (ImageFile.ImageFile): PIL Image object
        Returns:
            bool: True if brightness is within [min_brightness, max_brightness], else False.
        """
        # Convert image to RGB
        image = img.convert("RGB")
        # Convert to HSV
        image_hsv = image.convert("HSV")
        # Check brightness
        stat = ImageStat.Stat(image_hsv)
        brightness = stat.mean[2]
        return self.min_brightness <= brightness <= self.max_brightness

    def _check_contrast(self, img: ImageFile.ImageFile) -> bool:
        """
        Check if the image contrast meets the minimum threshold.

        Args:
            img (ImageFile.ImageFile): PIL Image object
        Returns:
            bool: True if contrast (stddev of grayscale) >= min_contrast, else False.
        """
        stat = ImageStat.Stat(img.convert('L'))
        return stat.stddev[0] >= self.min_contrast

    def _filter_images(self, images: list[ImageFile.ImageFile]) -> tuple[list[ImageFile.ImageFile], list[ImageFile.ImageFile]]:
        """
        Filter images based on resolution, blurriness, brightness, and contrast.

        Args:
            images (list[ImageFile.ImageFile]): List of PIL Image objects
        Returns:
            tuple[list[ImageFile.ImageFile], list[ImageFile.ImageFile]]: Valid and invalid images
        """
        valid_images: list[ImageFile.ImageFile] = []
        invalid_images: list[ImageFile.ImageFile] = []

        for image in images:
            is_valid = True
            # Check resolution quality
            if not self._check_resolution_quality(image):
                logging.warning(f"{image.filename} fail in resolution check")
                is_valid = False
            # Check if image is blurry
            if not self._check_blurry(image):
                logging.warning(f"{image.filename} fail in blury check")
                is_valid = False
            # Check brightness and contrast
            if not self._check_contrast(image) or not self._check_brightness(image):
                logging.warning(f"{image.filename} fail in contract or brightness check")
                is_valid = False

            # If it passed all checks -> Valid image
            if is_valid:
                valid_images.append(image)
            else:
                invalid_images.append(image)

        logging.info(f"Filtered images was successful with {len(valid_images)} valid images & {len(invalid_images)} invalid images")
        return valid_images, invalid_images

    def _detect_main_person(self, img: ImageFile.ImageFile) -> tuple[int, int] | None:
        """
        Detect the main person in the image using MediaPipe face detection.
        Returns the center coordinates of the largest detected face.

        Args:
            img (ImageFile.ImageFile): PIL Image object
        Returns:
            tuple[int, int] | None: (center_x, center_y) of the main person's face, or None if no face detected
        """
        # Convert PIL image to numpy array for MediaPipe
        img_array = np.array(img.convert('RGB'))
        # Detect faces
        results = self.face_detection.process(img_array)
        if not results.detections:
            logging.info(f"No face detected in {img.filename}")
            return None

        # Find the largest face (assumed to be the main person)
        largest_face = None
        largest_area = 0

        img_width, img_height = img.size

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * img_width)
            y = int(bbox.ymin * img_height)
            width = int(bbox.width * img_width)
            height = int(bbox.height * img_height)

            area = width * height

            if area > largest_area:
                largest_area = area
                # Calculate center of the face
                center_x = x + width // 2
                center_y = y + height // 2
                largest_face = (center_x, center_y)

        return largest_face

    def _reshape_image(self, img: ImageFile.ImageFile) -> ImageFile.ImageFile:
        """
        Crop image to 1080x1920 (portrait) centered on the main person if detected,
        otherwise centered on the image. Scales the image to cover the target dimensions
        while maintaining aspect ratio, then crops.

        Args:
            img: PIL Image object to crop
        Returns:
            Cropped and centered PIL Image object (1080x1920)
        """
        original_width, original_height = img.size

        # Detect main person
        person_center = self._detect_main_person(img)

        # Calculate scaling factor to ensure image covers target dimensions
        scale_width = self.default_width / original_width
        scale_height = self.default_height / original_height
        scale = max(scale_width, scale_height)

        # Resize image to cover target dimensions while maintaining aspect ratio
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Calculate crop box
        if person_center:
            # Scale the person center coordinates
            scaled_center_x = int(person_center[0] * scale)
            scaled_center_y = int(person_center[1] * scale)

            # Center crop around the person
            left = scaled_center_x - self.default_width // 2
            top = scaled_center_y - self.default_height // 2

            # Ensure crop box stays within image boundaries
            left = max(0, min(left, new_width - self.default_width))
            top = max(0, min(top, new_height - self.default_height))
        else:
            # No person detected, center crop on image
            left = (new_width - self.default_width) // 2
            top = (new_height - self.default_height) // 2

        right = left + self.default_width
        bottom = top + self.default_height

        # Crop the image
        return img_resized.crop((left, top, right, bottom))

    def _reshape_images(self, images: list[ImageFile.ImageFile]) -> list[ImageFile.ImageFile]:
        """
        Reshape a list of images to the default dimensions (1080x1920).

        Args:
            images (list[ImageFile.ImageFile]): List of PIL Image objects
        Returns:
            list[ImageFile.ImageFile]: List of reshaped images
        """
        reshaped_images: list[ImageFile.ImageFile] = []
        for image in images:
            reshaped_images.append(self._reshape_image(image))

        return reshaped_images

    def run(self, images_urls: list[str]) -> tuple[list[ImageFile.ImageFile], list[ImageFile.ImageFile], list[str]]:
        """
        Main method to process images: loads, reshapes, and filters them.

        Args:
            images_urls (list[str]): List of image file paths
        Returns:
            tuple: (valid_images, invalid_images, failed_images)
        """
        logging.info(f"Running image pre-processing pipeline step with {len(images_urls)} images...")
        # Transform images paths into images objects
        images, failed_images = self._create_images(images_urls)
        # Analyze and filter the images
        valid_images, invalid_images = self._filter_images(images)
        # Reshape valid images (now centers on detected person)
        reshaped_images = self._reshape_images(valid_images)

        return reshaped_images, invalid_images, failed_images
