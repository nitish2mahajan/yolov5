import os
import torch

from helpers import draw_box, url_to_img, img_to_bytes


class PythonPredictor:

    def __init__(self, config):
        """
        Called once before the API becomes available. Performs setup such as 
        downloading/initializing the model or downloading a vocabulary.
        Args:
            config (required): Dictionary passed from API configuration
        """

        # Load a model from directory
        self.model = torch.hub.load('yolov5', 'yolov5s.pt', pretrained=True)

    def predict(self, payload):
        """
        Called once per request. Preprocesses the request payload (if necessary), 
        runs inference, and postprocesses the inference output (if necessary).
        Args:
            payload: The request payload
        Returns:
            Prediction or a batch of predictions.
        """

        # Convert url image to PIL format
        img = url_to_img(payload["url"])

        # Run a model
        results = self.model(img)

        # Draw boxes
        boxes = results.xyxy[0].cpu().numpy()
        box_img = draw_box(img, boxes)

        # Return an image in the base64 format
        return img_to_bytes(box_img)
