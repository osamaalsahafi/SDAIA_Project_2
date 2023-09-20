import torch
import cv2
import numpy as np


class MiDaSInferenceLocal:
    def __init__(
        self, model_name="facebookresearch/midas:latest", model_path: str = None
    ):
        """
        Initialize the MiDaS model for real-time inference using a locally saved model.

        Args:
            model_path (str): Path to the saved MiDaS model (.pt file).
        """
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image):
        """
        Preprocess the input image for inference.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to torch tensor and normalize
        tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1).unsqueeze(0)
        tensor = tensor / 255.0

        return tensor

    def infer(self, image):
        """
        Perform inference on the input image.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            torch.Tensor: Depth map tensor.
        """
        tensor = self.preprocess(image)
        with torch.no_grad():
            tensor = tensor.to(self.device)
            depth_map = self.model(tensor)
        return depth_map

    def postprocess(self, depth_map):
        """
        Postprocess the depth map for visualization.

        Args:
            depth_map (torch.Tensor): Depth map tensor from the model.

        Returns:
            np.ndarray: Processed depth map for visualization.
        """
        # Convert tensor to numpy array
        depth_map = depth_map.squeeze().cpu().numpy()

        # Normalize for visualization
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        return depth_map
