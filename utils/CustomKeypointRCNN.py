import torch
import torchvision
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
import logging


class CustomKeypointRCNN:
    def __init__(
        self,
        num_keypoints=17,
        weights="KeypointRCNN_ResNet50_FPN_Weights.DEFAULT",
        verbosity="minimal",
    ):
        # Load the pre-trained Keypoint R-CNN model
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights=weights
        )

        # Modify the number of keypoints in the model based on the dataset
        self.model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
            512, num_keypoints
        )

        # Use GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self._init_logger(verbosity)

    def _init_logger(self, verbosity):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        if verbosity == "silent":
            ch.setLevel(logging.CRITICAL)
        elif verbosity == "minimal":
            ch.setLevel(logging.INFO)
        elif verbosity == "verbose":
            ch.setLevel(logging.WARNING)
        elif verbosity == "debug":
            ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(levelname)s - %(message)s")
        ch.setFormatter(formatter)

        self.logger.addHandler(ch)

    def _log(self, message, level):
        if level == "debug":
            self.logger.debug(message)
        elif level == "verbose":
            self.logger.warning(message)  # Using warning to make it visually distinct
        elif level == "minimal":
            self.logger.info(message)

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs,
        optimizer,
        save_path="model_checkpoint.pth",
    ):
        # set model to training mode
        self.model.train()

        # Define the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        best_val_loss = float("inf")  # for checkpointing best model

        for epoch in range(num_epochs):
            self._log(f"Starting epoch {epoch + 1} of {num_epochs}", "verbose")

            epoch_loss = 0.0  # Track cumulative loss for this epoch

            # Training loop
            for batch in train_loader:
                # Move images and targets to device
                images, targets = batch
                images = list(image.to(self.device) for image in images)
                targets = [
                    {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()
                    }
                    for t in targets
                ]

                # Zero out the gradients
                optimizer.zero_grad()

                # Make predictions
                loss_dict = self.model(images, targets)

                # Compute the loss
                loss = sum(loss for loss in loss_dict.values())

                # Perform backpropagation
                loss.backward()
                optimizer.step()

                # Accumulate the batch loss into epoch loss
                epoch_loss += loss.item()

                # Logging
            self._log(f"Batch loss: {loss.item()}", "debug")

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(train_loader)
            self._log(
                f"Epoch {epoch + 1}: Average training loss = {avg_epoch_loss}",
                "debug",
            )

            # Evaluate on validation set
            val_loss = self.validate(val_loader)
            # Return to training mode
            self.model.train()
            self._log(f"Epoch {epoch + 1}: Validation loss = {val_loss}", "debug")

            # Save the model checkpoint if it's the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path + "model.pt")
                self._log(f"Saved best model checkpoint to {save_path}.", "verbose")

            scheduler.step()

    def validate(self, val_loader):
        # Set the model to evaluation mode
        self.model.eval()

        total_val_loss = 0.0  # to accumulate validation loss

        # Turn off gradients
        with torch.no_grad():
            for batch in val_loader:
                # Move images and targets to device
                images, targets = batch
                images = list(image.to(self.device) for image in images)
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                # Make predictions
                loss_dict = self.model(images, targets)

                # Compute the loss
                loss = sum(loss for loss in loss_dict[0].values())

                # Accumulate the validation loss
                total_val_loss += loss.item()

        # Calculate average validation loss
        avg_val_loss = total_val_loss / len(val_loader)

        # Return the model to training mode
        self.model.train()

        return avg_val_loss

    def predict(self, image):
        # Ensure the model is in evaluation mode
        self.model.eval()
        image = image.astype(float) / 255.0

        image_tensor = torch.tensor(image.transpose((2, 0, 1))).float().unsqueeze(0)

        # Move the image tensor to the same device as the model
        image_tensor = image_tensor.to(self.device)

        # Turn off gradients and get model predictions
        with torch.no_grad():
            prediction = self.model(image_tensor)

        # Convert predictions to a more user-friendly format if necessary
        # For example, you might convert tensors to numpy arrays or lists

        return prediction

    def save(self, path):
        # Save the model's state_dict (weights)
        torch.save(self.model.state_dict(), path + "model.pt")

    def load(self, path):
        # Load the model's weights
        self.model.load_state_dict(torch.load(path))
