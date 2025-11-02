"""
Webots Robot Controller with BLIP VLM
Lightweight and fully compatible with Python 3.9
"""

from controller import Robot, Camera
import numpy as np
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import time


class BLIPRobotController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Initialize camera
        self.camera = self.robot.getDevice("head_camera")
        self.camera.enable(self.timestep)

        # Initialize BLIP model (smaller and faster)
        print("Loading BLIP model...")
        model_name = "Salesforce/blip-image-captioning-base"  # ~990MB
        # Alternative: "Salesforce/blip-image-captioning-large" for better quality

        # Detect device
        if torch.backends.mps.is_available():
            self.device = "mps"  # Apple Silicon GPU
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"Using device: {self.device}")

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
        ).to(self.device)

        self.query_interval = 100  # Query every N milliseconds
        self.last_query_time = 0

    def get_camera_image(self):
        """Convert Webots camera image to PIL Image"""
        width = self.camera.getWidth()
        height = self.camera.getHeight()

        # Get image data from camera
        image_data = self.camera.getImage()

        # Convert to numpy array
        image = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))

        # Convert BGRA to RGB
        image = image[:, :, [2, 1, 0]]

        # Convert to PIL Image
        return Image.fromarray(image)

    def caption_image(self, image):
        """Generate caption for image (unconditional)"""
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=50)

        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

    def answer_question(self, image, question):
        """Answer question about image (conditional)"""
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=50)

        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer

    def run(self):
        """Main control loop"""
        print("Robot controller started. Press Ctrl+C to stop.")

        while self.robot.step(self.timestep) != -1:
            current_time = self.robot.getTime() * 1000  # Convert to ms

            # Query VLM at specified intervals
            if current_time - self.last_query_time >= self.query_interval:
                try:
                    # Get current camera image
                    image = self.get_camera_image()

                    # Get caption
                    print("\nGenerating scene description...")
                    start_time = time.time()

                    # Option 1: Simple caption
                    caption = self.caption_image(image)

                    # Option 2: Question answering
                    obstacle_check = self.answer_question(image, "Is there an obstacle in front?")

                    inference_time = time.time() - start_time

                    print(f"Caption ({inference_time:.2f}s): {caption}")
                    print(f"Obstacle check: {obstacle_check}")

                    # Use the description for robot behavior
                    self.process_scene_understanding(caption, obstacle_check)

                    self.last_query_time = current_time

                except Exception as e:
                    print(f"Error querying VLM: {e}")
                    import traceback

                    traceback.print_exc()

    def process_scene_understanding(self, caption, obstacle_info):
        """Process VLM output and control robot accordingly"""
        caption_lower = caption.lower()
        obstacle_lower = obstacle_info.lower()

        # Simple decision logic
        if "yes" in obstacle_lower or "obstacle" in caption_lower:
            print("Action: Obstacle detected - avoiding")
            # Add motor control here

        else:
            print("Action: Path appears clear")
            # Add motor control here


if __name__ == "__main__":
    controller = BLIPRobotController()
    controller.run()
