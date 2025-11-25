from controller import Robot
from vlm import RobotDecision
import numpy as np
from PIL import Image
import base64
import io
import math

from controller import Robot
from vlm import RobotDecision
import math


class Wheels:
    def __init__(self, robot: Robot, max_speed: float = 6.28):
        self.robot = robot
        self.max_speed = max_speed
        self.last_executed_command = None

        # Robotino3 has 3 omnidirectional wheels at 120° angles
        self.wheel_0 = robot.getDevice("wheel0_joint")  # Front
        self.wheel_1 = robot.getDevice("wheel1_joint")  # Back-left
        self.wheel_2 = robot.getDevice("wheel2_joint")  # Back-right

        for wheel in [self.wheel_0, self.wheel_1, self.wheel_2]:
            wheel.setPosition(float("inf"))
            wheel.setVelocity(0.0)

    def execute_command(self, command: RobotDecision, force_print=False):
        """Execute movement based on compass direction and speed"""
        speed_mult = self.get_speed_multiplier(command.speed)
        s = self.max_speed * speed_mult

        # Convert compass directions to vx, vy, omega
        # N = camera forward, E = camera right, W = camera left, S = camera backward
        direction_vectors = {
            "N": (s, 0, 0),  # Forward (toward camera view)
            "S": (-s, 0, 0),  # Backward (away from camera view)
            "E": (0, -s, 0),  # Right (strafe right while facing forward)
            "W": (0, s, 0),  # Left (strafe left while facing forward)
            "NE": (s * 0.7, -s * 0.7, 0),  # Forward-right diagonal
            "NW": (s * 0.7, s * 0.7, 0),  # Forward-left diagonal
            "SE": (-s * 0.7, -s * 0.7, 0),  # Backward-right diagonal
            "SW": (-s * 0.7, s * 0.7, 0),  # Backward-left diagonal
            "STOP": (0, 0, 0),
            "ROAM": (s * 0.3, 0, 0),  # Slow forward exploration
        }

        vx, vy, omega = direction_vectors.get(command.direction, (0, 0, 0))
        self.set_omnidirectional_velocity(vx, vy, omega)

        # Only print if command changed or force_print is True
        if (
            force_print
            or self.last_executed_command != command.direction + command.speed
        ):
            arrows = {
                "N": "↑",
                "S": "↓",
                "E": "→",
                "W": "←",
                "NE": "↗",
                "NW": "↖",
                "SE": "↘",
                "SW": "↙",
                "STOP": "⏸",
                "ROAM": "◉",
            }
            print(
                f"{arrows.get(command.direction, '?')} {command.direction} at {command.speed}"
            )
            self.last_executed_command = command.direction + command.speed

    def get_speed_multiplier(self, speed):
        """Convert speed name to multiplier"""
        return {"SLOW": 0.3, "MEDIUM": 0.6, "FAST": 1.0, "STOP": 0.0}.get(speed, 0.6)

    def set_omnidirectional_velocity(self, vx, vy, omega):
        """
        Set omnidirectional wheel velocities for Robotino3.
        Wheels are arranged at 120° angles:
        - Wheel 0: 0° (front)
        - Wheel 1: 120° (back-left)
        - Wheel 2: 240° (back-right)
        """
        # Wheel positions in radians
        wheel_angles = [0, 2 * math.pi / 3, 4 * math.pi / 3]
        wheel_radius = 0.05  # Approximate radius in meters

        velocities = []
        for angle in wheel_angles:
            # Project velocity onto wheel direction
            v_wheel = vx * math.cos(angle) + vy * math.sin(angle)
            # Add rotational component (perpendicular to wheel)
            v_wheel += omega * wheel_radius
            velocities.append(v_wheel)

        self.wheel_0.setVelocity(velocities[0])
        self.wheel_1.setVelocity(velocities[1])
        self.wheel_2.setVelocity(velocities[2])

    def stop(self):
        """Emergency stop"""
        self.wheel_0.setVelocity(0)
        self.wheel_1.setVelocity(0)
        self.wheel_2.setVelocity(0)


class Camera:
    def __init__(self, robot: Robot, timestep: int):
        self.robot = robot
        self.timestep = timestep

        self.camera = self.robot.getDevice("camera")
        if self.camera is None:
            raise Exception("RGB camera not found!")
        self.camera.enable(self.timestep)

    def get_camera_image(self):
        """Convert Webots RGB camera to PIL Image"""
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        image_data = self.camera.getImage()
        image = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
        image = image[:, :, [2, 1, 0]]
        pil = Image.fromarray(image)

        return self.image_to_base64(pil)

    def image_to_base64(self, image, size=(512, 512), quality=100):
        """Convert PIL Image to base64 (resized JPEG)"""
        img = image.copy()
        img = img.resize(size)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=quality)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
