from controller import Robot
from vlm import RobotDecision
import numpy as np
from PIL import Image
import base64
import io
import math

WHEEL_RADIUS = 0.063
DIST_TO_CENTER = 0.1826

import math

WHEEL_RADIUS = 0.063
DIST_TO_CENTER = 0.1826


class Wheels:
    def __init__(self, robot, max_speed=6.28):
        self.robot = robot
        self.max_speed = max_speed
        self.last_executed_command = None

        self.wheel_0 = robot.getDevice("wheel0_joint")
        self.wheel_1 = robot.getDevice("wheel1_joint")
        self.wheel_2 = robot.getDevice("wheel2_joint")

        for w in (self.wheel_0, self.wheel_1, self.wheel_2):
            w.setPosition(float("inf"))
            w.setVelocity(0.0)

        self.actual = [0.0, 0.0, 0.0]  # vx, vy, omega
        self.target = [0.0, 0.0, 0.0]
        self.max_accel = [10.0, 6.0, 20.0]

    def execute_command(self, command, force_print=False):
        speed_mult = self.get_speed_multiplier(command.speed)
        s = self.max_speed * speed_mult

        # --- NEW SIMPLIFIED MAPPING ---
        direction_vectors = {
            "FORWARD": (s, 0, 0),
            "BACKWARD": (-s, 0, 0),
            "LEFT": (0, s, 0),
            "RIGHT": (0, -s, 0),
            "STOP": (0, 0, 0),
        }

        vx, vy, omega = direction_vectors.get(command.direction, (0, 0, 0))

        self.set_target_speeds(vx, vy, omega)
        self.accelerate()

        if (
            force_print
            or self.last_executed_command != command.direction + command.speed
        ):
            self.last_executed_command = command.direction + command.speed
            print(f"{command.direction} at {command.speed}")

    def get_speed_multiplier(self, speed):
        return {"SLOW": 0.3, "MEDIUM": 0.6, "FAST": 1.0, "STOP": 0.0}.get(speed, 0.6)

    # -----------------------------
    # C-LIKE MOVEMENT LOGIC BELOW
    # -----------------------------

    def set_target_speeds(self, vx, vy, omega):
        self.target[0] = vx
        self.target[1] = vy
        self.target[2] = omega

    def accelerate(self):
        timestep = self.robot.getBasicTimeStep() / 1000.0

        maxSteps = 1
        for i in range(3):
            diff = abs(self.target[i] - self.actual[i])
            steps = diff / (self.max_accel[i] * timestep)
            if steps > maxSteps:
                maxSteps = steps

        for i in range(3):
            self.actual[i] += (self.target[i] - self.actual[i]) / maxSteps

        self.apply_speeds(*self.actual)

    def apply_speeds(self, vx, vy, omega):
        vx /= WHEEL_RADIUS
        vy /= WHEEL_RADIUS
        omega *= DIST_TO_CENTER / WHEEL_RADIUS

        w0 = vy - omega
        w1 = -math.sqrt(0.75) * vx - 0.5 * vy - omega
        w2 = math.sqrt(0.75) * vx - 0.5 * vy - omega

        self.wheel_0.setVelocity(w0)
        self.wheel_1.setVelocity(w1)
        self.wheel_2.setVelocity(w2)

    def stop(self):
        self.set_target_speeds(0, 0, 0)
        self.accelerate()


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
