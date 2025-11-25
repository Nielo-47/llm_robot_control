from controller import Robot
import numpy as np
from PIL import Image
import base64
import io


def image_to_base64(image, size=(512, 512), quality=100):
    """Convert PIL Image to base64 (resized JPEG)"""
    img = image.copy()
    img = img.resize(size)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=quality)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class RobotDecision:
    """Decisão de navegação do robô (standalone para evitar import circular)"""
    def __init__(self, direction: str, speed: str, reason: str):
        self.direction = direction
        self.speed = speed
        self.reason = reason


class Wheels:
    def __init__(self, robot: Robot, max_speed: float = 6.28):
        self.robot = robot
        self.max_speed = max_speed

        self.last_executed_command = ""
        self.front_left = robot.getDevice("front_left_wheel_joint")
        self.front_right = robot.getDevice("front_right_wheel_joint")
        self.back_left = robot.getDevice("back_left_wheel_joint")
        self.back_right = robot.getDevice("back_right_wheel_joint")

        for wheel in [self.front_left, self.front_right, self.back_left, self.back_right]:
            wheel.setPosition(float("inf"))
            wheel.setVelocity(0.0)

    def execute_command(self, command, force_print=False):
        """Execute movement based on compass direction and speed"""
        speed_mult = self.get_speed_multiplier(command.speed)
        s = self.max_speed * speed_mult

        movements = {
            "N": (s, 0, 0),
            "S": (-s, 0, 0),
            "E": (0, s, 0),
            "W": (0, -s, 0),
            "NE": (s * 0.7, s * 0.7, 0),
            "NW": (s * 0.7, -s * 0.7, 0),
            "SE": (-s * 0.7, s * 0.7, 0),
            "SW": (-s * 0.7, -s * 0.7, 0),
            "STOP": (0, 0, 0),
            "ROAM": (0, 0, 0),
        }

        vx, vy, omega = movements.get(command.direction, (0, 0, 0))
        self.set_mecanum_velocity(vx, vy, omega)

        current_command = f"{command.direction} {command.speed}"

        if force_print or current_command != self.last_executed_command:
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
            print(f"{arrows.get(command.direction, '?')} Executing: {command.direction} at {command.speed} speed")
            self.last_executed_command = current_command

    def get_speed_multiplier(self, speed):
        """Convert speed name to multiplier"""
        return {"SLOW": 0.3, "MEDIUM": 0.6, "FAST": 1.0}.get(speed, 0.6)

    def set_mecanum_velocity(self, vx, vy, omega):
        """Set mecanum wheel velocities for omnidirectional movement
        
        Mecanum wheel configuration:
        - vx: forward/backward velocity (positive = forward)
        - vy: left/right strafe velocity (positive = right)
        - omega: rotation velocity (positive = counter-clockwise)
        
        Standard mecanum formula:
        FL = vx - vy - omega
        FR = vx + vy + omega  
        BL = vx + vy - omega
        BR = vx - vy + omega
        """
        fl = vx - vy - omega
        fr = vx + vy + omega
        bl = vx + vy - omega
        br = vx - vy + omega  # CORRIGIDO: era vx + vy + omega

        self.front_left.setVelocity(fl)
        self.front_right.setVelocity(fr)
        self.back_left.setVelocity(bl)
        self.back_right.setVelocity(br)


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
        
        if image_data is None:
            return None
            
        image = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
        image = image[:, :, [2, 1, 0]]
        pil = Image.fromarray(image)

        return pil  # Retorna PIL Image, não base64
