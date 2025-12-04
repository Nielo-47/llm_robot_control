from controller import Robot
from vlm import RobotDecision
import numpy as np
from PIL import Image
import base64
import io
import math

WHEEL_RADIUS = 0.063
DIST_TO_CENTER = 0.1826


class Wheels:
    def __init__(self, robot, max_speed=6.28):
        self.robot = robot
        self.max_speed = max_speed
        self.last_executed_command = None

        self.imu = None
        try:
            self.imu = robot.getDevice("inertial unit")
            if self.imu:
                self.imu.enable(int(robot.getBasicTimeStep()))
        except Exception as e:
            print("Error enabling IMU:", e)
            self.imu = None

        self.wheel_0 = robot.getDevice("wheel0_joint")
        self.wheel_1 = robot.getDevice("wheel1_joint")
        self.wheel_2 = robot.getDevice("wheel2_joint")

        for w in (self.wheel_0, self.wheel_1, self.wheel_2):
            w.setPosition(float("inf"))
            w.setVelocity(0.0)

        self.actual = [0.0, 0.0, 0.0]
        self.target = [0.0, 0.0, 0.0]
        self.max_accel = [10.0, 6.0, 20.0]

        self.angle_tolerance_rad = math.radians(2.0)
        self.angular_gain = 1.2
        self.max_omega = 2.0

        self.state = "IDLE"  # IDLE, TURNING, MOVING
        self.target_angle_rad = 0.0
        self.target_distance = 0.0
        self.distance_traveled = 0.0

    def execute_command(self, command: RobotDecision, force_print=False):
        """
        Execute vector-based navigation command.
        Sequence: STOP -> TURN relative -> MOVE forward distance
        """
        angle_deg = getattr(command, "angle", 0.0)
        distance = getattr(command, "distance", 0.0)

        # If no movement, just stop
        if distance == 0.0:
            self.stop()
            self.state = "IDLE"
            if force_print or self.last_executed_command != "STOP":
                self.last_executed_command = "STOP"
                print(f"Command -> STOP (target reached)")
            return

        key = f"{angle_deg:.1f}:{distance:.2f}"
        if self.last_executed_command != key:
            self.stop()
            self.state = "TURNING"

            # -----------------------------
            # 🔥 NEW: interpret angle as RELATIVE
            # -----------------------------
            current_yaw = self.get_yaw()
            self.target_angle_rad = current_yaw + math.radians(angle_deg)

            # normalize into [-pi, pi]
            while self.target_angle_rad > math.pi:
                self.target_angle_rad -= 2 * math.pi
            while self.target_angle_rad < -math.pi:
                self.target_angle_rad += 2 * math.pi

            self.target_distance = distance
            self.distance_traveled = 0.0
            self.last_executed_command = key

            if force_print:
                print(
                    f"New Command -> relative turn +{angle_deg:.1f}° then {distance:.2f}m"
                )

    def update(self, force_print=False):
        """
        Call this every simulation step to execute the current command state machine.
        """
        if self.state == "IDLE":
            return

        elif self.state == "TURNING":
            current_yaw_rad = self.get_yaw()
            diff_rad = self._angle_difference(self.target_angle_rad, current_yaw_rad)

            if abs(diff_rad) > self.angle_tolerance_rad:
                omega_cmd = self.angular_gain * diff_rad
                omega = max(-self.max_omega, min(self.max_omega, omega_cmd))
                self.set_target_speeds(0.0, 0.0, omega)
                self.accelerate()
            else:
                self.stop()
                self.state = "MOVING"
                if force_print:
                    print(
                        f"Turn complete, now moving {self.target_distance:.2f}m forward"
                    )

        elif self.state == "MOVING":
            if self.distance_traveled >= self.target_distance:
                self.stop()
                self.state = "IDLE"
                if force_print:
                    print(f"Movement complete: {self.target_distance:.2f}m")
            else:
                timestep = self.robot.getBasicTimeStep() / 1000.0
                self.distance_traveled += self.max_speed * timestep
                self.set_target_speeds(self.max_speed, 0.0, 0.0)
                self.accelerate()

    def _angle_difference(self, target_rad, current_rad):
        """Calculate shortest angular difference in radians (-pi to pi)"""
        diff = target_rad - current_rad
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def get_yaw(self):
        if not self.imu:
            return 0.0
        try:
            rpy = self.imu.getRollPitchYaw()
            if rpy and len(rpy) >= 3:
                return float(rpy[2])
        except Exception:
            pass
        return 0.0

    def set_target_speeds(self, vx, vy, omega):
        self.target[0] = vx
        self.target[1] = vy
        self.target[2] = omega

    def accelerate(self):
        timestep = self.robot.getBasicTimeStep() / 1000.0
        maxSteps = 1.0

        for i in range(3):
            diff = abs(self.target[i] - self.actual[i])
            steps = (
                diff / (self.max_accel[i] * timestep)
                if self.max_accel[i] * timestep > 0
                else 1.0
            )
            if steps > maxSteps:
                maxSteps = steps

        for i in range(3):
            self.actual[i] += (self.target[i] - self.actual[i]) / maxSteps

        self.apply_speeds(*self.actual)

    def apply_speeds(self, vx, vy, omega):
        wheel_vx = vx / WHEEL_RADIUS
        wheel_vy = vy / WHEEL_RADIUS
        wheel_omega = omega * DIST_TO_CENTER / WHEEL_RADIUS

        w0 = wheel_vy - wheel_omega
        w1 = -math.sqrt(0.75) * wheel_vx - 0.5 * wheel_vy - wheel_omega
        w2 = math.sqrt(0.75) * wheel_vx - 0.5 * wheel_vy - wheel_omega

        try:
            self.wheel_0.setVelocity(w0)
            self.wheel_1.setVelocity(w1)
            self.wheel_2.setVelocity(w2)
        except Exception as e:
            print("Error setting wheel velocities:", e)

    def stop(self):
        self.set_target_speeds(0.0, 0.0, 0.0)
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
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        image_data = self.camera.getImage()
        image = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
        image = image[:, :, [2, 1, 0]]
        pil = Image.fromarray(image)
        return self.image_to_base64(pil)

    def image_to_base64(self, image, size=(512, 512), quality=100):
        img = image.copy()
        img = img.resize(size)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=quality)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
