from controller import Robot
import threading
from vlm import GeminiRobotController, RobotDecision
from devices import Wheels, Camera


class Droid:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.wheels = Wheels(self.robot, max_speed=6.28)
        self.camera = Camera(self.robot, self.timestep)
        self.controller = GeminiRobotController()

        self.current_command = RobotDecision(
            direction="STOP", speed="SLOW", reason="Initializing"
        )
        self.is_processing = False

    def process_frame_async(self, image_b64, frame_num):
        """Background thread for VLM processing"""
        try:
            self.is_processing = True

            result = self.controller.generate_command(image_b64)

            if result:
                self.current_command = result
                print(
                    f"✅ [{frame_num}] {result.direction} {result.speed} - {result.reason}"
                )

        except Exception as e:
            print(f"❌ [{frame_num}] {e}")
            self.current_command = RobotDecision(
                direction="STOP", speed="SLOW", reason="Error"
            )
        finally:
            self.is_processing = False

    def run(self):
        print("🤖 Starting navigation...\n")

        # Warmup
        for _ in range(300):
            self.robot.step(self.timestep)

        frame_count = 0

        try:
            while self.robot.step(self.timestep) != -1:
                frame_count += 1

                # Process every 10th frame
                if not self.is_processing and frame_count % 10 == 0:
                    try:
                        image_b64 = self.camera.get_camera_image()
                        if image_b64:
                            thread = threading.Thread(
                                target=self.process_frame_async,
                                args=(image_b64, frame_count),
                                daemon=True,
                            )
                            thread.start()
                    except Exception as e:
                        print(f"⚠️ [{frame_count}] Camera error: {e}")

                # Execute current command
                self.wheels.execute_command(self.current_command)

                # Status update every 500 frames
                if frame_count % 500 == 0:
                    status = "Processing..." if self.is_processing else "Ready"
                    print(f"[{frame_count}] {status}")

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        finally:
            self.wheels.stop()
            print("👋 Shutdown complete")


if __name__ == "__main__":
    Droid().run()
