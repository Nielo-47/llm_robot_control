from controller import Robot
import warnings
import asyncio
import threading
from vlm import VisionLanguageModel, RobotDecision
from devices import Wheels, Camera

warnings.filterwarnings("ignore")


class Droid:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.wheels = Wheels(self.robot, max_speed=6.28)
        self.camera = Camera(self.robot, self.timestep)

        print("🔧 Initializing VLM models...")
        self.model = VisionLanguageModel()

        # Current command to execute (starts with STOP)
        self.current_command = RobotDecision(
            direction="STOP", speed="SLOW", reason="Initializing"
        )

        # Thread-safe lock for updating command
        self.command_lock = threading.Lock()

        # Track VLM processing state
        self.is_processing = False
        self.vlm_thread = None
        self.last_update_time = 0

        print("✅ Droid initialized")

    def process_vlm_thread(self, image, capture_time):
        """Background thread to process VLM decision"""
        try:
            self.is_processing = True

            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run async VLM call
            result = loop.run_until_complete(
                self.model.generate_command(new_image=image)
            )

            if result:
                # Thread-safe update of current command
                with self.command_lock:
                    self.current_command = result
                    self.last_update_time = self.robot.getTime()

                print(
                    f"✅ [{capture_time:.1f}s] New command: {result.direction} {result.speed}"
                )
                print(f"🧠 Reason: '{result.reason}'")

            loop.close()

        except Exception as e:
            print(f"❌ VLM error: {e}")
            # On error, default to safe behavior
            with self.command_lock:
                self.current_command = RobotDecision(
                    direction="STOP", speed="SLOW", reason="VLM error - stopping"
                )
        finally:
            self.is_processing = False

    def run(self):
        """Main control loop"""
        print("\n🤖 Navigation started!")
        print(
            f"📸 Camera: {self.camera.camera.getWidth()}x{self.camera.camera.getHeight()}\n"
        )

        iteration = 0

        try:
            while self.robot.step(self.timestep) != -1:
                current_time = self.robot.getTime()

                # Get current camera image
                new_image = self.camera.get_camera_image()

                # Start new VLM thread if not already processing
                if not self.is_processing:
                    # Clean up old thread if it exists
                    if self.vlm_thread is not None and self.vlm_thread.is_alive():
                        self.vlm_thread.join(timeout=0.1)

                    self.vlm_thread = threading.Thread(
                        target=self.process_vlm_thread,
                        args=(new_image, current_time),
                        daemon=True,
                    )
                    self.vlm_thread.start()

                # Execute current command (thread-safe read)
                with self.command_lock:
                    command_to_execute = self.current_command
                    time_since_update = current_time - self.last_update_time

                # Safety: Stop if VLM hasn't updated in too long (30 seconds)
                if time_since_update > 30.0 and self.last_update_time > 0:
                    print(f"⚠️ VLM timeout ({time_since_update:.1f}s) - stopping")
                    command_to_execute = RobotDecision(
                        direction="STOP", speed="SLOW", reason="VLM timeout"
                    )

                self.wheels.execute_command(command_to_execute)

                iteration += 1

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        finally:
            # Cleanup
            print("🧹 Cleaning up...")
            self.wheels.stop()

            # Wait for VLM thread to finish (with timeout)
            if self.vlm_thread is not None and self.vlm_thread.is_alive():
                print("⏳ Waiting for VLM thread...")
                self.vlm_thread.join(timeout=5.0)

            print("👋 Shutdown complete")


if __name__ == "__main__":
    controller = Droid()
    controller.run()
