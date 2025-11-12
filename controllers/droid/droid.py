from controller import Robot
import warnings
import asyncio
import threading
from vlm import VisionLanguageModel, RobotDecision
from devices import Wheels, Camera

warnings.filterwarnings("ignore")


class Droid:
    def __init__(self, vision_model="riven/smolvlm", navigation_model="gemma3:1b-it-qat"):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.wheels = Wheels(self.robot, max_speed=6.28)
        self.camera = Camera(self.robot, self.timestep)
        self.model = VisionLanguageModel(vision_model=vision_model, navigation_model=navigation_model)

        # Current command to execute (starts with STOP)
        self.current_command = RobotDecision(direction="STOP", speed="SLOW", reason="Initializing")

        # Thread-safe lock for updating command
        self.command_lock = threading.Lock()

        # Track VLM processing state
        self.is_processing = False
        self.vlm_thread = None

        print(f"Using Ollama models: Vision - {vision_model}, Navigation - {navigation_model}")

    def process_vlm_thread(self, image):
        """Background thread to process VLM decision"""
        try:
            self.is_processing = True

            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run async VLM call
            result = loop.run_until_complete(self.model.generate_command(new_image=image))

            if result:
                # Thread-safe update of current command
                with self.command_lock:
                    self.current_command = result

                current_time = self.robot.getTime() * 1000
                print(f"✅ [{current_time/1000:.1f}s] New command: {result.direction} {result.speed}")
                print(f"🧠 VLM reason: '{result.reason}'")

            loop.close()

        except Exception as e:
            print(f"❌ VLM error: {e}")
        finally:
            self.is_processing = False

    def run(self):
        """Main control loop"""
        print("\n🤖 Navigation started!")
        print("📸 Camera: {}x{}\n".format(self.camera.camera.getWidth(), self.camera.camera.getHeight()))

        while self.robot.step(self.timestep) != -1:
            # Get current camera image
            new_image = self.camera.get_camera_image()

            # Start new VLM thread if not already processing
            if not self.is_processing:
                self.vlm_thread = threading.Thread(target=self.process_vlm_thread, args=(new_image,), daemon=True)
                self.vlm_thread.start()

            # Execute current command (thread-safe read)
            with self.command_lock:
                command_to_execute = self.current_command

            self.wheels.execute_command(command_to_execute)


if __name__ == "__main__":
    controller = Droid()
    controller.run()
