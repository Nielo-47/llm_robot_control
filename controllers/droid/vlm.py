from pydantic import BaseModel
from typing import Optional
import base64
from io import BytesIO
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from config import NAVIGATION_MODEL_NAME, GOOGLE_API_KEY, NAVIGATION_PROMPT
import asyncio


class RobotDecision(BaseModel):
    angle: float  # Degrees from -180 to 180
    distance: float  # Meters from 0.0 to 5.0
    reason: str


class GeminiRobotController:
    def __init__(self, model_name: str = NAVIGATION_MODEL_NAME):
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1,
        ).with_structured_output(RobotDecision)
        self.decisions = []

    def generate_command(self, image_b64: str) -> RobotDecision:
        """Process image and return navigation decision as vector (angle, distance)"""
        raw = base64.b64decode(image_b64)
        img = Image.open(BytesIO(raw)).convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        history = (
            "\n".join(
                [
                    f"- Turn {d.angle:.1f}° and move {d.distance:.2f}m: {d.reason}"
                    for d in self.decisions[-3:]
                ]
            )
            or "No previous decisions"
        )
        prompt = NAVIGATION_PROMPT.format(decisions=history)

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{img_str}"},
            ]
        )

        decision = self.model.invoke([message])
        self.decisions.append(decision)
        return decision


async def main():
    print("\n🤖 Gemini Robot Control System (Vector Navigation)\n")
    print("Behavior: STOP → TURN to angle → MOVE forward at full speed → STOP\n")
    controller = GeminiRobotController()

    for i in range(3):
        print(f"\n{'='*50}\nITERATION {i+1}\n{'='*50}")
        decision = controller.generate_command("fake_base64_image_data")

        if decision.distance == 0.0:
            print(f"🛑 STOP: Target reached!")
            print(f"   Reason: {decision.reason}")
        else:
            print(f"📍 NEW COMMAND:")
            print(f"   1. Stop completely")
            print(f"   2. Turn to {decision.angle:.1f}°")
            print(f"   3. Move forward {decision.distance:.2f}m at full speed")
            print(f"   4. Stop completely")
            print(f"   Reason: {decision.reason}")

        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
