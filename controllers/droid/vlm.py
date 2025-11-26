from pydantic import BaseModel
from typing import Literal
import base64
from io import BytesIO
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from config import NAVIGATION_MODEL_NAME, GOOGLE_API_KEY, NAVIGATION_PROMPT


class RobotDecision(BaseModel):
    direction: Literal["FORWARD", "LEFT", "RIGHT", "BACWARDS", "STOP"]
    speed: Literal["SLOW", "MEDIUM", "FAST", "STOP"]
    reason: str


class GeminiRobotController:
    def __init__(self, model_name: str = NAVIGATION_MODEL_NAME):
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.5,
        ).with_structured_output(RobotDecision)
        self.decisions = []

    def generate_command(self, image_b64: str) -> RobotDecision:
        """Process image and return navigation decision (single API call)"""
        # Decode and prepare image
        raw = base64.b64decode(image_b64)
        img = Image.open(BytesIO(raw)).convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Build prompt with history
        history = (
            "\n".join(
                [
                    f"- {d.direction} at {d.speed}: {d.reason}"
                    for d in self.decisions[-3:]
                ]
            )
            or "No previous decisions"
        )

        prompt = NAVIGATION_PROMPT.format(decisions=history)

        # Single API call with image and prompt
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
    print("\n🤖 Gemini Robot Control System\n")
    controller = GeminiRobotController()

    for i in range(3):
        print(f"\n{'='*50}\nITERATION {i+1}\n{'='*50}")
        decision = await controller.generate_command("fake_base64_image_data")
        print(f"📍 COMMAND: {decision.direction} at {decision.speed}")
        print(f"   Reason: {decision.reason}\n")
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
