import asyncio
from pydantic import BaseModel
from typing import Literal, TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
import torch
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from langchain_google_genai import ChatGoogleGenerativeAI
import operator
from config import (
    DEVICE,
    VISION_MODEL_NAME,
    NAVIGATION_MODEL_NAME,
    GOOGLE_API_KEY,
    VISION_PROMPT,
    NAVIGATION_PROMPT,
)


class RobotDecision(BaseModel):
    direction: Literal["N", "S", "E", "W", "NE", "NW", "SE", "SW", "STOP", "ROAM"]
    speed: Literal["SLOW", "MEDIUM", "FAST"]
    reason: str


class RobotState(TypedDict):
    scene_descriptions: Annotated[List[str], operator.add]
    decisions: Annotated[List[RobotDecision], operator.add]
    current_image: str
    current_description: str
    api_decision: RobotDecision | None


class VisionLanguageModel:
    def __init__(
        self,
        vision_model: str = VISION_MODEL_NAME,
        navigation_model: str = NAVIGATION_MODEL_NAME,
    ):
        print("🔧 Initializing vision model...")
        self.vision_processor = AutoProcessor.from_pretrained(vision_model)
        self.vision_model = AutoModelForVision2Seq.from_pretrained(
            vision_model, torch_dtype=torch.float32
        ).to(DEVICE)

        print("🔧 Initializing navigation agent (Gemini)...")
        self.nav_model = ChatGoogleGenerativeAI(
            model=navigation_model,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
        ).with_structured_output(RobotDecision)

        self.graph = self.build_graph()
        print("✅ Models initialized.")

    def vision_node(self, state: RobotState) -> dict:
        """Stage 1: Local VLM describes the scene"""
        print("🔍 Analyzing scene...")

        image_b64 = state["current_image"]
        raw = base64.b64decode(image_b64)
        img = Image.open(BytesIO(raw)).convert("RGB")
        img.thumbnail((320, 320))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ]

        inputs = self.vision_processor(
            text=self.vision_processor.apply_chat_template(
                messages, add_generation_prompt=True
            ),
            images=[img],
            return_tensors="pt",
        ).to(DEVICE)

        generated_ids = self.vision_model.generate(**inputs, max_new_tokens=512)
        description = self.vision_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        if not description:
            description = "No notable obstacles detected; area appears clear."

        print(f"📝 Scene: {description}")
        return {"scene_descriptions": [description], "current_description": description}

    async def api_node(self, state: RobotState) -> dict:
        """Stage 2: Gemini makes navigation decision"""
        print("🧠 Gemini navigation agent deciding...")

        description = state["current_description"]
        decision_history = "No previous decisions"
        if state["decisions"]:
            recent = state["decisions"][-3:]
            decision_history = "\n".join(
                [f"- {d.direction} at {d.speed}: {d.reason}" for d in recent]
            )

        prompt = NAVIGATION_PROMPT.format(
            decisions=decision_history, scene_description=description
        )

        decision = await self.nav_model.ainvoke(prompt)
        print(f"✅ Decision: {decision.direction} at {decision.speed}")
        return {"api_decision": decision, "decisions": [decision]}

    def build_graph(self):
        """Build LangGraph workflow"""
        graph = StateGraph(RobotState)

        graph.add_node("vision", self.vision_node)
        graph.add_node("api", self.api_node)

        graph.add_edge(START, "vision")
        graph.add_edge("vision", "api")
        graph.add_edge("api", END)

        return graph.compile()

    async def generate_command(self, new_image: str) -> RobotDecision:
        """Main loop: process image through graph"""

        state = {
            "scene_descriptions": [],
            "decisions": [],
            "current_image": new_image,
            "current_description": "",
            "api_decision": None,
        }

        result = await self.graph.ainvoke(state)
        return result["decisions"][-1]


async def main():
    print("\n🤖 Initializing LangGraph Robot Control System")
    print("Stage 1: Local VLM (Scene Analysis)")
    print("Stage 2: Gemini (Strategic Navigation)\n")

    model = VisionLanguageModel()

    for i in range(3):
        print(f"\n{'='*60}")
        print(f"ITERATION {i+1}")
        print("=" * 60)

        decision = await model.generate_command(new_image="fake_base64_image_data")

        print(f"\n📍 FINAL COMMAND: {decision.direction} at {decision.speed}")
        print(f"   Reason: {decision.reason}")

        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
