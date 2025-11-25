import asyncio
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from typing import Literal, TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import base64
from io import BytesIO
from PIL import Image

DEVICE = "cpu"


VISION_PROMPT = """
Analyze this camera image from a robot's perspective. Describe what you see in terms of:
- Obstacles or objects ahead
- Open paths or clear areas
- Potential hazards
- Overall environment
Be concise and focus on navigation-relevant details.
"""

NAVIGATION_PROMPT = """You are a robot navigation system. Your goal is to find a green plant and navigate to it without hitting any obstacles. Based on the visual description, decide the next movement:

Previous Commands: {decisions}
Visual Description: {scene_description}
Total images analyzed: {num_images}

Provide a navigation decision with direction, speed, and reasoning based on the scene."""


class CameraFeed(BaseModel):
    new_image: str = Field(description="Base64 encoded camera image")


class RobotDecision(BaseModel):
    """Structured response model for robot navigation decisions"""

    direction: Literal["N", "S", "E", "W", "NE", "NW", "SE", "SW", "STOP", "ROAM"] = Field(
        description="Navigation direction"
    )
    speed: Literal["SLOW", "MEDIUM", "FAST"] = Field(description="Movement speed")
    reason: str = Field(description="Short explanation for the decision (one sentence)")


class RobotState(TypedDict):
    images: List[str]  # List of base64 images
    scene_descriptions: List[str]  # Descriptions from SmolVLM
    decisions: List[RobotDecision]  # Navigation decisions from Gemma3
    current_image: str  # Current image being processed
    current_description: str  # Current scene description
    custom_vision_prompt: Optional[str]  # Custom prompt for vision model
    custom_navigation_prompt: Optional[str]  # Custom prompt for navigation model


class VisionLanguageModel:
    def __init__(
        self,
        vision_model: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
        navigation_model: str = "gemma3:1b-it-qat",
        max_history: int = 5,
    ):
        print("🔧 Initializing SmolVLM-256M-Instruct vision model...")
        self.vision_processor = AutoProcessor.from_pretrained(vision_model)
        self.vision_model = AutoModelForVision2Seq.from_pretrained(
            vision_model,
            torch_dtype=torch.float32,
        ).to(DEVICE)
        print(f"   ✓ Vision model loaded on {DEVICE}")

        # --- Navigation model still via Ollama ---
        self.navigation_model = ChatOllama(
            model=navigation_model,
            ollama_url="http://localhost:11434",
            temperature=0.3,  # Aumentado de 0 para dar mais criatividade
            num_ctx=512,
        ).with_structured_output(RobotDecision)

        self.max_history = max_history
        self.state = {
            "images": [],
            "scene_descriptions": [],
            "decisions": [],
            "current_image": "",
            "current_description": "",
            "custom_vision_prompt": None,
            "custom_navigation_prompt": None,
        }
        self.graph = self.generate_graph()
        print("✅ VisionLanguageModel initialized.")

    def vision_node(self, state: RobotState) -> RobotState:
        """Stage 1: use SmolVLM-256M-Instruct to describe the image."""
        current_image_b64 = state["current_image"]
        print("🔍 SmolVLM analyzing scene...")

        # Decode base64 image
        image = None
        raw = base64.b64decode(current_image_b64)
        img = Image.open(BytesIO(raw)).convert("RGB")
        # Optional: downscale for speed
        img.thumbnail((320, 320))
        image = img

        # Use custom prompt if provided, otherwise use default
        vision_prompt = state.get("custom_vision_prompt") or VISION_PROMPT

        # Build message prompt + image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": vision_prompt},
                ],
            }
        ]

        # Prepare inputs for Transformer
        inputs = self.vision_processor(
            text=self.vision_processor.apply_chat_template(messages, add_generation_prompt=True),
            images=[image],
            return_tensors="pt",
        ).to(DEVICE)

        # Generate output text
        generated_ids = self.vision_model.generate(**inputs, max_new_tokens=128)  # Aumentado de 64
        generated_text = self.vision_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        if not generated_text:
            generated_text = "No notable obstacles detected; area appears clear."

        print("📝 Scene:", generated_text)

        updated_descriptions = state["scene_descriptions"] + [generated_text]
        return {"scene_descriptions": updated_descriptions, "current_description": generated_text}

    def navigation_node(self, state: RobotState) -> RobotState:
        """Stage 2: Gemma3 makes navigation decision based on scene description"""

        print("🧠 Gemma3 deciding navigation...")

        # Format decision history
        decision_history = "No previous decisions"
        if state["decisions"]:
            recent_decisions = state["decisions"][-3:]
            decision_history = "\n".join([f"- {d.direction} at {d.speed}: {d.reason}" for d in recent_decisions])

        # Use custom prompt if provided, otherwise use default
        navigation_prompt_template = state.get("custom_navigation_prompt") or NAVIGATION_PROMPT

        # Create navigation prompt with scene description
        prompt = navigation_prompt_template.format(
            decisions=decision_history, 
            scene_description=state["current_description"], 
            num_images=len(state["images"])
        )

        # Get structured navigation decision
        decision = self.navigation_model.invoke(prompt)

        print(f"✅ Decision: {decision.direction} at {decision.speed}")

        # Update state with new decision
        updated_decisions = state["decisions"] + [decision]

        return {"decisions": updated_decisions}

    def generate_graph(self):
        """Generate two-stage LangGraph workflow"""
        graph = StateGraph(RobotState)

        # Add both nodes
        graph.add_node("vision", self.vision_node)
        graph.add_node("navigation", self.navigation_node)

        # Define flow: START -> vision -> navigation -> END
        graph.add_edge(START, "vision")
        graph.add_edge("vision", "navigation")
        graph.add_edge("navigation", END)

        return graph.compile()

    async def generate_command(
        self, 
        new_image: str, 
        custom_vision_prompt: Optional[str] = None,
        custom_navigation_prompt: Optional[str] = None
    ) -> RobotDecision:
        """Generate a movement command using two-stage processing
        
        Args:
            new_image: Base64 encoded image
            custom_vision_prompt: Optional custom prompt for vision analysis
            custom_navigation_prompt: Optional custom prompt for navigation decision
        """

        # Add new image to persistent state
        self.state["images"].append(new_image)
        self.state["current_image"] = new_image
        self.state["custom_vision_prompt"] = custom_vision_prompt
        self.state["custom_navigation_prompt"] = custom_navigation_prompt

        # Trim history if needed
        if len(self.state["images"]) > self.max_history:
            self.state["images"] = self.state["images"][-self.max_history :]
        if len(self.state["scene_descriptions"]) > self.max_history:
            self.state["scene_descriptions"] = self.state["scene_descriptions"][-self.max_history :]
        if len(self.state["decisions"]) > self.max_history:
            self.state["decisions"] = self.state["decisions"][-self.max_history :]

        # Invoke graph with current persistent state
        result = await self.graph.ainvoke(self.state)

        # Update persistent state
        self.state["scene_descriptions"] = result["scene_descriptions"]
        self.state["decisions"] = result["decisions"]

        # Return the latest decision
        return result["decisions"][-1]

    def reset_state(self):
        """Clear all history"""
        self.state = {
            "images": [],
            "scene_descriptions": [],
            "decisions": [],
            "current_image": "",
            "current_description": "",
            "custom_vision_prompt": None,
            "custom_navigation_prompt": None,
        }


async def main():
    """Test the two-stage pipeline"""
    print("\n🤖 Initializing Two-Stage VLM Pipeline")
    print("Stage 1: SmolVLM (Vision Analysis)")
    print("Stage 2: Gemma3 (Navigation Decision)\n")

    model = VisionLanguageModel()

    # Simulate processing with fake image
    print("=" * 60)
    decision = await model.generate_command(new_image="fake_base64_image_data")
    print("=" * 60)
    print(f"\n📍 FINAL DECISION:")
    print(f"   Direction: {decision.direction}")
    print(f"   Speed: {decision.speed}")
    print(f"   Reason: {decision.reason}")
    print(f"\n📊 History:")
    print(f"   Images processed: {len(model.state['images'])}")
    print(f"   Scene descriptions: {len(model.state['scene_descriptions'])}")
    print(f"   Total decisions: {len(model.state['decisions'])}")


if __name__ == "__main__":
    asyncio.run(main())