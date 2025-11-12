import asyncio
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from typing import Literal, TypedDict, List
from langgraph.graph import StateGraph, START, END

VISION_PROMPT = """Analyze this camera image from a robot's perspective. Describe what you see in terms of:
- Obstacles or objects ahead
- Open paths or clear areas
- Potential hazards
- Overall environment

Be concise and focus on navigation-relevant details."""

NAVIGATION_PROMPT = """You are a robot navigation system. Based on the visual description, decide the next movement:

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


class VisionLanguageModel:
    def __init__(self, vision_model: str, navigation_model: str, max_history: int = 5):
        # SmolVLM for visual understanding
        self.vision_model = ChatOllama(
            model=vision_model,
            ollama_url="http://localhost:11434",
            temperature=0,
            num_predict=150,
        )

        # Gemma3 for navigation decisions
        self.navigation_model = ChatOllama(
            model=navigation_model,
            ollama_url="http://localhost:11434",
            temperature=0,
            num_predict=150,
        ).with_structured_output(RobotDecision)

        self.max_history = max_history

        # Initialize persistent state
        self.state = {
            "images": [],
            "scene_descriptions": [],
            "decisions": [],
            "current_image": "",
            "current_description": "",
        }

        self.graph = self.generate_graph()

    def vision_node(self, state: RobotState) -> RobotState:
        """Stage 1: SmolVLM analyzes the image and provides scene description"""

        current_image = state["current_image"]

        print(f"🔍 {self.vision_model.model} analyzing scene...")

        # Build message with image for SmolVLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{current_image}"},
                ],
            }
        ]

        # Get scene description from SmolVLM
        response = self.vision_model.invoke(messages)
        scene_description = response.content if hasattr(response, "content") else str(response)

        print(f"📝 Scene: {scene_description}...")

        # Update state with new description
        updated_descriptions = state["scene_descriptions"] + [scene_description]

        return {"scene_descriptions": updated_descriptions, "current_description": scene_description}

    def navigation_node(self, state: RobotState) -> RobotState:
        """Stage 2: Gemma3 makes navigation decision based on scene description"""

        print("🧠 Gemma3 deciding navigation...")

        # Format decision history
        decision_history = "No previous decisions"
        if state["decisions"]:
            recent_decisions = state["decisions"][-3:]
            decision_history = "\n".join([f"- {d.direction} at {d.speed}: {d.reason}" for d in recent_decisions])

        # Create navigation prompt with scene description
        prompt = NAVIGATION_PROMPT.format(
            decisions=decision_history, scene_description=state["current_description"], num_images=len(state["images"])
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

    async def generate_command(self, new_image: str) -> RobotDecision:
        """Generate a movement command using two-stage processing"""

        # Add new image to persistent state
        self.state["images"].append(new_image)
        self.state["current_image"] = new_image

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
