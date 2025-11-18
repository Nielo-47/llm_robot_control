import os
from dotenv import load_dotenv

load_dotenv()

DEVICE = "cpu"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
VISION_MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"
REFLEX_MODEL_NAME = "gemma3:1b"
NAVIGATION_MODEL_NAME = "gemini-2.5-flash"


VISION_PROMPT = """Describe the image in detail. After this, list the objects you see grouped by FAR, NEAR, and VERY NEAR."""

NAVIGATION_PROMPT = """You are a robot navigation system. Your goal is to find a green plant and navigate to it without hitting any obstacles.

Previous Commands: {decisions}
Visual Description: {scene_description}

Respond in TOML format:
direction = "N"  # Options: N/S/E/W/NE/NW/SE/SW/STOP/ROAM
speed = "MEDIUM"  # Options: SLOW/MEDIUM/FAST
reason = "brief explanation"
"""

REFLEX_PROMPT = """You are a robot. Based on the scene description of what the robot is seeing, is there an IMMEDIATE danger or obstacle requiring emergency stop to the robot?
Scene: {description}
Answer: 

DECISION: 'DANGER' or 'SAFE'.
REASON: brief explanation
"""
