"""
prompts.py

Prompt templates for LLM-BT Intention Reasoning (Stage 1).

Reproduces the original LLM-BT prompt engineering flow:
  - System prompt (from Prompts.txt)
  - Few-shot demonstrations (from Demonstrates.txt)
  - Incomplete instruction guidance (from Guidance.txt)
  - Mode-specific templates (household domain extension)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SYSTEM_PROMPT_TEMPLATE = (
    "As a household service robot, you need to generate the goal conditions "
    "of the task according to the user's instructions.\n"
    "Only one object can be operated at a time.\n"
    "The operable objects are arranged as shown in the SemanticMap.\n"
    "Please only generate goal conditions without explanation.\n"
    "The goal conditions templates: {templates}\n"
    "obj represents an object, loc represents a location, des represents a destination.\n"
    "There are some examples in the demonstrations."
)

LEGACY_SYSTEM_PROMPT = (
    "As a household service robot, you need to generate the goal conditions "
    "of the task according to the user's instructions.\n"
    "Only one object can be operated each time.\n"
    "The operable objects are arranged as shown in SemanticMap.xml\n"
    "Please only generate goal conditions without explanation.\n"
    "The goal conditions templates: hold(obj,loc),(obj,loc)On(des),Near(loc),Near(des).\n"
    "obj represents an object, loc represents a location, des represents a destination.\n"
    "There are some examples in Demonstrates.txt"
)

HOUSEHOLD_SYSTEM_PROMPT = (
    "As a household service robot, you need to generate the goal conditions "
    "of the task according to the user's instructions.\n"
    "Only one object can be operated at a time.\n"
    "The operable objects and locations are described in the SemanticMap below.\n"
    "Please only generate goal conditions without explanation.\n"
    "Available goal condition templates:\n"
    "  At(location) - robot should be at a location\n"
    "  Holding(object) - robot should be holding an object\n"
    "  On(object, surface) - object should be on a surface\n"
    "  In(object, container) - object should be inside a container\n"
    "  Open(container) - container should be open\n"
    "  Closed(container) - container should be closed\n"
    "  Clean(object) - object should be clean\n"
    "  Cooked(food) - food should be cooked\n"
    "  Sliced(object) - object should be sliced\n"
)


DEMONSTRATIONS_LEGACY = [
    {"input": "Go to the bar", "output": "Near(bar)", "note": "the goal is the robot near the bar."},
    {"input": "I want to eat chocolate, take it on the desk", "output": "(chocolate,table)On(desk)", "note": "the goal is the chocolate from table on the desk."},
    {"input": "I am thirsty, could you give me some drink to me, i am on the desk", "output": "( , )On(desk)", "note": "if the information is unclear or abstract, you can leave it blank."},
    {"input": "I want to drink water", "output": "(water,bar)On( )", "note": "if the information is unclear or abstract, you can leave it blank."},
    {"input": "Place all yellow cargos on the first layer", "output": "(yellow3,L2)On(L1)", "note": "yellow1 and yellow2 are already on L1, no need to operate them. just operate yellow3."},
    {"input": "grasp a bottle of tea", "output": "hold(tea,bar)", "note": "the goal is grasp the tea."},
    {"input": "put two red cargos on my desk", "output": "(red1,L2)On(desk),(red2,L2)On(desk)", "note": "the goal is put separate red cargos on the desk."},
]

DEMONSTRATIONS_HOUSEHOLD = [
    {"input": "Pick up the apple from the kitchen counter and put it on the dining table", "output": "On(apple, dining_table)", "note": "object should end up on the target surface"},
    {"input": "Go to the kitchen", "output": "At(kitchen)", "note": "robot should navigate to the location"},
    {"input": "Hold the knife", "output": "Holding(knife)", "note": "robot should be holding the object"},
    {"input": "Put the milk in the fridge", "output": "In(milk, fridge)", "note": "object should be inside the container"},
    {"input": "Open the cabinet", "output": "Open(cabinet)", "note": "container should be open"},
    {"input": "Close the fridge", "output": "Closed(fridge)", "note": "container should be closed"},
    {"input": "Clean the apple", "output": "Clean(apple)", "note": "object should be clean"},
    {"input": "Cook the egg using the stove", "output": "Cooked(egg)", "note": "food should be cooked"},
    {"input": "Slice the bread", "output": "Sliced(bread)", "note": "object should be sliced"},
    {"input": "Bring me the knife from the counter, I am at the table", "output": "On(knife, dining_table)", "note": "deliver object to where the user is"},
]


GUIDANCE_CASES = [
    {"type": "On", "missing": ["obj"], "filled": ["loc", "des"], "question": "What object do you want to move?"},
    {"type": "On", "missing": ["des"], "filled": ["obj", "loc"], "question": "Where do you want to put {obj}?"},
    {"type": "On", "missing": ["obj", "des"], "filled": ["loc"], "question": "What do you want from {loc}, and where should it go?"},
    {"type": "Holding", "missing": ["obj"], "filled": [], "question": "What object do you want to hold?"},
    {"type": "In", "missing": ["obj"], "filled": ["container"], "question": "What object do you want to put in {container}?"},
    {"type": "In", "missing": ["container"], "filled": ["obj"], "question": "Which container should {obj} go into?"},
    {"type": "At", "missing": ["location"], "filled": [], "question": "Where should the robot go?"},
    {"type": "Clean", "missing": ["obj"], "filled": [], "question": "What object should be cleaned?"},
    {"type": "Cooked", "missing": ["food"], "filled": [], "question": "What food should be cooked?"},
    {"type": "Sliced", "missing": ["obj"], "filled": [], "question": "What object should be sliced?"},
]


@dataclass
class PromptConfig:
    domain: str = "household"
    use_legacy_format: bool = False
    max_guidance_rounds: int = 2
    include_reasoning_line: bool = False

    def get_system_prompt(self, semantic_map_desc: str = "") -> str:
        if self.use_legacy_format:
            base = LEGACY_SYSTEM_PROMPT
        else:
            base = HOUSEHOLD_SYSTEM_PROMPT

        parts = [base]
        if self.include_reasoning_line:
            parts.append(
                "Before the goal conditions, output exactly one line starting "
                "with 'reasoning: ' briefly describing your task plan."
            )
        if semantic_map_desc:
            parts.append(f"\nSemanticMap (scene layout and objects):\n{semantic_map_desc}")
        return "\n".join(parts)

    def get_demonstrations(self) -> List[Dict[str, str]]:
        if self.use_legacy_format:
            return list(DEMONSTRATIONS_LEGACY)
        return list(DEMONSTRATIONS_HOUSEHOLD)

    def format_demonstration_messages(
        self,
        semantic_map_desc: str = "",
    ) -> List[Dict[str, str]]:
        system = self.get_system_prompt(semantic_map_desc)
        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        for demo in self.get_demonstrations():
            messages.append({"role": "user", "content": f"Instruction: {demo['input']}"})
            messages.append({"role": "assistant", "content": demo["output"]})
        return messages


def build_reasoning_prompt(
    instruction: str,
    prompt_config: PromptConfig,
    semantic_map_desc: str = "",
) -> List[Dict[str, str]]:
    messages = prompt_config.format_demonstration_messages(semantic_map_desc)
    messages.append({"role": "user", "content": f"Instruction: {instruction}"})
    return messages


def build_guidance_prompt(
    partial_output: str,
    instruction: str,
    prompt_config: PromptConfig,
) -> str:
    return (
        f"The instruction '{instruction}' produced an incomplete goal condition: '{partial_output}'.\n"
        f"Some parameters are missing. Please ask the user a clarifying question, "
        f"or if you can infer the missing information from context, fill it in and "
        f"output the complete goal condition.\n"
        f"Output only the goal condition(s)."
    )


def parse_goal_conditions(raw_output: str, domain: str = "household") -> List[Dict[str, Any]]:
    from .household_domain import GOAL_TEMPLATE_TO_PREDICATE

    goals: List[Dict[str, Any]] = []
    raw_output = raw_output.strip()
    used_spans: List[tuple] = []

    def _span_overlaps(start: int, end: int) -> bool:
        for s, e in used_spans:
            if start < e and end > s:
                return True
        return False

    legacy_pattern_pairs = [
        (r"\((\w+),(\w+)\)On\((\w+)\)", "On"),
        (r"hold\((\w+),(\w+)\)", "Hold"),
        (r"Near\((\w+)\)", "Near"),
    ]
    for pattern, pred_name in legacy_pattern_pairs:
        import re
        for match in re.finditer(pattern, raw_output):
            start, end = match.start(), match.end()
            if _span_overlaps(start, end):
                continue
            used_spans.append((start, end))
            args = [g for g in match.groups() if g]
            goals.append({
                "predicate": pred_name,
                "args": args,
                "raw_text": match.group(0),
            })

    household_patterns = [
        (r"At\(([^)]+)\)", "At"),
        (r"Holding\(([^)]+)\)", "Holding"),
        (r"On\(([^)]+)\)", "On"),
        (r"In\(([^)]+)\)", "In"),
        (r"Open\(([^)]+)\)", "Open"),
        (r"Closed\(([^)]+)\)", "Closed"),
        (r"Clean\(([^)]+)\)", "Clean"),
        (r"Cooked\(([^)]+)\)", "Cooked"),
        (r"Sliced\(([^)]+)\)", "Sliced"),
    ]
    for pattern, pred_name in household_patterns:
        import re
        for match in re.finditer(pattern, raw_output):
            start, end = match.start(), match.end()
            if _span_overlaps(start, end):
                continue
            used_spans.append((start, end))
            inner = match.group(1)
            args = [a.strip() for a in inner.split(",")]
            goals.append({
                "predicate": pred_name,
                "args": args,
                "raw_text": match.group(0),
            })

    if not goals:
        for token in raw_output.split(","):
            token = token.strip()
            if token and "(" in token:
                import re
                m = re.match(r"(\w+)\(([^)]*)\)", token)
                if m:
                    pred = m.group(1)
                    args = [a.strip() for a in m.group(2).split(",") if a.strip()]
                    goals.append({
                        "predicate": pred,
                        "args": args,
                        "raw_text": token,
                    })

    return goals
