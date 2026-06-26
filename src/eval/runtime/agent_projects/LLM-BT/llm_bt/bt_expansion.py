"""
bt_expansion.py

LLM-assisted BT expansion engine — the core of LLM-BT planning.

Instead of relying on a simulator to tick conditions and detect obstacles,
this engine uses an LLM to:
  1. Evaluate whether conditions hold in the current world state
  2. Detect obstacles preventing approach/navigation
  3. Suggest goal re-prioritization when obstacles block progress

The expansion algorithm faithfully reproduces the original C++ ExpandTree():
  - Build initial BT from goal conditions (Sequence of condition nodes)
  - Symbolic tick: evaluate each condition via LLM or state dict
  - On FAILURE: expand the failed condition using ATL rules (IF-THEN-ELSE)
  - Recurse into expanded sub-trees until max depth or all conditions pass
  - Extract ordered action sequence from fully expanded BT
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .bt_core import BTNode, NodeType, ReturnStatus, symbolic_tick
from .household_domain import (
    GoalCondition,
    HouseholdDomain,
    ExpansionRule,
    _cond,
    _act,
    _make_fallback_if_then_else,
)
from .semantic_map import SemanticMap

logger = logging.getLogger(__name__)


class LLMConditionEvaluator:
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        self._llm_config = llm_config or {}
        self._client = None
        self._model = ""
        if llm_config:
            self._model = str(llm_config.get("model", ""))

    def _get_client(self):
        if self._client is None and self._llm_config:
            try:
                from openai import OpenAI
                api_key = str(self._llm_config.get("api_key", "") or "")
                base_url = str(self._llm_config.get("base_url", "") or "") or None
                if api_key:
                    self._client = OpenAI(api_key=api_key, base_url=base_url)
            except ImportError:
                logger.warning("openai package not available; LLM condition evaluation will use fallback")
        return self._client

    async def evaluate_condition(
        self,
        condition_name: str,
        params: List[str],
        world_state: Dict[str, bool],
        semantic_map: SemanticMap,
    ) -> bool:
        state_key = self._make_state_key(condition_name, params)
        if state_key in world_state:
            return world_state[state_key]

        fallback_result = self._fallback_evaluate(condition_name, params, world_state, semantic_map)
        if fallback_result is not None:
            return fallback_result

        client = self._get_client()
        if client is None or not self._model:
            return False

        return await self._llm_evaluate(condition_name, params, world_state, semantic_map)

    @staticmethod
    def _make_state_key(condition_name: str, params: List[str]) -> str:
        if condition_name in ("On",) and len(params) >= 3:
            return f"{condition_name}({params[0]},{params[1]},{params[2]})"
        if condition_name in ("On",) and len(params) >= 2:
            return f"{condition_name}({params[0]},{params[1]})"
        if condition_name in ("Hold", "Holding") and len(params) >= 1:
            return f"Holding({params[0]})"
        if condition_name in ("Near", "At", "ExistPath", "Approach") and len(params) >= 1:
            return f"{condition_name}({params[-1]})"
        if condition_name in ("In",) and len(params) >= 2:
            return f"{condition_name}({params[0]},{params[1]})"
        if condition_name in ("Open", "Closed", "Clean", "Cooked", "Sliced") and len(params) >= 1:
            return f"{condition_name}({params[0]})"
        if condition_name == "HandEmpty":
            return "HandEmpty"
        return f"{condition_name}({','.join(params)})"

    def _fallback_evaluate(
        self,
        condition_name: str,
        params: List[str],
        world_state: Dict[str, bool],
        semantic_map: SemanticMap,
    ) -> Optional[bool]:
        if condition_name == "ExistPath":
            return True
        if condition_name == "HandEmpty":
            return world_state.get("HandEmpty", True)
        if condition_name in ("Open", "Closed"):
            obj = params[0] if params else ""
            for key, val in world_state.items():
                if key.startswith(f"{condition_name}(") and obj in key:
                    return val
            return condition_name == "Closed"
        if condition_name == "At":
            return False
        if condition_name == "Near":
            return False
        if condition_name == "Approach":
            return True
        if condition_name in ("Holding", "Hold"):
            return False
        if condition_name in ("On", "In", "Clean", "Cooked", "Sliced"):
            return False
        return None

    async def _llm_evaluate(
        self,
        condition_name: str,
        params: List[str],
        world_state: Dict[str, bool],
        semantic_map: SemanticMap,
    ) -> bool:
        client = self._get_client()
        if client is None:
            return False

        state_desc = self._format_state(world_state, semantic_map)
        params_str = ",".join(params)
        prompt = (
            f"You are a household robot condition evaluator. "
            f"Given the current world state, determine if the condition "
            f"'{condition_name}({params_str})' is satisfied.\n\n"
            f"World state:\n{state_desc}\n\n"
            f"Is the condition '{condition_name}({params_str})' satisfied? "
            f"Answer only 'Yes' or 'No'."
        )

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0,
                ),
            )
            content = (response.choices[0].message.content or "").strip().lower()
            return "yes" in content
        except Exception as e:
            logger.warning("LLM condition evaluation failed for %s(%s): %s", condition_name, params_str, e)
            return False

    @staticmethod
    def _format_state(world_state: Dict[str, bool], semantic_map: SemanticMap) -> str:
        true_states = [k for k, v in world_state.items() if v]
        false_states = [k for k, v in world_state.items() if not v]
        parts = []
        if true_states:
            parts.append("TRUE: " + ", ".join(true_states))
        if false_states:
            parts.append("FALSE: " + ", ".join(false_states[:20]))
        return "\n".join(parts)

    async def detect_obstacles(
        self,
        target_location: str,
        world_state: Dict[str, bool],
        semantic_map: SemanticMap,
    ) -> List[str]:
        client = self._get_client()
        if client is None or not self._model:
            loc = semantic_map.locations.get(target_location)
            if loc and len(loc.objects) > 3:
                return [loc.objects[0]["name"]]
            return []

        state_desc = self._format_state(world_state, semantic_map)
        map_desc = semantic_map.get_state_description()
        prompt = (
            f"You are a household robot navigation assistant. "
            f"The robot wants to approach location '{target_location}'.\n\n"
            f"World state:\n{state_desc}\n\n"
            f"Scene layout:\n{map_desc}\n\n"
            f"Are there any objects blocking the robot's path to '{target_location}'? "
            f"If yes, list the object names separated by commas. If no, answer 'None'."
        )

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.0,
                ),
            )
            content = (response.choices[0].message.content or "").strip()
            if content.lower() in ("none", "no", "no objects", "nothing", ""):
                return []
            return [o.strip() for o in content.split(",") if o.strip()]
        except Exception as e:
            logger.warning("LLM obstacle detection failed: %s", e)
            return []


class BTExpansionEngine:
    def __init__(
        self,
        domain: HouseholdDomain,
        semantic_map: SemanticMap,
        llm_config: Optional[Dict[str, Any]] = None,
        max_expand_depth: int = 5,
        max_nodes: int = 500,
    ):
        self.domain = domain
        self.semantic_map = semantic_map
        self.max_expand_depth = max_expand_depth
        self.max_nodes = max_nodes
        self.evaluator = LLMConditionEvaluator(llm_config)
        self._initial_state: Dict[str, bool] = semantic_map.get_initial_state()
        self._expanded_keys: set = set()

    def set_initial_state(self, state: Dict[str, bool]) -> None:
        self._initial_state = dict(state)

    def build_initial_tree(self, goal_conditions: List[GoalCondition]) -> BTNode:
        if len(goal_conditions) == 1:
            return goal_conditions[0].to_bt_node()
        root = BTNode(node_type=NodeType.SEQUENCE, name="Sequence", text_attr="IFTHENELSE")
        for gc in goal_conditions:
            root.add_child(gc.to_bt_node())
        return root

    async def expand_from_goals(self, goal_conditions: List[GoalCondition]) -> BTNode:
        bt = self.build_initial_tree(goal_conditions)
        self._expanded_keys = set()
        expanded = await self._recursive_expand(bt, self._initial_state, depth=0)
        return expanded

    async def _recursive_expand(
        self,
        bt: BTNode,
        state: Dict[str, bool],
        depth: int,
    ) -> BTNode:
        if depth >= self.max_expand_depth:
            return bt

        bt = bt.deep_copy()
        changed = True
        iterations = 0
        max_iterations = self.max_expand_depth * 3

        while changed and iterations < max_iterations:
            if bt.count_nodes() > self.max_nodes:
                logger.warning("BT exceeded max_nodes (%d), stopping expansion", self.max_nodes)
                break
            changed = False
            iterations += 1
            bt, did_expand = await self._expand_one_pass(bt, state, depth)
            if did_expand:
                changed = True

        return bt

    async def _expand_one_pass(
        self,
        bt: BTNode,
        state: Dict[str, bool],
        depth: int,
    ) -> Tuple[BTNode, bool]:
        return await self._expand_pass_recursive(bt, state, depth)

    async def _expand_pass_recursive(
        self,
        node: BTNode,
        state: Dict[str, bool],
        depth: int,
    ) -> Tuple[BTNode, bool]:
        if node.node_type == NodeType.CONDITION:
            is_satisfied = await self.evaluator.evaluate_condition(
                node.name, node.params, state, self.semantic_map
            )
            if is_satisfied:
                return node, False

            expansion_key = f"{node.name}({','.join(node.params)})"
            if expansion_key in self._expanded_keys:
                return node, False
            self._expanded_keys.add(expansion_key)

            rule = self.domain.resolve_rule_for_node(node)
            if rule is None:
                rule = self.domain.get_expansion_rule(node.name)
            if rule is not None and rule.matches(node):
                context = {
                    "semantic_map": self.semantic_map,
                    "world_state": state,
                    "obstacles": [],
                }

                if node.name == "Approach":
                    obstacles = await self.evaluator.detect_obstacles(
                        node.params[0] if node.params else "unknown",
                        state,
                        self.semantic_map,
                    )
                    context["obstacles"] = obstacles

                expanded_nodes = rule.expand(node, context)
                if expanded_nodes:
                    return expanded_nodes[0], True
            return node, False

        if node.node_type == NodeType.ACTION:
            return node, False

        if node.node_type in (NodeType.SEQUENCE, NodeType.FALLBACK, NodeType.PARALLEL):
            new_children: List[BTNode] = []
            any_expanded = False
            for child in node.children:
                new_child, child_expanded = await self._expand_pass_recursive(child, state, depth + 1)
                new_children.append(new_child)
                if child_expanded:
                    any_expanded = True
            node.children = new_children
            return node, any_expanded

        return node, False

    async def expand_tree_from_xml(self, xml_str: str, state: Optional[Dict[str, bool]] = None) -> BTNode:
        from .bt_core import from_xml
        bt = from_xml(xml_str)
        if state is not None:
            self._initial_state = dict(state)
        expanded = await self._recursive_expand(bt, self._initial_state, depth=0)
        return expanded

    def get_initial_state(self) -> Dict[str, bool]:
        return dict(self._initial_state)