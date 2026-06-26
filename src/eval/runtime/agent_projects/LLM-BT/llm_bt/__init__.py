"""LLM-BT planning-only reproduction package.

Modules:
  bt_core           -- Behavior tree data structures + XML serialization
  household_domain   -- Extended predicates, actions, ATL expansion rules
  semantic_map       -- XML semantic map parsing + initial state generation
  bt_expansion       -- LLM-assisted BT expansion engine
  prompts            -- Prompt templates for intention reasoning
  intention_reasoning -- Stage 1: NL -> goal conditions via LLM
"""

from .bt_core import BTNode, NodeType, ReturnStatus, from_xml, symbolic_tick
from .household_domain import GoalCondition, HouseholdDomain, ExpansionRule
from .semantic_map import SemanticMap, Location, parse_semantic_map_xml, parse_semantic_map_file
from .bt_expansion import BTExpansionEngine, LLMConditionEvaluator
from .intention_reasoning import IntentionReasoner
from .prompts import PromptConfig, parse_goal_conditions