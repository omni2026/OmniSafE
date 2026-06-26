"""
household_domain.py

Extended predicate/action definitions and Action Template Library (ATL)
expansion rules for the household scenario.

Preserves the original LLM-BT IF-THEN-ELSE expansion pattern while adding
household-specific predicates (At, Holding, In, Open, Closed, Clean, Cooked, Sliced)
and actions (Navigate, PickUp, PutDown, PutIn, OpenContainer, CloseContainer, Cook, Slice, Clean).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .bt_core import BTNode, NodeType


@dataclass
class PredicateDef:
    name: str
    arg_count: int
    description: str


@dataclass
class ActionDef:
    name: str
    arg_count: int
    description: str


PREDICATES: Dict[str, PredicateDef] = {
    "At": PredicateDef("At", 1, "Robot is at location"),
    "Holding": PredicateDef("Holding", 1, "Robot is holding object"),
    "On": PredicateDef("On", 2, "Object is on surface"),
    "In": PredicateDef("In", 2, "Object is in container"),
    "Open": PredicateDef("Open", 1, "Container is open"),
    "Closed": PredicateDef("Closed", 1, "Container is closed"),
    "Clean": PredicateDef("Clean", 1, "Object is clean"),
    "Cooked": PredicateDef("Cooked", 1, "Food is cooked"),
    "Sliced": PredicateDef("Sliced", 1, "Object is sliced"),
    "Near": PredicateDef("Near", 1, "Robot is near location"),
    "ExistPath": PredicateDef("ExistPath", 1, "Path exists to location"),
    "HandEmpty": PredicateDef("HandEmpty", 0, "Robot hand is empty"),
    "Approach": PredicateDef("Approach", 1, "Robot can approach object at location"),
    "Hold": PredicateDef("Hold", 2, "Robot is holding object from location (legacy)"),
}

ACTIONS: Dict[str, ActionDef] = {
    "Navigate": ActionDef("Navigate", 1, "Navigate robot to location"),
    "PickUp": ActionDef("PickUp", 2, "Pick up object from location"),
    "PutDown": ActionDef("PutDown", 2, "Put down object on surface"),
    "PutIn": ActionDef("PutIn", 2, "Put object into container"),
    "OpenContainer": ActionDef("OpenContainer", 1, "Open a container"),
    "CloseContainer": ActionDef("CloseContainer", 1, "Close a container"),
    "Cook": ActionDef("Cook", 2, "Cook food using appliance"),
    "Slice": ActionDef("Slice", 1, "Slice object"),
    "Clean": ActionDef("Clean", 1, "Clean object"),
    "Pick": ActionDef("Pick", 2, "Legacy: pick object from location"),
    "Place": ActionDef("Place", 3, "Legacy: place object from loc to dest"),
    "MoveTo": ActionDef("MoveTo", 1, "Legacy: move to location"),
    "Remove": ActionDef("Remove", 3, "Legacy: remove obstacle from location"),
    "Drop": ActionDef("Drop", 1, "Legacy: drop held object"),
}

LEGACY_PREDICATE_TO_HOUSEHOLD: Dict[str, str] = {
    "Hold": "Holding",
    "HandEmpty": "HandEmpty",
    "On": "On",
    "Near": "Near",
    "Approach": "Approach",
    "ExistPath": "ExistPath",
}

LEGACY_ACTION_TO_HOUSEHOLD: Dict[str, str] = {
    "Pick": "PickUp",
    "Place": "PutDown",
    "MoveTo": "Navigate",
    "Remove": "Remove",
    "Drop": "PutDown",
}


ExpansionFunc = Callable[[str, List[str], Dict[str, Any]], List[BTNode]]


def _make_fallback_if_then_else(condition: BTNode, then_sequence: List[BTNode]) -> BTNode:
    fallback = BTNode(node_type=NodeType.FALLBACK, name="Fallback", text_attr="IFTHENELSE")
    fallback.add_child(condition)
    seq = BTNode(node_type=NodeType.SEQUENCE, name="Sequence", text_attr="IFTHENELSE")
    for child in then_sequence:
        seq.add_child(child)
    fallback.add_child(seq)
    return fallback


def _cond(pred_name: str, args: List[str]) -> BTNode:
    text = ",".join(args) if args else ""
    return BTNode(node_type=NodeType.CONDITION, name=pred_name, params=list(args), text_attr=text)


def _act(act_name: str, args: List[str]) -> BTNode:
    text = ",".join(args) if args else ""
    return BTNode(node_type=NodeType.ACTION, name=act_name, params=list(args), text_attr=text)


class ExpansionRule:
    def __init__(
        self,
        predicate_name: str,
        arg_count: int,
        expand_fn: ExpansionFunc,
        description: str = "",
    ):
        self.predicate_name = predicate_name
        self.arg_count = arg_count
        self.expand_fn = expand_fn
        self.description = description

    def matches(self, node: BTNode) -> bool:
        return node.name == self.predicate_name and len(node.params) >= self.arg_count

    def expand(self, node: BTNode, context: Dict[str, Any]) -> List[BTNode]:
        return self.expand_fn(node.name, node.params, context)


def _expand_at(name: str, params: List[str], context: Dict[str, Any]) -> List[BTNode]:
    loc = params[0]
    condition = _cond("At", [loc])
    then_steps = [
        _cond("ExistPath", [loc]),
        _act("Navigate", [loc]),
    ]
    return [_make_fallback_if_then_else(condition, then_steps)]


def _expand_holding(name: str, params: List[str], context: Dict[str, Any]) -> List[BTNode]:
    obj = params[0]
    semantic_map = context.get("semantic_map")
    source_loc = ""
    if semantic_map is not None:
        source_loc = semantic_map.find_object_location(obj) or "unknown_location"

    condition = _cond("Holding", [obj])
    then_steps = [
        _cond("HandEmpty", []),
    ]
    if source_loc:
        then_steps.extend([
            _cond("Near", [source_loc]),
            _cond("Approach", [source_loc]),
            _act("PickUp", [obj, source_loc]),
        ])
    else:
        then_steps.extend([
            _act("PickUp", [obj]),
        ])
    return [_make_fallback_if_then_else(condition, then_steps)]


def _expand_on_surface(name: str, params: List[str], context: Dict[str, Any]) -> List[BTNode]:
    if len(params) >= 3:
        obj = params[0]
        source = params[1]
        dest = params[2]
    elif len(params) >= 2:
        obj = params[0]
        dest = params[1]
        source = ""
        semantic_map = context.get("semantic_map")
        if semantic_map is not None:
            source = semantic_map.find_object_location(obj) or ""
    else:
        return []

    condition = _cond("On", [obj, dest])
    then_steps = [
        _cond("Holding", [obj]),
        _cond("Near", [dest]),
        _cond("Approach", [dest]),
        _act("PutDown", [obj, dest]),
    ]
    return [_make_fallback_if_then_else(condition, then_steps)]


def _expand_near(name: str, params: List[str], context: Dict[str, Any]) -> List[BTNode]:
    loc = params[-1] if params else "unknown"
    condition = _cond("Near", [loc])
    then_steps = [
        _cond("ExistPath", [loc]),
        _act("Navigate", [loc]),
    ]
    return [_make_fallback_if_then_else(condition, then_steps)]


def _expand_in(name: str, params: List[str], context: Dict[str, Any]) -> List[BTNode]:
    obj = params[0]
    container = params[1]
    condition = _cond("In", [obj, container])
    then_steps = [
        _cond("Holding", [obj]),
        _cond("Open", [container]),
        _cond("Near", [container]),
        _act("PutIn", [obj, container]),
    ]
    return [_make_fallback_if_then_else(condition, then_steps)]


def _expand_open(name: str, params: List[str], context: Dict[str, Any]) -> List[BTNode]:
    container = params[0]
    condition = _cond("Open", [container])
    then_steps = [
        _cond("Near", [container]),
        _act("OpenContainer", [container]),
    ]
    return [_make_fallback_if_then_else(condition, then_steps)]


def _expand_closed(name: str, params: List[str], context: Dict[str, Any]) -> List[BTNode]:
    container = params[0]
    condition = _cond("Closed", [container])
    then_steps = [
        _cond("Near", [container]),
        _act("CloseContainer", [container]),
    ]
    return [_make_fallback_if_then_else(condition, then_steps)]


def _expand_clean(name: str, params: List[str], context: Dict[str, Any]) -> List[BTNode]:
    obj = params[0]
    semantic_map = context.get("semantic_map")
    sink_loc = "sink"
    if semantic_map is not None:
        sink_loc = semantic_map.find_location_of_type("sink") or "sink"

    condition = _cond("Clean", [obj])
    then_steps = [
        _cond("Holding", [obj]),
        _cond("Near", [sink_loc]),
        _act("Clean", [obj]),
    ]
    return [_make_fallback_if_then_else(condition, then_steps)]


def _expand_cooked(name: str, params: List[str], context: Dict[str, Any]) -> List[BTNode]:
    food = params[0]
    semantic_map = context.get("semantic_map")
    appliance = "stove"
    if semantic_map is not None:
        appliance = semantic_map.find_location_of_type("appliance") or "stove"

    condition = _cond("Cooked", [food])
    then_steps = [
        _cond("Holding", [food]),
        _cond("Near", [appliance]),
        _act("Cook", [food, appliance]),
    ]
    return [_make_fallback_if_then_else(condition, then_steps)]


def _expand_sliced(name: str, params: List[str], context: Dict[str, Any]) -> List[BTNode]:
    obj = params[0]
    condition = _cond("Sliced", [obj])
    then_steps = [
        _cond("Holding", [obj]),
        _act("Slice", [obj]),
    ]
    return [_make_fallback_if_then_else(condition, then_steps)]


def _expand_approach(name: str, params: List[str], context: Dict[str, Any]) -> List[BTNode]:
    dest = params[0] if params else "unknown"
    condition = _cond("Approach", [dest])
    then_steps = [
        _cond("HandEmpty", []),
    ]
    obstacles = context.get("obstacles", [])
    if obstacles:
        obstacle_name = obstacles[0] if obstacles else "obstacle"
        then_steps.append(_act("Remove", [dest, obstacle_name]))
    else:
        then_steps.append(_act("Remove", [dest, "obstacle"]))
    return [_make_fallback_if_then_else(condition, then_steps)]


def _expand_hand_empty(name: str, params: List[str], context: Dict[str, Any]) -> List[BTNode]:
    condition = _cond("HandEmpty", [])
    drop = _act("Drop", ["held_object"])
    return [_make_fallback_if_then_else(condition, [drop])]


def _expand_hold_legacy(name: str, params: List[str], context: Dict[str, Any]) -> List[BTNode]:
    if len(params) >= 2:
        obj, loc = params[0], params[1]
    else:
        obj = params[0] if params else "unknown"
        loc = "unknown_location"

    condition = _cond("Hold", [obj, loc])
    then_steps = [
        _cond("HandEmpty", [obj]),
        _cond("Near", [obj, loc]),
        _cond("Approach", [obj, loc]),
        _act("Pick", [obj, loc]),
    ]
    return [_make_fallback_if_then_else(condition, then_steps)]


def _expand_on_legacy(name: str, params: List[str], context: Dict[str, Any]) -> List[BTNode]:
    if len(params) >= 3:
        obj, loc, dest = params[0], params[1], params[2]
    elif len(params) >= 2:
        obj, dest = params[0], params[1]
        loc = ""
        semantic_map = context.get("semantic_map")
        if semantic_map is not None:
            loc = semantic_map.find_object_location(obj) or ""
    else:
        return []

    condition = _cond("On", [obj, loc, dest] if loc else [obj, dest])
    then_steps = [
        _cond("Hold", [obj, loc] if loc else [obj]),
    ]
    then_steps.extend([
        _cond("Near", [dest]),
        _cond("Approach", [dest]),
        _act("Place", [obj, loc, dest] if loc else [obj, dest]),
    ])
    return [_make_fallback_if_then_else(condition, then_steps)]


DEFAULT_EXPANSION_RULES: Dict[str, ExpansionRule] = {
    "At": ExpansionRule("At", 1, _expand_at, "Robot at location"),
    "Holding": ExpansionRule("Holding", 1, _expand_holding, "Robot holding object"),
    "On": ExpansionRule("On", 2, _expand_on_surface, "Object on surface (household)"),
    "In": ExpansionRule("In", 2, _expand_in, "Object in container"),
    "Open": ExpansionRule("Open", 1, _expand_open, "Container is open"),
    "Closed": ExpansionRule("Closed", 1, _expand_closed, "Container is closed"),
    "Clean": ExpansionRule("Clean", 1, _expand_clean, "Object is clean"),
    "Cooked": ExpansionRule("Cooked", 1, _expand_cooked, "Food is cooked"),
    "Sliced": ExpansionRule("Sliced", 1, _expand_sliced, "Object is sliced"),
    "Near": ExpansionRule("Near", 1, _expand_near, "Robot near location"),
    "Approach": ExpansionRule("Approach", 1, _expand_approach, "Robot can approach"),
    "HandEmpty": ExpansionRule("HandEmpty", 0, _expand_hand_empty, "Robot hand empty"),
    "Hold": ExpansionRule("Hold", 2, _expand_hold_legacy, "Robot holding object from loc (legacy)"),
    "On_legacy": ExpansionRule("On", 3, _expand_on_legacy, "Object on dest from loc (legacy)"),
}


GOAL_CONDITION_TEMPLATES: Dict[str, str] = {
    "At": "At(location)",
    "Holding": "Holding(object)",
    "On": "On(object, surface)",
    "In": "In(object, container)",
    "Open": "Open(container)",
    "Closed": "Closed(container)",
    "Clean": "Clean(object)",
    "Cooked": "Cooked(food)",
    "Sliced": "Sliced(object)",
    "Near": "Near(location)",
    "Hold": "Hold(object, location)",
    "On_legacy": "(object,location)On(destination)",
}

GOAL_TEMPLATE_TO_PREDICATE: Dict[str, str] = {
    "hold": "Hold",
    "on": "On",
    "near": "Near",
    "at": "At",
    "holding": "Holding",
    "in": "In",
    "open": "Open",
    "closed": "Closed",
    "clean": "Clean",
    "cooked": "Cooked",
    "sliced": "Sliced",
}


@dataclass
class GoalCondition:
    predicate: str
    args: List[str]
    raw_text: str = ""

    def __post_init__(self):
        if not self.raw_text:
            self.raw_text = f"{self.predicate}({','.join(self.args)})"

    def to_bt_node(self) -> BTNode:
        return _cond(self.predicate, self.args)


class HouseholdDomain:
    def __init__(self, use_legacy: bool = False):
        self.use_legacy = use_legacy
        self.expansion_rules: Dict[str, ExpansionRule] = dict(DEFAULT_EXPANSION_RULES)

    def get_expansion_rule(self, predicate_name: str) -> Optional[ExpansionRule]:
        return self.expansion_rules.get(predicate_name)

    def get_all_predicate_names(self) -> List[str]:
        return list(PREDICATES.keys())

    def get_all_action_names(self) -> List[str]:
        return list(ACTIONS.keys())

    def resolve_rule_for_node(self, node: BTNode) -> Optional[ExpansionRule]:
        rule = self.expansion_rules.get(node.name)
        if rule is not None and rule.matches(node):
            return rule
        if node.name == "On" and len(node.params) == 3 and not self.use_legacy:
            return self.expansion_rules.get("On_legacy")
        return None