"""
bt_core.py

Pure-Python behavior tree data structures for LLM-BT planning-only reproduction.

Supports:
  - Sequence, Fallback (Selector), Parallel control nodes
  - Condition and Action leaf nodes (domain-agnostic names + params)
  - XML serialization/deserialization (compatible with original C++ BT format)
  - Action sequence extraction from fully-expanded trees
  - Symbolic tick with pluggable state evaluator
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence


class ReturnStatus(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()


class NodeType(Enum):
    SEQUENCE = "Sequence"
    FALLBACK = "Fallback"
    PARALLEL = "Parallel"
    CONDITION = "Condition"
    ACTION = "Action"


@dataclass
class BTNode:
    node_type: NodeType
    name: str = ""
    params: List[str] = field(default_factory=list)
    text_attr: str = ""
    children: List["BTNode"] = field(default_factory=list)
    last_failed_node: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        return self.node_type in (NodeType.CONDITION, NodeType.ACTION)

    def add_child(self, child: "BTNode") -> "BTNode":
        self.children.append(child)
        return child

    def deep_copy(self) -> "BTNode":
        return BTNode(
            node_type=self.node_type,
            name=self.name,
            params=list(self.params),
            text_attr=self.text_attr,
            children=[c.deep_copy() for c in self.children],
        )

    def find_nodes(self, name: Optional[str] = None, node_type: Optional[NodeType] = None) -> List["BTNode"]:
        results: List[BTNode] = []
        if name is not None and self.name == name:
            results.append(self)
        if node_type is not None and self.node_type == node_type:
            results.append(self)
        for child in self.children:
            results.extend(child.find_nodes(name=name, node_type=node_type))
        return results

    def find_first(self, name: Optional[str] = None, node_type: Optional[NodeType] = None) -> Optional["BTNode"]:
        if name is not None and self.name == name:
            return self
        if node_type is not None and self.node_type == node_type:
            return self
        for child in self.children:
            found = child.find_first(name=name, node_type=node_type)
            if found is not None:
                return found
        return None

    def replace_child(self, old_child: "BTNode", new_child: "BTNode") -> bool:
        for i, c in enumerate(self.children):
            if c is old_child:
                self.children[i] = new_child
                return True
            if not c.is_leaf:
                if c.replace_child(old_child, new_child):
                    return True
        return False

    def remove_child(self, child: "BTNode") -> bool:
        for i, c in enumerate(self.children):
            if c is child:
                self.children.pop(i)
                return True
        for c in self.children:
            if not c.is_leaf and c.remove_child(child):
                return True
        return False

    def count_nodes(self) -> int:
        return 1 + sum(c.count_nodes() for c in self.children)

    def extract_action_sequence(self) -> List[Dict[str, Any]]:
        seq: List[Dict[str, Any]] = []
        self._collect_actions(seq)
        return seq

    def _collect_actions(self, acc: List[Dict[str, Any]]) -> None:
        if self.node_type == NodeType.ACTION:
            acc.append({
                "type": "bt_action",
                "name": self.name,
                "args": list(self.params),
                "raw": self.text_attr,
                "step_index": len(acc),
            })
        elif self.node_type in (NodeType.SEQUENCE, NodeType.FALLBACK, NodeType.PARALLEL):
            for child in self.children:
                child._collect_actions(acc)

    def extract_goals(self) -> List[str]:
        if self.is_leaf:
            return [self.text_attr] if self.text_attr else []
        goals: List[str] = []
        for child in self.children:
            goals.extend(child.extract_goals())
        return goals

    def to_xml_element(self) -> ET.Element:
        if self.node_type == NodeType.SEQUENCE:
            el = ET.Element("Sequence")
            el.set("text", self.text_attr or "IFTHENELSE")
        elif self.node_type == NodeType.FALLBACK:
            el = ET.Element("Fallback")
            el.set("text", self.text_attr or "IFTHENELSE")
        elif self.node_type == NodeType.PARALLEL:
            el = ET.Element("Parallel")
            el.set("text", self.text_attr or "")
        elif self.node_type in (NodeType.CONDITION, NodeType.ACTION):
            el = ET.Element(self.name)
            el.set("text", self.text_attr)
        else:
            el = ET.Element("Node")
            el.set("type", self.node_type.value)
            el.set("name", self.name)
            if self.text_attr:
                el.set("text", self.text_attr)

        for child in self.children:
            el.append(child.to_xml_element())
        return el

    def to_xml(self, indent: int = 4) -> str:
        root_el = ET.Element("root")
        bt_el = ET.SubElement(root_el, "BehaviorTree")
        bt_el.append(self.to_xml_element())

        lines = ET.tostring(root_el, encoding="unicode")
        import re
        lines = re.sub(r"(<\/?\w+[^>]*>)(?=<\/?\w+)", r"\1\n", lines)
        pretty = _prettify_xml(lines, indent)
        return pretty

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "node_type": self.node_type.value,
            "name": self.name,
            "text_attr": self.text_attr,
        }
        if self.params:
            d["params"] = self.params
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"{self.name}({self.text_attr})"
        child_repr = ", ".join(repr(c) for c in self.children)
        return f"{self.node_type.value}({child_repr})"


def _prettify_xml(xml_str: str, indent: int = 4) -> str:
    root = ET.fromstring(xml_str)
    ET.indent(root, space=" " * indent)
    return ET.tostring(root, encoding="unicode", xml_declaration=False)


def from_xml(xml_str: str) -> BTNode:
    root = ET.fromstring(xml_str)
    bt_el = root.find("BehaviorTree")
    if bt_el is None:
        bt_el = root
    first_child = None
    for child in bt_el:
        first_child = child
        break
    if first_child is None:
        raise ValueError("No BehaviorTree child found in XML")
    return _parse_element(first_child)


def _parse_element(el: ET.Element) -> BTNode:
    tag = el.tag
    text = el.get("text", "")

    children_els = list(el)

    if tag == "Sequence":
        node = BTNode(node_type=NodeType.SEQUENCE, name="Sequence", text_attr=text)
        for child_el in children_els:
            node.add_child(_parse_element(child_el))
        return node

    if tag == "Fallback":
        node = BTNode(node_type=NodeType.FALLBACK, name="Fallback", text_attr=text)
        for child_el in children_els:
            node.add_child(_parse_element(child_el))
        return node

    if tag == "Parallel":
        node = BTNode(node_type=NodeType.PARALLEL, name="Parallel", text_attr=text)
        for child_el in children_els:
            node.add_child(_parse_element(child_el))
        return node

    action_names = {
        "Pick", "Place", "MoveTo", "Remove", "Drop",
        "Navigate", "PickUp", "PutDown", "PutIn",
        "OpenContainer", "CloseContainer", "Cook", "Slice", "Clean",
    }
    condition_names = {
        "Hold", "On", "Near", "HandEmpty", "Approach", "ExistPath",
        "At", "Holding", "In", "Open_pred", "Closed", "Clean_pred",
        "Cooked", "Sliced",
    }

    if tag in action_names:
        parts = text.split(",") if text else []
        return BTNode(node_type=NodeType.ACTION, name=tag, params=parts, text_attr=text)

    if tag in condition_names:
        parts = text.split(",") if text else []
        return BTNode(node_type=NodeType.CONDITION, name=tag, params=parts, text_attr=text)

    if children_els:
        node = BTNode(node_type=NodeType.SEQUENCE, name=tag, text_attr=text)
        for child_el in children_els:
            node.add_child(_parse_element(child_el))
        return node

    return BTNode(node_type=NodeType.CONDITION, name=tag, text_attr=text)


StateEvaluator = Callable[[str, List[str]], ReturnStatus]


def symbolic_tick(node: BTNode, state: Dict[str, Any], evaluator: StateEvaluator) -> ReturnStatus:
    if node.node_type == NodeType.CONDITION:
        return evaluator(node.name, node.params)

    if node.node_type == NodeType.ACTION:
        return ReturnStatus.SUCCESS

    if node.node_type == NodeType.SEQUENCE:
        for child in node.children:
            status = symbolic_tick(child, state, evaluator)
            if status == ReturnStatus.FAILURE:
                node.last_failed_node = child.text_attr or child.name
                return ReturnStatus.FAILURE
            if status == ReturnStatus.RUNNING:
                return ReturnStatus.RUNNING
        return ReturnStatus.SUCCESS

    if node.node_type == NodeType.FALLBACK:
        for child in node.children:
            status = symbolic_tick(child, state, evaluator)
            if status == ReturnStatus.SUCCESS:
                return ReturnStatus.SUCCESS
            if status == ReturnStatus.RUNNING:
                return ReturnStatus.RUNNING
        node.last_failed_node = node.text_attr
        return ReturnStatus.FAILURE

    if node.node_type == NodeType.PARALLEL:
        successes = 0
        failures = 0
        for child in node.children:
            status = symbolic_tick(child, state, evaluator)
            if status == ReturnStatus.SUCCESS:
                successes += 1
            elif status == ReturnStatus.FAILURE:
                failures += 1
        if successes == len(node.children):
            return ReturnStatus.SUCCESS
        if failures == len(node.children):
            return ReturnStatus.FAILURE
        return ReturnStatus.RUNNING

    return ReturnStatus.FAILURE