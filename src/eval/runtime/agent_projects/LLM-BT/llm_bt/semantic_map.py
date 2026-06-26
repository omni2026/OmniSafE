"""
semantic_map.py

Parses LLM-BT SemanticMap XML (and extended household format) and provides
initial world state, object location lookup, and location-type queries needed
by the BT expansion engine and LLM condition evaluator.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Location:
    name: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    objects: List[Dict[str, Any]] = field(default_factory=list)
    location_type: str = ""
    state: str = ""

    def object_names(self) -> List[str]:
        return [obj["name"] for obj in self.objects if "name" in obj]


@dataclass
class SemanticMap:
    locations: Dict[str, Location] = field(default_factory=dict)

    def find_object_location(self, obj_name: str) -> Optional[str]:
        for loc_name, loc in self.locations.items():
            if any(o.get("name") == obj_name for o in loc.objects):
                return loc_name
        return None

    def find_object(self, obj_name: str) -> Optional[Dict[str, Any]]:
        for loc in self.locations.values():
            for obj in loc.objects:
                if obj.get("name") == obj_name:
                    return obj
        return None

    def find_location_of_type(self, loc_type: str) -> Optional[str]:
        for loc_name, loc in self.locations.items():
            if loc.location_type == loc_type:
                return loc_name
        if loc_type in self.locations:
            return loc_type
        for loc_name in self.locations:
            if loc_type in loc_name.lower():
                return loc_name
        return None

    def get_all_object_names(self) -> List[str]:
        names: List[str] = []
        for loc in self.locations.values():
            names.extend(loc.object_names())
        return names

    def get_all_location_names(self) -> List[str]:
        return list(self.locations.keys())

    def get_initial_state(self) -> Dict[str, bool]:
        state: Dict[str, bool] = {}
        for loc_name, loc in self.locations.items():
            state[f"At({loc_name})"] = False
            state[f"Near({loc_name})"] = False
            state[f"ExistPath({loc_name})"] = True
            state[f"Approach({loc_name})"] = True
            state[f"Open({loc_name})"] = loc.state == "open"
            state[f"Closed({loc_name})"] = loc.state == "closed"
            for obj_info in loc.objects:
                obj_name = obj_info.get("name", "")
                if obj_name:
                    state[f"On({obj_name},{loc_name})"] = True
                    state[f"Holding({obj_name})"] = False
                    state[f"Clean({obj_name})"] = False
                    state[f"Cooked({obj_name})"] = False
                    state[f"Sliced({obj_name})"] = False
                    props = obj_info.get("property", "")
                    if isinstance(props, str):
                        for prop in props.split(","):
                            prop = prop.strip()
                            if prop == "cooked":
                                state[f"Cooked({obj_name})"] = True
                            elif prop == "clean":
                                state[f"Clean({obj_name})"] = True
                            elif prop == "sliced":
                                state[f"Sliced({obj_name})"] = True
                    elif isinstance(props, list):
                        for prop in props:
                            if prop == "cooked":
                                state[f"Cooked({obj_name})"] = True
                            elif prop == "clean":
                                state[f"Clean({obj_name})"] = True
                            elif prop == "sliced":
                                state[f"Sliced({obj_name})"] = True
        state["HandEmpty"] = True
        return state

    def get_state_description(self) -> str:
        parts: List[str] = []
        for loc_name, loc in self.locations.items():
            parts.append(f"{loc_name}:")
            if loc.objects:
                for obj_info in loc.objects:
                    name = obj_info.get("name", "?")
                    props = obj_info.get("property", "")
                    if props:
                        parts.append(f"  - {name} ({props})")
                    else:
                        parts.append(f"  - {name}")
            else:
                parts.append(f"  (empty)")
        return "\n".join(parts)

    def get_objects_at(self, loc_name: str) -> List[str]:
        loc = self.locations.get(loc_name)
        if loc is None:
            return []
        return loc.object_names()

    def get_container_names(self) -> List[str]:
        return [
            loc_name for loc_name, loc in self.locations.items()
            if loc.location_type in ("fridge", "cabinet", "oven", "microwave", "container", "drawer")
            or loc.state in ("open", "closed")
        ]


def parse_semantic_map_xml(xml_str: str) -> SemanticMap:
    xml_str = xml_str.strip()
    import re as _re
    xml_str = _re.sub(r'<\?xml[^?]*\?>', '', xml_str).strip()
    xml_str = _re.sub(r'(\w+)=(\d+)(?=[\s/>])', r'\1="\2"', xml_str)
    if not xml_str.startswith('<'):
        xml_str = f'<SemanticMapRoot>{xml_str}</SemanticMapRoot>'
    elif not (xml_str.startswith('<root') or xml_str.startswith('<BehaviorTree') or xml_str.startswith('<SemanticMapRoot')):
        xml_str = f'<SemanticMapRoot>{xml_str}</SemanticMapRoot>'
    root = ET.fromstring(xml_str)
    sm = SemanticMap()

    _COORD_TAGS = {"x", "y", "z"}
    _SKIP_TAGS = {"BehaviorTree", "root", "SemanticMapRoot"}
    _LOCATION_TYPE_TAGS = {"fridge", "cabinet", "oven", "microwave", "container", "drawer",
                           "kitchen_counter", "dining_table", "counter", "stove", "sink",
                           "bed", "couch", "sofa"}

    for child in root:
        tag = child.tag
        if tag in _SKIP_TAGS:
            continue

        loc_name = tag
        loc = Location(name=loc_name)

        if tag in _LOCATION_TYPE_TAGS or any(
            t in tag.lower() for t in ("fridge", "cabinet", "oven", "counter", "table", "shelf", "desk", "bar", "sink", "stove")
        ):
            loc.location_type = tag

        for sub in child:
            if sub.tag in ("x", "y", "z"):
                coord = sub.tag
                val = float(sub.text or "0")
                setattr(loc, coord, val)
            elif sub.tag == "obj":
                obj_info: Dict[str, Any] = {"name": ""}
                for obj_child in sub:
                    if obj_child.tag == "name":
                        obj_info["name"] = obj_child.text or ""
                    elif obj_child.tag == "property":
                        obj_info["property"] = obj_child.text or ""
                    else:
                        obj_info[obj_child.tag] = obj_child.text or ""
                if obj_info.get("name"):
                    loc.objects.append(obj_info)
            elif sub.tag == "state":
                loc.state = sub.text or ""
            elif sub.tag == "type":
                loc.location_type = sub.text or ""
            elif sub.tag.startswith("layer"):
                layer_num = sub.attrib.get("L", sub.tag.replace("layer", ""))
                for obj_sub in sub:
                    if obj_sub.tag == "obj":
                        obj_info = {"name": ""}
                        for obj_child in obj_sub:
                            if obj_child.tag == "name":
                                obj_info["name"] = obj_child.text or ""
                            elif obj_child.tag == "property":
                                obj_info["property"] = obj_child.text or ""
                        if loc_name == "shelf":
                            obj_info["name"] = obj_info.get("name", "") + "," + f"L{layer_num}"
                        if obj_info.get("name"):
                            loc.objects.append(obj_info)

        sm.locations[loc_name] = loc

    return sm


def parse_semantic_map_file(xml_path: str) -> SemanticMap:
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"Semantic map file not found: {xml_path}")
    xml_str = path.read_text(encoding="utf-8")
    return parse_semantic_map_xml(xml_str)


def build_semantic_map_from_context(context: Dict[str, Any]) -> SemanticMap:
    sm = SemanticMap()
    metadata = context.get("metadata", {})

    vis_objs = (
        metadata.get("vis_objs")
        or metadata.get("visible_objects")
        or context.get("vis_objs")
        or []
    )
    scene_description = metadata.get("scene_description", "")

    known_locations = {
        "kitchen_counter", "dining_table", "fridge", "cabinet",
        "shelf", "stove", "sink", "table", "bar", "desk",
        "floor", "bed", "couch", "sofa", "oven", "microwave",
    }

    for loc_name in known_locations:
        if loc_name in scene_description.lower() or loc_name in " ".join(str(o) for o in vis_objs).lower():
            loc_type = loc_name
            if loc_name in ("fridge", "cabinet", "oven", "microwave", "drawer"):
                loc_type = loc_name
                sm.locations[loc_name] = Location(name=loc_name, location_type=loc_type, state="closed")
            else:
                sm.locations[loc_name] = Location(name=loc_name, location_type=loc_name)

    for obj_name in vis_objs:
        if isinstance(obj_name, str):
            obj_found = False
            for loc_name, loc in sm.locations.items():
                if not obj_found:
                    loc.objects.append({"name": obj_name})
                    obj_found = True
            if not obj_found:
                default_loc = Location(name="table", location_type="table")
                default_loc.objects.append({"name": obj_name})
                sm.locations["table"] = default_loc

    return sm