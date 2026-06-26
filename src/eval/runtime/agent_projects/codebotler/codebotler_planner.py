"""
CodeBotler Standalone Planner
=============================
独立的高层规划器，完整保留 CodeBotler 的代码生成能力。
与原始 codebotler.py 解耦：不依赖 WebSocket/HTTP/ROS，仅负责
「自然语言指令 → 完整 Python 程序」的规划生成。

核心机制：
  - Chat 模式：structured messages（system + few-shot user/assistant）
  - Completion 模式：prefix + instruction + suffix 拼接
  - 支持 pick/place 扩展 API 集合

参考原始文件：
  - codebotler.py: generate_code() 函数
  - code_generation/openai_chat_completion_prefix.py: chat 模式 prompt
  - code_generation/openai_chat_completion_prefix_minimal.py: 精简版 chat prompt
  - code_generation/prompt_prefix.py: completion 模式 prompt (含 pick/place)
  - code_generation/prompt_prefix_minimal.py: 精简版 completion prompt
  - code_generation/prompt_suffix.py: completion 后缀
  - models/OpenAIChatModel.py: chat 模型调用
"""

import os
import re
import time
import copy
from typing import Dict, List, Optional, Any
from openai import OpenAI

from runtime.planning.reasoning_utils import extract_chat_completion_trace


REASONING_OUTPUT_INSTRUCTION = (
    "Before the code, output exactly one line starting with 'reasoning: ' "
    "briefly describing your task plan."
)
_REASONING_PREFIX_RE = re.compile(r'^\s*reasoning\s*:\s*(.+?)\s*$', re.I)


def split_reasoning_output(content: str) -> tuple[str, str]:
    """Remove the optional reasoning line before Python post-processing."""
    text = str(content or '')
    lines = text.splitlines()
    first_content_idx = next(
        (idx for idx, line in enumerate(lines) if line.strip()),
        None,
    )
    if first_content_idx is not None:
        match = _REASONING_PREFIX_RE.match(lines[first_content_idx])
        if match:
            return (
                match.group(1).strip(),
                '\n'.join(lines[first_content_idx + 1:]).strip(),
            )
    return '', text.strip()


# ============================================================================
# API Definitions — CodeBotler 原子动作集合
# ============================================================================

# 基础 API（不含 pick/place，对应 openai_chat_completion_prefix_minimal）
BASIC_APIS = {
    'get_current_location': {
        'sig': 'get_current_location() -> str',
        'desc': 'Get the current location of the robot.',
    },
    'get_all_rooms': {
        'sig': 'get_all_rooms() -> list[str]',
        'desc': 'Get a list of all rooms.',
    },
    'is_in_room': {
        'sig': 'is_in_room(object: str) -> bool',
        'desc': 'Check if an object is in the current room.',
    },
    'go_to': {
        'sig': 'go_to(location: str) -> None',
        'desc': 'Go to a specific named location, e.g. go_to("kitchen"), go_to("Arjun\'s office").',
    },
    'ask': {
        'sig': 'ask(person: str, question: str, options: list[str]) -> str',
        'desc': 'Ask a person a question, and offer a set of specific options for the person to respond. Returns the response.',
    },
    'say': {
        'sig': 'say(message: str) -> None',
        'desc': 'Say the message out loud.',
    },
}

# 扩展 API（含 pick/place，对应 prompt_prefix.py）
EXTENDED_APIS = {
    **BASIC_APIS,
    'pick': {
        'sig': 'pick(obj: str) -> None',
        'desc': 'Pick up an object if you are not already holding one. You can only hold one object at a time.',
    },
    'place': {
        'sig': 'place(obj: str) -> None',
        'desc': 'Place an object down if you are holding one.',
    },
}


# ============================================================================
# Prompt Templates — 完整保留原始 CodeBotler 的 few-shot 示例
# ============================================================================

# --- Chat 模式：完整版 few-shot messages（含 pick/place）---
CHAT_MESSAGES_FULL: List[Dict[str, str]] = [
    {
        "role": "system",
        "content": (
            '"""Robot task programs.\n\n'
            'Robot task programs may use the following functions:\n'
            'get_current_location()\nget_all_rooms()\nis_in_room()\n'
            'go_to(location)\nask(person, question, options)\nsay(message)\n'
            'pick(object)\nplace(object)\n\n'
            'Robot tasks are defined in named functions, with docstrings describing the task.\n"""\n'
            '# Get the current location of the robot.\n'
            'def get_current_location() -> str:\n  ...\n\n'
            '# Get a list of all rooms.\n'
            'def get_all_rooms() -> list[str]:\n  ...\n\n'
            '# Check if an object is in the current room.\n'
            'def is_in_room(object : str) -> bool:\n  ...\n\n'
            '# Go to a specific named location, e.g. go_to("kitchen"), go_to("Arjun\'s office").\n'
            'def go_to(location : str) -> None:\n  ...\n\n'
            '# Ask a person a question, and offer a set of specific options for the person to respond. '
            'Returns the response selected by the person.\n'
            'def ask(person : str, question : str, options: list[str]) -> str:\n  ...\n\n'
            '# Say the message out loud.\n'
            'def say(message : str) -> None:\n  ...\n\n'
            '# Pick up an object if you are not already holding one. You can only hold one object at a time.\n'
            'def pick(obj: str) -> None:\n  ...\n\n'
            '# Place an object down if you are holding one.\n'
            'def place(obj: str) -> None:\n  ...\n'
        ),
    },
    {
        "role": "user",
        "content": "Go to Arjun's office, ask him if he is ready to head out, and come back and tell me what he said",
    },
    {
        "role": "assistant",
        "content": (
            'def task_program():\n'
            '    start_loc = get_current_location()\n'
            '    go_to("Arjun\'s office")\n'
            '    response = ask("Arjun", "Are you ready to go?", ["Yes", "No"])\n'
            '    go_to(start_loc)\n'
            '    say("Arjun said: " + response)\n'
        ),
    },
    {
        "role": "user",
        "content": (
            "Ask Alice if she needs 1, 2, or 3 boxes. Go to the storage room and ask if they have "
            "that many boxes. If so, go place the boxes in Alice's office. Otherwise, tell Alice "
            "you could not get the boxes."
        ),
    },
    {
        "role": "assistant",
        "content": (
            'def task_program():\n'
            '    go_to("Alice\'s office")\n'
            '    num_boxes = ask("Alice", "How many boxes do you need?", ["1", "2", "3"])\n'
            '    go_to("storage room")\n'
            '    response = ask("", "Do you have " + num_boxes + " boxes?", ["Yes", "No"])\n'
            '    if response == "Yes":\n'
            '        for _ in range(int(num_boxes)):\n'
            '            pick("box")\n'
            '            go_to("Alice\'s office")\n'
            '            place("box")\n'
            '            go_to("storage room")\n'
            '    else:\n'
            '        go_to("Alice\'s office")\n'
            '        say("I could not get the boxes")\n'
        ),
    },
    {
        "role": "user",
        "content": (
            "Check if there is a red marker in the main office, and if so, tell Eve that there is a marker there. "
            "If not, go to the supply room and bring a red marker to the main office."
        ),
    },
    {
        "role": "assistant",
        "content": (
            'def task_program():\n'
            '    go_to("main office")\n'
            '    red_marker_found = is_in_room("red marker")\n'
            '    if red_marker_found:\n'
            '        go_to("Eve\'s office")\n'
            '        say("There is a red marker in the main office")\n'
            '    else:\n'
            '        go_to("supply room")\n'
            '        pick("red marker")\n'
            '        go_to("main office")\n'
            '        place("red marker")\n'
        ),
    },
    {
        "role": "user",
        "content": (
            "Check all classrooms if there is a whiteboard. Go to Aiden's office to tell him which room does not have "
            "a whiteboard. Finally, come back and tell me task is completed."
        ),
    },
    {
        "role": "assistant",
        "content": (
            'def task_program():\n'
            '    start_loc = get_current_location()\n'
            '    list_of_rooms = get_all_rooms()\n'
            '    room_without_whiteboard = []\n'
            '    for room in list_of_rooms:\n'
            '        if "classrooms" not in room:\n'
            '            continue\n'
            '        go_to(room)\n'
            '        if not is_in_room("whiteboard"):\n'
            '            room_without_whiteboard.append(room)\n'
            '    go_to("Aiden\'s office")\n'
            '    if len(room_without_whiteboard) > 0:\n'
            '        message = ""\n'
            '        for room in room_without_whiteboard:\n'
            '            message += room + ", "\n'
            '        message += "do not have a whiteboard"\n'
            '    else:\n'
            '        message = "all classrooms have a whiteboard"\n'
            '    say(message)\n'
            '    go_to(start_loc)\n'
            '    say("task is completed")\n'
        ),
    },
    {
        "role": "user",
        "content": (
            "Go to the kitchen and wait for someone to show up. When someone shows up, "
            "ask them to open the fridge, then pick up a diet coke. Finally, put the "
            "diet coke in the living room."
        ),
    },
    {
        "role": "assistant",
        "content": (
            'def task_program():\n'
            '    go_to("kitchen")\n'
            '    while True:\n'
            '        if is_in_room("person"):\n'
            '            response = ask("", "Please open the fridge", ["Yes", "No"])\n'
            '            if response == "Yes":\n'
            '                pick("diet coke")\n'
            '                break\n'
            '        time.sleep(1)\n'
            '    go_to("living room")\n'
            '    place("diet coke")\n'
        ),
    },
    {
        "role": "user",
        "content": "Take a bed sheet from the laundry room and put it in each of the bedrooms.",
    },
    {
        "role": "assistant",
        "content": (
            'def task_program():\n'
            '    start_loc = get_current_location()\n'
            '    list_of_rooms = get_all_rooms()\n'
            '    for room in list_of_rooms:\n'
            '        if "bedroom" not in room:\n'
            '            continue\n'
            '        go_to("laundry room")\n'
            '        pick("bed sheet")\n'
            '        go_to(room)\n'
            '        place("bed sheet")\n'
            '    go_to(start_loc)\n'
        ),
    },
]

# --- Chat 模式：精简版 few-shot messages（不含 pick/place）---
CHAT_MESSAGES_MINIMAL: List[Dict[str, str]] = [
    CHAT_MESSAGES_FULL[0],  # system message 会被单独替换
    CHAT_MESSAGES_FULL[1],  # 第一个 few-shot
    CHAT_MESSAGES_FULL[2],  # 第一个 few-shot answer
]

# --- Completion 模式：完整版 prompt prefix（含 pick/place）---
COMPLETION_PREFIX_FULL: str = (
    '"""Robot task programs.\n\n'
    'Robot task programs may use the following functions:\n'
    'get_current_location()\nget_all_rooms()\nis_in_room()\n'
    'go_to(location)\nask(person, question, options)\nsay(message)\n'
    'pick(object)\nplace(object)\n\n'
    'Robot tasks are defined in named functions, with docstrings describing the task.\n"""\n\n'
    '# Get the current location of the robot.\n'
    'def get_current_location() -> str:\n    ...\n\n'
    '# Get a list of all rooms.\n'
    'def get_all_rooms() -> list[str]:\n    ...\n\n'
    '# Check if an object is in the current room.\n'
    'def is_in_room(object : str) -> bool:\n    ...\n\n'
    '# Go to a specific named location, e.g. go_to("kitchen"), go_to("Arjun\'s office").\n'
    'def go_to(location : str) -> None:\n    ...\n\n'
    '# Ask a person a question, and offer a set of specific options for the person to respond. '
    'Returns the response selected by the person.\n'
    'def ask(person : str, question : str, options: list[str]) -> str:\n    ...\n\n'
    '# Say the message out loud.\n'
    'def say(message : str) -> None:\n    ...\n\n'
    '# Pick up an object if you are not already holding one. You can only hold one object at a time.\n'
    'def pick(obj: str) -> None:\n    ...\n\n'
    '# Place an object down if you are holding one.\n'
    'def place(obj: str) -> None:\n    ...\n\n'
    '# Go to Arjun\'s office, ask him if he is ready to head out, and come back and tell me what he said\n'
    'def task_program():\n'
    '    start_loc = get_current_location()\n'
    '    go_to("Arjun\'s office")\n'
    '    response = ask("Arjun", "Are you ready to go?", ["Yes", "No"])\n'
    '    go_to(start_loc)\n'
    '    say("Arjun said: " + response)\n\n'
    '# Ask Alice if she needs 1, 2, or 3 boxes. Go to the storage room and ask if they have that many boxes. '
    'If so, go place the boxes in Alice\'s office. Otherwise, tell Alice you could not get the boxes.\n'
    'def task_program():\n'
    '    go_to("Alice\'s office")\n'
    '    num_boxes = ask("Alice", "How many boxes do you need?", ["1", "2", "3"])\n'
    '    go_to("storage room")\n'
    '    response = ask("", "Do you have" + num_boxes + " boxes?", ["Yes", "No"])\n'
    '    if response == "Yes":\n'
    '        for _ in range(int(num_boxes)):\n'
    '            pick("box")\n'
    '            go_to("Alice\'s office")\n'
    '            place("box")\n'
    '            go_to("storage room")\n'
    '    else:\n'
    '        go_to("Alice\'s office")\n'
    '        say("I could not get the boxes")\n\n'
    '# Check if there is a red marker in the main office, and if so, tell Eve that there is a marker there. '
    'If not, go to the supply room and bring a red marker to the main office.\n'
    'def task_program():\n'
    '    go_to("main office")\n'
    '    red_marker_found = is_in_room("red marker")\n'
    '    if red_marker_found:\n'
    '        go_to("Eve\'s office")\n'
    '        say("There is a red marker in the main office")\n'
    '    else:\n'
    '        go_to("supply room")\n'
    '        pick("red marker")\n'
    '        go_to("main office")\n'
    '        place("red marker")\n\n'
    '# Check every classroom if there is a whiteboard. Go to Aiden\'s office to tell him which room does not have '
    'a whiteboard. Come back and tell me task is completed.\n'
    'def task_program():\n'
    '    start_loc = get_current_location()\n'
    '    list_of_rooms = get_all_rooms()\n'
    '    room_without_whiteboard = []\n'
    '    for room in list_of_rooms:\n'
    '        if "classroom" not in room:\n'
    '            continue\n'
    '        go_to(room)\n'
    '        if not is_in_room("whiteboard"):\n'
    '            room_without_whiteboard.append(room)\n'
    '    go_to("Aiden\'s office")\n'
    '    if len(room_without_whiteboard) > 0:\n'
    '        message = ""\n'
    '        for room in room_without_whiteboard:\n'
    '            message += room + ", "\n'
    '        message += "do not have a whiteboard"\n'
    '    else:\n'
    '        message = "all classrooms have a whiteboard"\n'
    '    say(message)\n'
    '    go_to(start_loc)\n'
    '    say("task is completed")\n\n'
    '# Go to the kitchen and wait for someone to show up. When someone shows up, ask them to open the fridge, '
    'then pick up a diet coke. Finally, put the diet coke in the living room.\n'
    'def task_program():\n'
    '    go_to("kitchen")\n'
    '    while True:\n'
    '        if is_in_room("person"):\n'
    '            response = ask("", "Please open the fridge", ["Yes", "No"])\n'
    '            if response == "Yes":\n'
    '                pick("diet coke")\n'
    '                break\n'
    '        time.sleep(1)\n'
    '    go_to("living room")\n'
    '    place("diet coke")\n\n'
    '# Take a bed sheet from the laundry room and put it in each of the bedrooms.\n'
    'def task_program():\n'
    '    start_loc = get_current_location()\n'
    '    list_of_rooms = get_all_rooms()\n'
    '    for room in list_of_rooms:\n'
    '        if "bedroom" not in room:\n'
    '            continue\n'
    '        go_to("laundry room")\n'
    '        pick("bed sheet")\n'
    '        go_to(room)\n'
    '        place("bed sheet")\n'
    '    go_to(start_loc)\n\n'
    '# '
)

# --- Completion 模式：精简版 prompt prefix（不含 pick/place）---
COMPLETION_PREFIX_MINIMAL: str = (
    '"""Robot task programs.\n\n'
    'Robot task programs may use the following functions:\n'
    'get_current_location()\nget_all_rooms()\nis_in_room()\n'
    'go_to(location)\nask(person, question, options)\nsay(message)\n\n'
    'Robot tasks are defined in named functions, with docstrings describing the task.\n"""\n\n'
    '# Get the current location of the robot.\n'
    'def get_current_location() -> str:\n    ...\n\n'
    '# Get a list of all rooms.\n'
    'def get_all_rooms() -> list[str]:\n    ...\n\n'
    '# Check if an object is in the current room.\n'
    'def is_in_room(object : str) -> bool:\n    ...\n\n'
    '# Go to a specific named location, e.g. go_to("kitchen"), go_to("Arjun\'s office").\n'
    'def go_to(location : str) -> None:\n    ...\n\n'
    '# Ask a person a question, and offer a set of specific options for the person to respond. '
    'Returns the response selected by the person.\n'
    'def ask(person : str, question : str, options: list[str]) -> str:\n    ...\n\n'
    '# Say the message out loud.\n'
    'def say(message : str) -> None:\n    ...\n\n'
    '# Go to Arjun\'s office, ask him if he is ready to head out, and come back and tell me what he said\n'
    'def task_program():\n'
    '    start_loc = get_current_location()\n'
    '    go_to("Arjun\'s office")\n'
    '    response = ask("Arjun", "Are you ready to go?", ["Yes", "No"])\n'
    '    go_to(start_loc)\n'
    '    say("Arjun said: " + response)\n\n'
    '# '
)

# --- Completion 模式后缀 ---
COMPLETION_SUFFIX: str = '\ndef task_program():\n'


# ============================================================================
# CodeBotlerPlanner
# ============================================================================

class CodeBotlerPlanner:
    """
    CodeBotler 独立规划器。

    完整保留原始 CodeBotler 的代码生成机制，提供两种生成模式：
      - 'chat': 使用 OpenAI Chat Completions API + structured few-shot messages
      - 'completion': 使用文本补全 API + prefix/suffix 拼接

    支持两种 API 集合：
      - 'basic': 6 个基础 API (get_current_location, get_all_rooms, is_in_room, go_to, ask, say)
      - 'extended': 8 个 API (basic + pick, place)

    使用示例：
        >>> from openai import OpenAI
        >>> client = OpenAI(api_key='...', base_url='https://api.deepseek.com')
        >>> planner = CodeBotlerPlanner(client, model_name='deepseek-chat')
        >>> program = planner.plan("put the apple in the fridge")
        >>> print(program)
    """

    # 自动检测 chat 模型的关键词
    _CHAT_KEYWORDS = ['gpt-3.5-turbo', 'gpt-4', 'glm', 'deepseek', 'qwen', 'claude']

    def __init__(
        self,
        client: OpenAI,
        model_name: str = 'gpt-4',
        *,
        mode: str = 'auto',
        api_set: str = 'extended',
        few_shot: str = 'full',
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 512,
        stop_sequences: Optional[List[str]] = None,
        verbose: bool = False,
        use_reasoning_prompt: bool = False,
    ):
        """
        Args:
            client: OpenAI 兼容客户端实例
            model_name: 模型名称
            mode: 生成模式，'chat' / 'completion' / 'auto'（自动根据 model_name 推断）
            api_set: API 集合，'basic'（6个API）/ 'extended'（8个API，含pick/place）
            few_shot: few-shot 数量，'full'（完整6例） / 'minimal'（精简1例）
            temperature: 生成温度
            top_p: top-p 采样
            max_tokens: 最大生成 token 数
            stop_sequences: 自定义 stop sequences（覆盖默认值）
            verbose: 是否打印调试信息
        """
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.use_reasoning_prompt = bool(use_reasoning_prompt)
        self.last_reasoning = ''
        self.last_traces = []

        # 解析生成模式
        if mode == 'auto':
            self._mode = 'chat' if self._is_chat_model(model_name) else 'completion'
        else:
            self._mode = mode

        # 解析 API 集合
        self._api_set = api_set
        if api_set == 'extended':
            self.apis = dict(EXTENDED_APIS)
        else:
            self.apis = dict(BASIC_APIS)

        # 解析 few-shot 配置
        self._few_shot = few_shot

        # 预构建 prompt 组件
        self._chat_messages = self._build_chat_messages()
        self._completion_prefix = self._build_completion_prefix()

        # Stop sequences
        if stop_sequences is not None:
            self._stop_sequences = list(stop_sequences)
        elif self._mode == 'chat':
            self._stop_sequences = ['\n#', '\nclass', '```']
        else:
            self._stop_sequences = ['\n#', '\nclass', '```', '\ndef']

    @staticmethod
    def _is_chat_model(model_name: str) -> bool:
        """根据模型名推断是否为 chat 模型。"""
        name_lower = model_name.lower()
        return any(kw in name_lower for kw in CodeBotlerPlanner._CHAT_KEYWORDS)

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------

    def _build_chat_messages(self) -> List[Dict[str, str]]:
        """构建 chat 模式的 few-shot messages。"""
        if self._api_set == 'extended':
            base_messages = CHAT_MESSAGES_FULL
        else:
            base_messages = CHAT_MESSAGES_MINIMAL

        if self._few_shot == 'minimal' and len(base_messages) > 3:
            # minimal: 只保留 system + 1个 few-shot
            return self._with_reasoning_output_contract(base_messages[:3])

        return self._with_reasoning_output_contract(base_messages)

    def _with_reasoning_output_contract(
        self,
        base_messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        messages = copy.deepcopy(base_messages)
        if self.use_reasoning_prompt and messages:
            messages[0]['content'] = (
                str(messages[0].get('content', '')).rstrip()
                + '\n\n'
                + REASONING_OUTPUT_INSTRUCTION
            )
        return messages

    def _build_completion_prefix(self) -> str:
        """构建 completion 模式的 prompt prefix。"""
        if self._api_set == 'extended':
            if self._few_shot == 'minimal':
                # 从完整版截取到第一个 few-shot 结束
                return COMPLETION_PREFIX_MINIMAL.replace(
                    'pick(object)\nplace(object)\n', ''
                )
            return COMPLETION_PREFIX_FULL
        else:
            if self._few_shot == 'minimal':
                return COMPLETION_PREFIX_MINIMAL
            # basic + full: 用 full 版本但去掉 pick/place 相关的 few-shot
            return COMPLETION_PREFIX_FULL  # 保留所有 few-shot，只是 API 定义中不含 pick/place

    def _build_api_signature_block(self) -> str:
        """生成 API 签名说明文本（用于注入 system message 或额外上下文）。"""
        lines = []
        for name, api in self.apis.items():
            lines.append(f"# {api['desc']}")
            lines.append(f"def {api['sig']}:")
            lines.append("  ...")
            lines.append("")
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # 代码生成
    # ------------------------------------------------------------------

    def plan(self, instruction: str, context: str = '') -> str:
        """
        根据自然语言指令生成完整的 Python 程序。

        Args:
            instruction: 自然语言指令（如 "put the apple in the fridge"）
            context: 额外上下文（如场景描述、可用对象列表等）

        Returns:
            完整的 Python 程序字符串（含头部注释 + API 说明 + 生成代码）
        """
        raw_code, assembled = self.plan_components(instruction, context)
        return assembled

    def plan_components(self, instruction: str, context: str = '') -> Dict[str, Any]:
        """
        生成代码并分别返回 LLM 原始输出与组装后的完整程序。

        与 plan() 不同，本方法把两段内容分开返回，便于上层（如评测 adapter）
        将 LLM 原文存入 raw_output，把组装程序另存到 metadata。

        Args:
            instruction: 自然语言指令
            context: 额外上下文

        Returns:
            dict with keys:
              - 'raw_code': LLM 原始输出经 _postprocess 清洗后的代码
                            （去 markdown 围栏、补 def task_program，但不含
                             头部注释 / API 说明等组装内容）
              - 'assembled': _assemble_program 组装后的完整程序字符串
                             （即原 plan() 的返回值）
        """
        start_time = time.time()
        self.last_reasoning = ''
        self.last_traces = []

        if self._mode == 'chat':
            code = self._generate_chat(instruction, context)
        else:
            code = self._generate_completion(instruction, context)

        if self.use_reasoning_prompt:
            reasoning, code = split_reasoning_output(code)
            if reasoning:
                self._record_reasoning(reasoning)
                if self.last_traces:
                    self.last_traces[-1]['reasoning_content'] = reasoning
                    self.last_traces[-1]['reasoning_field_source'] = 'prompt_prefix'

        elapsed = time.time() - start_time
        if self.verbose:
            print(f'[CodeBotlerPlanner] Generation time: {elapsed:.2f}s')

        # 后处理：确保代码以 def task_program() 开头
        raw_code = self._postprocess(code)

        # 组装完整程序
        assembled = self._assemble_program(instruction, raw_code, context)

        return {
            'raw_code': raw_code,
            'assembled': assembled,
            'reasoning': self.last_reasoning,
            'llm_trace': list(self.last_traces),
        }

    def _record_reasoning(self, reasoning: Any) -> None:
        text = str(reasoning or '').strip()
        if not text:
            return
        if not self.last_reasoning:
            self.last_reasoning = text
        elif text not in self.last_reasoning.split('\n\n'):
            self.last_reasoning += '\n\n' + text

    def _generate_chat(self, instruction: str, context: str = '') -> str:
        """Chat 模式生成。"""
        messages = list(self._chat_messages)

        # 如果有额外上下文，注入到最后一个 system message 或添加新 context
        if context:
            messages.append({
                "role": "system",
                "content": f"Context: {context}",
            })

        # 添加用户指令
        messages.append({"role": "user", "content": instruction})

        if self.verbose:
            print(f'[CodeBotlerPlanner] Chat mode, {len(messages)} messages')

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_tokens,
            stop=self._stop_sequences,
        )

        trace = extract_chat_completion_trace(
            response,
            prompt='\n'.join(
                str(message.get('content', '') or '')
                for message in messages
            ),
            model=self.model_name,
        )
        trace['label'] = 'code_generation'
        trace['attempt'] = len(self.last_traces)
        self.last_traces.append(trace)
        self._record_reasoning(trace.get('reasoning_content'))

        if self.verbose and response.usage:
            print(
                f'[CodeBotlerPlanner] Tokens: input={response.usage.prompt_tokens}, '
                f'output={response.usage.completion_tokens}'
            )

        return str(trace.get('content') or '').strip()

    def _generate_completion(self, instruction: str, context: str = '') -> str:
        """Completion 模式生成。"""
        prompt = self._completion_prefix
        if context:
            prompt += f'\n{context}\n'
        if self.use_reasoning_prompt:
            prompt += f'\n{REASONING_OUTPUT_INSTRUCTION}\n'
        prompt += instruction + COMPLETION_SUFFIX

        if self.verbose:
            print(f'[CodeBotlerPlanner] Completion mode, prompt length={len(prompt)}')

        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self._stop_sequences,
        )

        code = response.choices[0].text.strip()
        if self.use_reasoning_prompt:
            reasoning, code = split_reasoning_output(code)
            if reasoning:
                self.last_reasoning = reasoning
        # completion 模式需要在前面加上 suffix
        code = COMPLETION_SUFFIX.strip() + '\n' + code
        return code

    def _postprocess(self, code: str) -> str:
        """后处理生成的代码。"""
        # 清理 markdown 代码块标记
        code = re.sub(r'^```python\s*\n', '', code)
        code = re.sub(r'^```\s*\n', '', code)
        code = re.sub(r'\n```\s*$', '', code)

        # 如果代码没有以 def task_program() 开头，补上
        if 'def task_program' not in code:
            code = 'def task_program():\n    ' + code.replace('\n', '\n    ')

        return code.strip()

    def _assemble_program(self, instruction: str, code: str, context: str = '') -> str:
        """将生成的代码组装为完整程序。"""
        lines = []

        # 头部注释
        lines.append('# ' + '=' * 60)
        lines.append('# CodeBotler Generated Program')
        lines.append(f'# Instruction: "{instruction}"')
        if context:
            lines.append(f'# Context: {context}')
        lines.append('# ' + '=' * 60)
        lines.append('')

        # API 依赖声明
        lines.append('# --- Available APIs ---')
        for name, api in self.apis.items():
            lines.append(f'#   {api["sig"]}: {api["desc"]}')
        lines.append('')

        # 生成的代码
        lines.append('# --- Generated Program ---')
        lines.append('')
        lines.append(code)
        lines.append('')

        return '\n'.join(lines)


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == '__main__':
    import os

    # 从环境变量获取 API key
    api_key = os.getenv('DEEPSEEK_API_KEY', '')
    base_url = 'https://api.deepseek.com'
    model_name = 'deepseek-v4-pro'

    if not api_key:
        # 尝试其他 provider
        api_key = os.getenv('OPENAI_API_KEY', '')
        base_url = None
        model_name = 'gpt-4'

    if not api_key:
        print('Please set DEEPSEEK_API_KEY or OPENAI_API_KEY environment variable.')
        exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    # --- Chat 模式 + Extended API ---
    print('=' * 70)
    print('CodeBotler Planner - Chat Mode + Extended API')
    print('=' * 70)

    planner = CodeBotlerPlanner(
        client=client,
        model_name=model_name,
        mode='chat',
        api_set='extended',
        few_shot='full',
        temperature=0.2,
        verbose=True,
    )

    program = planner.plan(
        instruction='put the apple in the fridge',
        context='objects = ["apple", "banana", "milk", "fridge", "cabinet"]',
    )
    print(program)

    # print('\n' + '=' * 70)
    # print('CodeBotler Planner - Chat Mode + Basic API')
    # print('=' * 70)

    # planner_basic = CodeBotlerPlanner(
    #     client=client,
    #     model_name=model_name,
    #     mode='chat',
    #     api_set='basic',
    #     few_shot='minimal',
    #     temperature=0.2,
    #     verbose=True,
    # )

    # program2 = planner_basic.plan(
    #     instruction='go to the kitchen and tell me what you see',
    # )
    # print(program2)
