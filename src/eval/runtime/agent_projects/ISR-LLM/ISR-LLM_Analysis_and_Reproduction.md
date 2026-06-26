# ISR-LLM：代码实现深度分析与 Household Planning Agent 复现指南

> 论文：*ISR-LLM: Iterative Self-Refined Large Language Model for Long-Horizon Sequential Task Planning*
> 发表：ICRA 2024 | 代码仓库：`github.com/ma-labo/ISR-LLM`

---

## 一、总体架构

ISR-LLM 是一个三阶段的 LLM 规划框架，核心思想是通过 **迭代自精化（Iterative Self-Refinement）** 提升 LLM 生成任务规划的成功率。

```
自然语言任务描述
       │
       ▼
┌─────────────────────────────────┐
│  Step 1: Preprocessing          │
│  LLM Translator (NL → PDDL)     │  few-shot in-context learning
│  输出: PDDL problem (:init/:goal)│
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Step 2: Planning               │
│  LLM Planner (PDDL → Actions)   │  few-shot in-context learning
│  输出: action sequence           │
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  Step 3: Iterative Self-Refinement Loop (max_num_refine) │
│                                                          │
│   ┌──────────────────┐    ┌────────────────────────┐     │
│   │  Self-Validator  │ OR │  External Validator    │     │
│   │  (LLM Validator) │    │  (Domain Simulator)    │     │
│   └────────┬─────────┘    └────────────┬───────────┘     │
│            │                           │                 │
│            └──────── Feedback ─────────┘                 │
│                          │                               │
│            ┌─────────────▼────────────┐                  │
│            │   LLM Planner re-query   │                  │
│            │   (is_append=True,       │                  │
│            │    温度随尝试次数递增)      │                  │
│            └──────────────────────────┘                  │
└──────────────────────────────────────────────────────────┘
```

---

## 二、仓库结构解析

```
ma-labo/ISR-LLM/
├── main.py                      # 主入口：参数解析、流程编排
├── LLM/
│   ├── Translator/Translator.py # LLM Translator：NL → PDDL 翻译器
│   ├── Planner/Planner.py       # LLM Planner：规划生成器
│   └── Validator/Validator.py   # LLM Validator：自验证器
├── Blocks_Sim/Block_Sim.py      # Blocksworld 外部仿真验证器
├── BallMoving_Sim/BallMoving_Sim.py
├── Cooking_Sim/Cooking_Sim.py   # ← Household 最相近的参考域
├── utils/
│   ├── utils.py                 # load_test_scenarios, extract_state_pddl 等工具函数
│   └── generate_*.py            # 随机场景生成器
└── test_scenario/               # 预存的测试场景 .npy 文件
```

---

## 三、核心组件深度分析

### 3.1 LLM Translator（预处理模块）

**功能**：将自然语言场景描述转换为 PDDL problem 格式（`:init` 和 `:goal` 两段）。

```python
# main.py 中调用方式
response_translator = LLM_Translator.query(description, is_append=False)
planning_problem = response_translator["choices"][0]["message"]["content"]
```

**Prompt 结构**（few-shot in-context learning）：
```
System: You are a translator that converts natural language descriptions of planning 
        problems into PDDL format...

User (example 1): [场景的 NL 描述]
Assistant (example 1): 
(:init
  (on blockA blockB)
  (ontable blockC)
  ...
)
(:goal
  (and (on blockC blockA) ...)
)

User (actual): [当前场景的 NL 描述]
```

**关键设计点**：
- `is_append=False`：每次翻译都是独立调用，不携带历史消息
- Few-shot examples 数量由 `--num_trans_example` 控制（1/2/3个）
- Domain knowledge（可用对象/谓词）被预嵌入到 few-shot examples 中

---

### 3.2 LLM Planner（规划模块）

**功能**：接收 PDDL 问题描述，输出具体动作序列。

```python
# 首次调用：is_append=True，将 planning_problem 作为新消息追加
response_planner = LLM_Planner.query(planning_problem, is_append=True, temperature=temperature)
action_sequence = response_planner["choices"][0]["message"]["content"]
```

**温度控制策略**：
```python
if i == 0:
    temperature = 0               # 第一个 test case 用确定性输出
else:
    temperature = min(max_refine_temperature, 0.1 * i)  # 随 attempt 次数缓慢升温
# max_refine_temperature = 0.4
```

**消息状态管理**：
- `is_append=True`：将反馈和 re-planning 请求追加到同一会话，保留对话历史
- `LLM_Planner.init_messages(is_reinitialize=True)`：每个 test case 结束后重置会话

---

### 3.3 External Validator（外部仿真验证器）

这是 `ISR-LLM-external`（即 `LLM_trans_exact_feedback`）方法的核心。

```python
is_satisfied, is_error, error_message, error_action = \
    scenario_simulator.simulate_actions(action_sequence, test_log_file_path)
```

**返回值语义**：

| `is_error` | `is_satisfied` | `error_action` | 含义 |
|:---:|:---:|:---:|:---|
| `True` | `False` | 非空 | 某个 action 执行失败，error_action 指向该动作 |
| `True` | `False` | `None` | LLM 未返回有效动作序列 |
| `True` | `True` | 非空 | 目标提前达成，error_action 之后的动作多余 |
| `None` | - | - | 没有解析出动作 |
| `False` | `True` | - | 所有动作成功，目标满足 → 跳出 refinement loop |

**Feedback 构造逻辑**：
```python
if is_error == True:
    if is_satisfied == False:
        if error_action != None:
            planning_problem = "Action " + error_action + " is wrong. Error info: " + error_message + " Please find a new plan."
        else:
            planning_problem = error_message + " Please find a new plan."
    else:
        planning_problem = error_message + " Please ignore actions after action " + error_action
elif is_error == None:
    planning_problem = "Please find a new plan."
else:
    break  # 成功，退出循环
```

---

### 3.4 Self-Validator（LLM 自验证器）

`ISR-LLM-self`（`LLM_trans_self_feedback`）方法，用 LLM 替代仿真器做验证。

```python
# 构造验证问题
validate_question = "Question:\nBlock initial state:\n" + pddl_init_state + \
                    "\nGoal state:\n" + pddl_goal_state + \
                    "\nExamined action sequence:\n" + action_description

response_validator = LLM_Validator.query(validate_question, is_append=False)
valid_result = response_validator_content.split('Final answer:', 1)
```

**Validator Prompt 期待输出格式**：
```
[step-by-step analysis...]
Final answer: Yes  (or No)
```

解析逻辑：
- `Final answer: Yes` → 自评估通过，break 出 refinement loop
- `Final answer: No` → 自评估失败，用 domain-specific hint 驱动 re-planning
- 解析失败（无 `Final answer:` 段） → 直接 break（保守退出）

---

### 3.5 Domain Simulator 结构（以 Cooking_Sim 为参考）

Simulator 需要实现三个核心接口：

```python
class DomainSim:
    def generate_scene_description(self, initial_state, goal_state, constraint) -> str:
        """生成当前场景的自然语言描述，用于 Translator 输入"""
        pass

    def initialize_state(self, initial_state, goal_state, constraint):
        """重置仿真器到初始状态（每次 refinement attempt 前调用）"""
        pass

    def simulate_actions(self, action_sequence: str, log_file_path: str) -> tuple:
        """
        解析并逐步执行 action_sequence
        返回: (is_satisfied, is_error, error_message, error_action)
        """
        pass
```

---

## 四、五种方法对比

| 方法名 | Translator | Feedback 来源 | 特点 |
|:---|:---:|:---:|:---|
| `LLM_trans_no_feedback` | ✓ PDDL | 无 | 直接规划，max_refine=0 |
| `LLM_trans_self_feedback` | ✓ PDDL | LLM Validator | 轻量，无需仿真器 |
| `LLM_trans_exact_feedback` | ✓ PDDL | External Sim | 精确反馈，最高成功率 |
| `LLM_no_trans` | ✗ NL | External Sim | 无 PDDL，直接 NL 规划 |
| `LLM_no_trans_self_feedback` | ✗ NL | LLM Validator | 纯 LLM pipeline |

**推荐用于 Household**：`LLM_trans_exact_feedback`（最强），或在无仿真器时用 `LLM_trans_self_feedback`。

---

## 五、Household Domain 复现方案

### 5.1 整体适配策略

原始代码的三个 domain（blocksworld/ballmoving/cooking）结构完全平行，Cooking 最接近 household。适配新 household domain 需要：

1. 编写 `Household_Sim/Household_Sim.py`
2. 为 Translator 提供 household 的 few-shot examples（PDDL 格式）
3. 为 Planner 提供 household 的 few-shot examples（动作序列格式）
4. 为 Validator 提供 household 的 few-shot examples（step-by-step 验证格式）
5. 在 `main.py` 中注册新 domain

---

### 5.2 定义 Household PDDL Domain

```pddl
;; household_domain.pddl
(define (domain household)
  (:requirements :strips :typing)
  
  (:types
    robot location object receptacle - entity
    container openable - receptacle
  )
  
  (:predicates
    ;; robot 状态
    (at-robot ?r - robot ?l - location)
    (holding ?r - robot ?o - object)
    (hand-empty ?r - robot)
    
    ;; 物体状态
    (at-object ?o - object ?l - location)
    (in-receptacle ?o - object ?rec - receptacle)
    (on-surface ?o - object ?l - location)
    
    ;; 容器状态
    (is-open ?c - container)
    (is-closed ?c - container)
    
    ;; 物体属性
    (is-pickable ?o - object)
    (is-cookable ?o - object)
    (is-cooked ?o - object)
    (is-sliceable ?o - object)
    (is-sliced ?o - object)
  )
  
  (:action navigate
    :parameters (?r - robot ?from ?to - location)
    :precondition (at-robot ?r ?from)
    :effect (and (at-robot ?r ?to) (not (at-robot ?r ?from)))
  )
  
  (:action pick-up
    :parameters (?r - robot ?o - object ?l - location)
    :precondition (and (at-robot ?r ?l) (at-object ?o ?l) 
                       (hand-empty ?r) (is-pickable ?o))
    :effect (and (holding ?r ?o) (not (at-object ?o ?l)) (not (hand-empty ?r)))
  )
  
  (:action place
    :parameters (?r - robot ?o - object ?l - location)
    :precondition (and (at-robot ?r ?l) (holding ?r ?o))
    :effect (and (at-object ?o ?l) (not (holding ?r ?o)) (hand-empty ?r))
  )
  
  (:action open
    :parameters (?r - robot ?c - container ?l - location)
    :precondition (and (at-robot ?r ?l) (at-object ?c ?l) (is-closed ?c))
    :effect (and (is-open ?c) (not (is-closed ?c)))
  )
  
  (:action close
    :parameters (?r - robot ?c - container ?l - location)
    :precondition (and (at-robot ?r ?l) (at-object ?c ?l) (is-open ?c))
    :effect (and (is-closed ?c) (not (is-open ?c)))
  )
  
  (:action cook
    :parameters (?r - robot ?o - object ?l - location)
    :precondition (and (at-robot ?r ?l) (holding ?r ?o) (is-cookable ?o))
    :effect (is-cooked ?o)
  )
  
  (:action slice
    :parameters (?r - robot ?o - object ?l - location)
    :precondition (and (at-robot ?r ?l) (holding ?r ?o) (is-sliceable ?o))
    :effect (and (is-sliced ?o) (not (is-sliceable ?o)))
  )
)
```

---

### 5.3 实现 Household_Sim

这里不需要，因为本复现仅仅关注Planning层面。
---

### 5.4 Few-Shot Prompt 设计

这是成败关键。参考原始 Cooking 的 prompt，为 Household 写三套 few-shot examples。

#### Translator Few-Shot Example（NL → PDDL）

```
System: You are a PDDL translator. Convert the natural language description of a 
        household task into PDDL :init and :goal sections.
        Available actions: navigate, pick-up, place, open, close, cook, slice.
        Available locations: kitchen_counter, dining_table, fridge, stove, sink.

User:
The robot is at the kitchen counter. An apple is at the kitchen counter. 
A knife is at the dining table. 
Goal: The apple should be sliced and placed on the dining table.


## Translator Few-Shot Output Example

```
(:init
  (at-robot robot kitchen_counter)
  (at-object apple kitchen_counter)
  (at-object knife dining_table)
  (hand-empty robot)
  (is-pickable apple)
  (is-sliceable apple)
)
(:goal
  (and (is-sliced apple) (at-object apple dining_table))
)
```

---

### 5.5 在 main.py 中注册 Household Domain

```python
DOMAINS = ["blocksworld", "ballmoving", "cooking", "household"]

from Household_Sim.Household_Sim import HouseholdSim
if args.domain == 'household':
    scenario_simulator = HouseholdSim()

# self-feedback 分支添加 domain-specific hint:
elif domain == 'household':
    validate_question = ("Question:\nHousehold initial state:\n" + pddl_init_state +
                         "\nGoal state:\n" + pddl_goal_state +
                         "\nExamined action sequence:\n" + action_description)
    planning_problem = error_description + " Re-check each action's preconditions."
```

---

## 六、依赖安装与版本兼容

```bash
git clone https://github.com/ma-labo/ISR-LLM.git
cd ISR-LLM
pip install numpy

# 方案 A: 使用旧版 openai（最省事，与原始代码完全兼容）
pip install openai==0.28.0

# 方案 B: 使用新版 openai >= 1.0.0（推荐长期维护）
pip install openai  # 需改写 Translator/Planner/Validator 的调用接口
```

**新版 openai 接口适配（Translator/Planner/Validator 中统一修改）**：

```python
# 旧版（原始代码）
response = openai.ChatCompletion.create(
    model=self.model, messages=self.messages, temperature=temperature
)
content = response["choices"][0]["message"]["content"]

# 新版（openai >= 1.0.0）
from openai import OpenAI
self.client = OpenAI(api_key="YOUR-KEY")
response = self.client.chat.completions.create(
    model=self.model, messages=self.messages, temperature=temperature
)
content = response.choices[0].message.content
```

---

## 七、运行命令

```bash
# household + external validator（最强，需要仿真器对接）
python3 main.py --domain household --method LLM_trans_exact_feedback \
                --model gpt-4 --num_objects 3

# household + self-validator（轻量，纯 LLM pipeline）
python3 main.py --domain household --method LLM_trans_self_feedback \
                --model gpt-3.5-turbo --num_objects 3

# baseline：无 refinement
python3 main.py --domain household --method LLM_trans_no_feedback \
                --model gpt-3.5-turbo
```

---

## 八、关键参数说明

| 参数 | 默认值 | 说明 |
|:---|:---:|:---|
| `num_test` | 10 | 测试用例数 |
| `num_prompt_examples_dataset` | 3 | 跳过用作 few-shot 的前 n 条数据 |
| `max_num_refine` | 10 | 最大 refinement 轮数（0=无反馈） |
| `gpt_api_wait_time` | 20 | API 调用间隔秒数（防 RPM 超限） |
| `max_refine_temperature` | 0.4 | 最大 temperature，随 attempt 次数递增 |
| `--num_trans_example` | 3 | Translator few-shot 数量 |
| `--num_plan_example` | 4 | Planner few-shot 数量 |
| `--num_valid_example` | 6 | Validator few-shot 数量 |

---

---

## 十、常见问题速查

| 问题 | 原因 | 解决方案 |
|:---|:---|:---|
| `AttributeError: module 'openai' has no attribute 'ChatCompletion'` | openai >= 1.0.0 接口变更 | 按第六节适配新版调用接口 |
| action 序列无法解析 | Planner 输出格式不稳定 | few-shot 中严格约束输出格式；加强 `_parse_action_sequence` |
| Translator PDDL 结构错误 | few-shot examples 质量不足 | 增加 few-shot 数量；System Prompt 添加格式约束 |
| Self-validator 无 `Final answer:` | LLM 输出偏离格式 | Validator Prompt 末尾明确要求 "End with: Final answer: Yes/No" |
| Refinement loop 不收敛 | feedback 不够具体 | error_description 中加入 domain-specific 修正提示 |
| API Rate Limit | gpt_api_wait_time 太短 | 调大等待时间或换更高 tier 的 API key |

---

*分析基于 `github.com/ma-labo/ISR-LLM` (ICRA 2024)，结合 OMNISAFE 项目需求整理。*
