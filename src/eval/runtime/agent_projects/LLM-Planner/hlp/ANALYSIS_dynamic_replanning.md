# 分析：execute_with_dynamic_replanning 的实现

## 1. 核心概念

`execute_with_dynamic_replanning` 是一个**混合型执行引擎**，支持两种工作模式：

### 模式1：自主执行模式 (Autonomous Mode)
- **触发条件**：`action_executor` 和 `visible_objects_provider` 都是可调用的
- **执行流程**：完全由该方法内部循环驱动
- **特点**：同步执行，方法不返回直到所有计划完成或达到重规划限制

### 模式2：外部驱动模式 (External Loop Mode)  
- **触发条件**：`action_executor` 和 `visible_objects_provider` 都是 `None`
- **执行流程**：由外部系统逐步驱动（通过 `execution_result` 反馈）
- **特点**：异步友好，支持与外部执行环境集成
- **返回值**：返回规划状态，供外部决策下一步

---

## 2. 核心数据结构

### Loop State（规划循环状态）
```python
{
    # 任务信息
    "task_instr": list,              # 高层任务指令
    "step_instr": list,              # 细粒度步骤指令
    
    # 计划管理
    "pending_plans": list,           # 待执行的计划队列
    "completed_plans": list,         # 已完成的计划
    "failed_plans": list,            # 失败的计划 + 失败原因
    "initial_high_level_plans": list,# 初始规划结果
    
    # 执行状态
    "retry_count": int,              # 当前重试次数
    "replanning_count": int,         # 重规划次数
    
    # 观察到的对象集合
    "seen_objs": list,               # 累积观察到的所有对象
    
    # 调试信息
    "last_prompt": str,              # 最后一次发送给LLM的提示
    "last_llm_output": str,          # 最后一次LLM输出
    "last_event": dict,              # 最后一次事件（见下表）
}
```

### Event Type（事件类型）
| 事件类型 | 触发条件 | 含义 |
|---------|--------|------|
| `initial_planning` | 首次规划完成 | 初始规划得到N个计划 |
| `execution_feedback` | 执行反馈到达 | 记录执行结果（成功/失败） |
| `unsupported_plan_skipped` | 指令不支持 | 该计划被跳过 |
| `fuzzy_match_retry_suggested` | 对象匹配成功 | 建议用模糊匹配的对象重试 |
| `dynamic_replan` | 触发重规划 | LLM生成新计划 |
| `replanning_limit_reached` | 达到限制 | 重规划次数或重试次数达到上限 |

### Response（返回值）
```python
{
    "status": str,                   # "finished" / "awaiting_execution" / "replanned" / "awaiting_retry_execution"
    "initial_high_level_plans": list,
    "completed_plans": list,
    "failed_plans": list,
    "remaining_plans": list,
    "next_plan": str or None,        # 下一个要执行的计划
    "replanning_count": int,
    "retry_count": int,
    "seen_objs": list,
    "last_prompt": str,
    "last_llm_output": str,
    "last_event": dict,
    "loop_state": dict,              # 完整的循环状态（用于外部模式的恢复）
}
```

---

## 3. 执行流程（自主模式）

```
initialize_state() 
    ↓
_build_response()
    ↓
[LOOP] while next_plan and replanning_count <= max_replanning:
    ├─ action_executor(next_plan)  # 执行计划
    │
    ├─ success?
    │  ├─ YES: 
    │  │  └─ append to completed_plans, retry_count=0
    │  │
    │  └─ NO:
    │     ├─ unsupported? 
    │     │  └─ skip, retry_count=0
    │     │
    │     ├─ object_not_found?
    │     │  ├─ fuzzy_match(object_name, available_objects)
    │     │  └─ found? insert matched_plan to queue head
    │     │
    │     └─ dynamic & retry_count < max_retries?
    │        ├─ collect new visible_objects
    │        ├─ call LLM for replanning
    │        └─ update pending_plans
    │
    └─ _build_response()
        ↓
    [END LOOP or continue]
        ↓
return response
```

---

## 4. 外部驱动模式工作流

### 典型流程（与外部执行器集成）

```
初始化 (Initialization)
└─ generate_hlp() → 获得初始计划

第1次迭代 (Iteration 1)
├─ 外部系统执行: plan[0]
├─ 反馈执行结果: { plan: "...", success: false, message: "..." }
├─ 调用 execute_with_dynamic_replanning(loop_state, execution_result)
├─ 方法处理反馈，生成新规划或建议重试
└─ 返回 response 给外部系统

第2次迭代 (Iteration 2)
├─ 外部系统执行: response['next_plan']
├─ 反馈执行结果
├─ 调用 execute_with_dynamic_replanning(response['loop_state'], execution_result)
└─ ...

直到完成 (Until Done)
└─ response['remaining_plans'] 为空 → 规划完成
```

### 外部模式的关键参数

| 参数 | 来源 | 用途 |
|-----|-----|------|
| `loop_state` | 上一次响应的 `response['loop_state']` | 恢复规划状态 |
| `execution_result` | 外部执行系统反馈 | 记录执行结果，触发重试/重规划 |
| `visible_objects` | 外部感知系统 | 最新的可见对象列表 |
| `images` | 外部视觉系统 | 视觉输入（如果 `vision=True`） |

---

## 5. 关键决策逻辑

### 5.1 对象匹配 (Object Matching)

当执行失败且错误信息包含"object not found"时：

```python
match_object_name(generated_name, available_objects, obj_sim_threshold=0.8)
```

**策略**：
1. **精确匹配** (Exact Match): 大小写不敏感的字符串相等 → 相似度 1.0
2. **语义匹配** (Semantic Matching): 使用 SentenceTransformer 编码计算余弦相似度
3. **阈值判定** (Threshold): 如果 `similarity >= obj_sim_threshold` 则认为匹配成功

**示例**：
- 用户要求："pickup apple"
- 系统回应："object apple not found"
- 可用对象：["red_apple", "banana", "orange"]
- 语义匹配："apple" → "red_apple" (similarity=0.95)
- **结果**：建议执行 "pickup red_apple"

### 5.2 重规划触发条件

只有当 `dynamic=True` 且未达到限制时才会重规划：

```python
if dynamic and retry_count < max_retries and replanning_count < max_replanning:
    # trigger replanning
```

**限制机制**：
- `max_retries=3`：单个计划最多重试3次
- `max_replanning=10`：整个任务最多重规划10次
- 两个限制是**独立且并行**的

### 5.3 可见对象累积

```python
for obj in available_objects:
    if obj and obj not in state["seen_objs"]:
        state["seen_objs"].append(obj)
state["seen_objs"].sort()
```

**作用**：
- 维护全局对象集合
- 下次LLM重规划时包含更多对象上下文
- 避免重复的"object not found"错误

---

## 6. 与原始 `execute_with_dynamic_replanning` 的差异

### 原始版本（行 345-492 的版本）
- 单一工作模式：回调式（必须提供 `action_executor` 和 `visible_objects_provider`）
- 紧耦合：循环逻辑与执行在同一方法内
- 难以集成：不适合异步或分布式系统

### 当前增强版本（行 641-763）
- **双模式**：自主模式 + 外部驱动模式
- **解耦**：通过 `loop_state` 和 `execution_result` 传递状态
- **序列化友好**：状态可以持久化/传输
- **视觉支持**：额外的 `images` 和 `vision` 参数
- **更灵活的对象来源**：支持直接传入 `visible_objects` 而非回调

---

## 7. 当前Adapter的实现

从用户提供的系统提示来看，**LLMPlannerAdapter已实现了双模式支持**（第148-234行）：

### 关键实现点

```python
# 检测是否使用动态重规划模式
use_dynamic_loop = bool(
    metadata.get('use_dynamic_replanning_loop', False)
    or metadata.get('planner_loop_state') is not None  # 恢复之前的状态
    or metadata.get('execution_result') is not None    # 有执行反馈
)

if use_dynamic_loop:
    # 外部驱动模式
    loop_result = await asyncio.to_thread(
        self._generator.execute_with_dynamic_replanning,
        ...,
        loop_state=metadata.get('planner_loop_state'),
        execution_result=metadata.get('execution_result'),
        visible_objects=...,
        images=metadata.get('images'),
    )
else:
    # 简单模式
    raw_output = await asyncio.to_thread(
        self._generator.generate_hlp, curr_task, self._k
    )
```

---

## 8. 使用场景

### 场景1：简单规划（当前Adapter默认）
```python
context = {
    'metadata': {
        'vis_objs': ['cup', 'table'],
        'completed_plans': [],
    }
}
results = await adapter.plan(instructions, context)
```
**执行**：单次LLM调用 → 获得计划 → 返回

---

### 场景2：动态执行与重规划（外部驱动模式）

```python
# 第1次调用：初始规划
context = {'metadata': {'use_dynamic_replanning_loop': True}}
result1 = await adapter.plan(['Pick up apple'], context)

# 外部执行系统执行 result1.metadata['dynamic_replanning']['next_plan']
# 假设执行失败: "object apple not found"

# 第2次调用：提供执行反馈，让规划器重新规划
feedback_context = {
    'metadata': {
        'planner_loop_state': result1.metadata['dynamic_replanning']['loop_state'],
        'execution_result': {
            'plan': 'PickupObject apple',
            'success': False,
            'message': 'object apple not found',
            'visible_objects': ['red_apple', 'banana'],  # 场景更新
        },
        'vision': False,
    }
}
result2 = await adapter.plan(['Pick up apple'], feedback_context)
# result2 会包含新的规划或重试建议
```

---

## 9. 总结

| 维度 | 原始版本 | 当前版本 | 状态 |
|-----|--------|--------|------|
| 自主执行 | ✓ | ✓ | 基础功能 |
| 外部驱动 | ✗ | ✓ | **新增** |
| 语义对象匹配 | ✓ | ✓ | 保留 |
| 动态重规划 | ✓ | ✓ | 增强 |
| 视觉支持 | ✗ | ✓ | **新增** |
| 状态持久化 | ✗ | ✓ | **新增** |
| Adapter集成 | ✗ | ✓ | **已集成** |

**Adapter现状**：已完整实现双模式支持，可直接用于：
- 简单规划任务
- 复杂的执行循环（带重规划）
- 多步骤任务执行和恢复
