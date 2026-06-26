import os
import sys
import json
import argparse

from .api_client import APIClient
from .agent import Agent


class PlanRunner:
    def __init__(self, agent):
        self.agent = agent
        self.step_count = 0
        self.action_history = []

    def run(self, task_instruction, obj_list,
            image_provider=None, feedback_provider=None,
            max_rounds=100, verbose=True):
        """
        Run the planning loop without a simulation environment.

        Args:
            task_instruction: Task description string.
            obj_list: List of observable objects in the scene.
            image_provider: Callable(step_count) -> str|None, returns image path.
                            If None, visual abilities will use no images.
            feedback_provider: Callable(action_str) -> bool, returns success/failure.
                               If None, assumes all actions succeed.
            max_rounds: Maximum number of get_qwen_action_raw calls.
            verbose: Print planning trace to stdout.

        Returns:
            dict with keys: task, actions, trace, finished
        """
        self.agent.reset(self.agent.save_path, obj_list)
        self.agent.process_task(None, task_instruction)
        self.step_count = 0
        self.action_history = []

        finished = False
        action_rounds = 0

        for round_idx in range(max_rounds):
            raw_actions = self.agent.get_qwen_action_raw()

            if raw_actions == ["fail"]:
                if verbose:
                    print(f"\n[ROUND {round_idx}] Agent failed.")
                finished = False
                break

            if raw_actions == ["pass"]:
                if verbose:
                    last_trace = self.agent.trace[-1] if self.agent.trace else {}
                    ability = last_trace.get("ability", "?")
                    resp = last_trace.get("response", "")
                    resp_short = resp[:120].replace("\n", " ") if resp else ""
                    print(f"  [ability: {ability}] {resp_short}...")
                continue

            action_rounds += 1
            if verbose:
                print(f"\n{'='*60}")
                print(f"[ACTION ROUND {action_rounds}] Planned actions: {raw_actions}")
                print(f"{'='*60}")

            self.action_history.extend(raw_actions)

            for action in raw_actions:
                self.step_count += 1

                if feedback_provider is not None:
                    success = feedback_provider(action)
                else:
                    success = True
                    if verbose:
                        print(f"  [step {self.step_count}] {action} -> assumed success")

                self.agent.process_feedback(success, action)

            if image_provider is not None:
                img_path = image_provider(self.step_count)
                if img_path is not None:
                    self.agent.process_observation(img_path)
                    if verbose:
                        print(f"  [observation] {img_path}")

            if "Stop" in self.agent.core_history or action_rounds >= 30:
                finished = True
                if verbose:
                    print(f"\n[DONE] Task completed or max action rounds reached.")
                break

        result = {
            "task": task_instruction,
            "actions": self.action_history,
            "trace": self.agent.trace,
            "finished": finished,
        }

        self.agent.save_trace()

        return result


class InteractiveRunner:
    def __init__(self, agent):
        self.agent = agent
        self.step_count = 0

    def run(self, task_instruction, obj_list, max_rounds=100):
        """
        Interactive planning loop: user provides feedback and images manually.
        """
        self.agent.reset(self.agent.save_path, obj_list)
        self.agent.process_task(None, task_instruction)
        self.step_count = 0

        for round_idx in range(max_rounds):
            raw_actions = self.agent.get_qwen_action_raw()

            if raw_actions == ["fail"]:
                print("\n[Agent failed.]")
                break

            if raw_actions == ["pass"]:
                last_trace = self.agent.trace[-1] if self.agent.trace else {}
                ability = last_trace.get("ability", "?")
                resp = last_trace.get("response", "")
                print(f"  [ability: {ability}] {resp[:200]}...")
                continue

            print(f"\n{'='*60}")
            print(f"Planned actions: {raw_actions}")
            print(f"{'='*60}")

            for action in raw_actions:
                self.step_count += 1
                while True:
                    ans = input(f"  Action '{action}' succeeded? (y/n/skip): ").strip().lower()
                    if ans in ["y", "n", "skip"]:
                        break
                    print("  Please enter y, n, or skip.")

                if ans == "skip":
                    continue
                success = (ans == "y")
                self.agent.process_feedback(success, action)

            img_path = input("  Enter image path for next observation (Enter to skip): ").strip()
            if img_path and os.path.exists(img_path):
                self.agent.process_observation(img_path)
            elif img_path:
                print(f"  [Warning] Image not found: {img_path}")

            cont = input("  Continue? (y/n): ").strip().lower()
            if cont == "n":
                break

        self.agent.save_trace()
        print(f"\nTrace saved to {os.path.join(self.agent.save_path, 'trace.json')}")


def main():
    parser = argparse.ArgumentParser(description="RoboAgent Planning with API-based LLM")
    parser.add_argument("--model", type=str, default="gpt-5.5", help="Model name for API")
    parser.add_argument("--base_url", type=str, default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--api_key", type=str, default=None, help="API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--env", type=str, default="alfworld", choices=["alfworld", "eb-alfred", "generic"])
    parser.add_argument("--save_path", type=str, default="api_agent_output")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "interactive"],
                        help="auto=assume success, interactive=user provides feedback")
    parser.add_argument("--task", type=str, default=None,
                        help="Task instruction (or use --task_file)")
    parser.add_argument("--task_file", type=str, default=None,
                        help="JSON file with task info: {task, objects, image_dir}")
    parser.add_argument("--objects", type=str, nargs="*", default=None,
                        help="List of observable objects (e.g., Apple 1 Fridge 1)")
    parser.add_argument("--max_rounds", type=int, default=100)
    args = parser.parse_args()

    api_client = APIClient(
        model_name=args.model,
        **({"base_url": args.base_url} if args.base_url else {}),
        **({"api_key": args.api_key} if args.api_key else {}),
    )
    agent = Agent(api_client, env_name=args.env)
    agent.save_path = args.save_path

    DEFAULT_OBJECTS = [
        "Apple 1", "CounterTop 1", "Fridge 1", "Microwave 1",
        "Cabinet 1", "Cabinet 2", "Sink 1", "Stove 1",
    ]

    if args.task_file:
        with open(args.task_file, "r", encoding="utf-8") as f:
            task_data = json.load(f)
        task_instruction = task_data["task"]
        obj_list = task_data.get("objects", DEFAULT_OBJECTS)
        image_dir = task_data.get("image_dir", None)
    elif args.task:
        task_instruction = args.task
        obj_list = args.objects if args.objects else DEFAULT_OBJECTS
        image_dir = None
    else:
        task_instruction = "heat Apple 1 with Microwave 1"
        obj_list = DEFAULT_OBJECTS
        image_dir = None
        print(f"[Demo mode] Using example task: {task_instruction}")

    def image_provider(step):
        if image_dir and os.path.isdir(image_dir):
            path = os.path.join(image_dir, f"step_{step}.png")
            if os.path.exists(path):
                return path
        return None

    os.makedirs(args.save_path, exist_ok=True)

    if args.mode == "interactive":
        runner = InteractiveRunner(agent)
        runner.run(task_instruction, obj_list, max_rounds=args.max_rounds)
    else:
        runner = PlanRunner(agent)
        result = runner.run(
            task_instruction, obj_list,
            image_provider=image_provider if image_dir else None,
            feedback_provider=None,
            max_rounds=args.max_rounds,
            verbose=True,
        )
        result_path = os.path.join(args.save_path, "result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResult saved to {result_path}")
        print(f"Total actions: {len(result['actions'])}")
        print(f"Actions: {result['actions']}")


if __name__ == "__main__":
    main()
