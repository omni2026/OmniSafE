"""Stub HouseholdSim used only by the upstream ISR-LLM CLI (main.py).

The OMNISAFE adapter (runtime/planning/adapters/isr_llm_planner.py) does NOT
go through this simulator — it drives Translator → Planner → (Self-)Validator
directly and emits the resulting PDDL action atoms. This stub exists so
``python main.py --domain household`` does not crash, and so the
``LLM_trans_exact_feedback`` path degrades gracefully when no real simulator
is available (it reports every action sequence as satisfied).
"""

import re


class HouseholdSim(object):
    """Minimal household simulator stub.

    The original Block / BallMoving / Cooking simulators in this repo are
    state-machine validators tightly coupled to their domain's PDDL encoding.
    For household we deliberately keep things lightweight: the simulator just
    parses out the action atoms and reports success.

    The adapter never calls into this class; it lives here purely to keep the
    upstream CLI runnable with ``--domain household``.
    """

    def __init__(self):
        self.initial_state = None
        self.goal_state = None
        self.constraint = None

    # ------------------------------------------------------------------
    # Interface methods expected by main.py
    # ------------------------------------------------------------------

    def generate_scene_description(self, initial_state, goal_state, constraint=None):
        # The upstream CLI test loop iterates over preloaded numpy scenarios,
        # which household does not ship. Return a placeholder description so
        # the CLI keeps working if someone wires it up later.
        return (
            "Household scene placeholder. Provide the natural-language task "
            "directly via the OMNISAFE adapter instead of relying on this stub."
        )

    def initialize_state(self, initial_state, goal_state, constraint=None):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.constraint = constraint

    def simulate_actions(self, action_sequence, test_log_file_path):
        """Return (is_satisfied, is_error, error_message, error_action).

        Conservative implementation: parse the atoms, log them, and report
        success. Real precondition checking lives in the LLM Validator path
        for household.
        """
        actions = re.findall(r'\(.*?\)', action_sequence or '')

        try:
            with open(test_log_file_path, "a") as f:
                f.write("HouseholdSim stub: parsed " + str(len(actions)) + " action atoms.\n")
                for i, atom in enumerate(actions):
                    f.write("  Action " + str(i) + ": " + atom + "\n")
                f.write("HouseholdSim stub: reporting goal as satisfied (no precondition checks).\n")
        except Exception:
            # Log file is optional from the stub's point of view.
            pass

        is_satisfied = True
        is_error = False
        error_message = ""
        error_action = None
        return is_satisfied, is_error, error_message, error_action
