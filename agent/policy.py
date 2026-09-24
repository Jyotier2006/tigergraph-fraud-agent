"""
The Fraud Policy, encoded.

Actions, approval routes and rules R1-R10 come straight from the policy
document shipped with the dataset.  Keeping them here -- rather than in a
prompt -- means the agent cannot invent an action, cannot mis-route an
approval, and every recommendation carries the rule that produced it.
"""
from __future__ import annotations

ACTIONS = {
    "ALLOW_TRANSACTION", "DECLINE_TRANSACTION", "MONITOR_CARD",
    "MONITOR_CONNECTED_CARDS", "WARN_CUSTOMER", "VERIFY_WITH_CUSTOMER",
    "STEP_UP_AUTH", "BLOCK_CARD", "BLOCK_ALL_CARDS", "GENERATE_REPORT",
    "CREATE_CASE", "FILE_REPORT", "ESCALATE_TO_ANALYST", "CLOSE_NO_FRAUD",
}

AUTO = {
    "ALLOW_TRANSACTION", "MONITOR_CARD", "MONITOR_CONNECTED_CARDS",
    "WARN_CUSTOMER", "VERIFY_WITH_CUSTOMER", "STEP_UP_AUTH",
    "GENERATE_REPORT", "CREATE_CASE", "ESCALATE_TO_ANALYST", "CLOSE_NO_FRAUD",
}


def route_for(action: str, exposure: float = 0.0) -> str:
    """Approval routing table, policy section 2."""
    if action in AUTO:
        return "auto"
    if action == "DECLINE_TRANSACTION":
        return "L1"
    if action == "BLOCK_CARD":
        return "L1" if exposure <= 2500 else "L2"
    if action in ("BLOCK_ALL_CARDS", "FILE_REPORT"):
        return "L2"
    raise ValueError(f"unknown action {action}")


def act(action: str, reason: str, exposure: float = 0.0) -> dict:
    assert action in ACTIONS, f"action not in policy: {action}"
    return {"action": action, "route": route_for(action, exposure), "reason": reason}


def sar_required(probability: float, verdict: str, exposure: float,
                 shared_origin: bool, connected_fraud: bool,
                 pattern: str) -> tuple[bool, str]:
    """
    Policy 3a.  A report is filed when fraud is confirmed or strongly suspected
    AND at least one of: exposure > $1,000; the activity connects to a shared
    device profile / region cluster / another customer's fraud; the pattern is
    coordinated or undocumented (R9).
    """
    strong = verdict == "fraud" or probability >= 0.70
    if not strong:
        return False, ("3a: fraud is neither confirmed nor strongly suspected "
                       f"(probability {probability:.2f}), so no regulatory filing is due. "
                       "The internal case carries the record.")
    triggers = []
    if exposure > 1000:
        triggers.append(f"exposure ${exposure:,.2f} exceeds the $1,000 threshold")
    if shared_origin:
        triggers.append("the activity connects to a shared device profile / region cluster")
    if connected_fraud:
        triggers.append("the activity connects to another card's fraud")
    if pattern == "undocumented" and (shared_origin or connected_fraud):
        triggers.append("R9: the pattern is undocumented and shows coordinated abuse "
                        "across customers")
    if not triggers:
        return False, ("3a: fraud is strongly suspected but none of the filing "
                       f"triggers are met (exposure ${exposure:,.2f} is under $1,000, "
                       "no shared origin, no link to another customer's fraud). "
                       "Case only, no report.")
    return True, "3a: " + "; ".join(triggers)


def stop_reason(probability: float, n_independent: int, settled_by_response: bool,
                exposure: float) -> str:
    """Policy section 6."""
    if settled_by_response:
        return ("Policy 6: the verification response settled the question; the "
                "remaining steps could not change the recommended actions.")
    if probability >= 0.85 and n_independent >= 2:
        return (f"Policy 6: fraud probability {probability:.2f} is at or above 0.85 and "
                f"rests on {n_independent} independent pieces of evidence. Further "
                f"investigation would not change a defensible action.")
    if probability <= 0.15 and n_independent >= 2:
        return (f"Policy 6: fraud probability {probability:.2f} is at or below 0.15 with "
                f"{n_independent} independent corroborating findings. Closing as "
                f"legitimate is defensible now.")
    return (f"Policy 6: probability {probability:.2f} is inside the uncertain band and "
            f"further graph evidence would not move it; the decision is handed on with "
            f"the uncertainty stated (exposure ${exposure:,.2f}).")
