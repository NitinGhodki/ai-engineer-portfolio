"""
Input safety — prompt injection defense from Day 5.
"""

INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "forget your instructions",
    "disregard your system prompt",
    "you are now",
    "pretend you are",
    "your new instructions",
    "override instructions",
    "jailbreak",
    "do anything now",
    "dan mode",
    "act as if you have no restrictions",
]


def check_input(user_input: str) -> dict:
    """
    Check user input for injection attempts.
    Returns: {is_safe, reason, flagged_pattern}
    """
    lower = user_input.lower()

    for pattern in INJECTION_PATTERNS:
        if pattern in lower:
            return {
                "is_safe": False,
                "reason": "Potential prompt injection detected",
                "flagged_pattern": pattern,
            }

    if len(user_input) > 1000:
        return {
            "is_safe": False,
            "reason": "Input exceeds maximum length of 1000 characters",
            "flagged_pattern": None,
        }

    return {
        "is_safe": True,
        "reason": "Input passed safety checks",
        "flagged_pattern": None,
    }