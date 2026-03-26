"""
Customer Support Tickets Database with Resolution Policies

Pre-generated support tickets with ground truth classifications,
resolution policies, and severity levels.
No external APIs - all rules are hardcoded.
"""

RESOLUTION_POLICIES = {
    "billing": {
        "duplicate_charge": ["refund_duplicate_charge", "investigate_fraud"],
        "wrong_amount": ["correct_invoice", "refund_difference"],
        "subscription_issue": ["cancel_subscription", "update_subscription"],
        "fraud": ["escalate_security", "freeze_account"],
    },
    "account": {
        "password": ["reset_password_link", "send_recovery_email"],
        "email": ["update_email_settings", "verify_new_email"],
        "2fa": ["reset_2fa", "send_recovery_codes"],
        "security": ["escalate_security", "freeze_account"],
    },
    "bug": {
        "app_crash": ["update_app_version", "clear_cache_restart"],
        "ui_glitch": ["clear_cache_restart", "escalate_engineering"],
        "missing_data": ["sync_data", "escalate_engineering"],
        "critical": ["escalate_engineering", "create_hotfix"],
    },
    "feature": {
        "how_to": ["explain_feature", "send_tutorial"],
        "capability": ["escalate_sales", "enable_feature_trial"],
        "api": ["escalate_sales", "schedule_consultation"],
        "custom": ["escalate_sales", "create_feature_request"],
    },
}

TICKETS = [
    # BILLING - Easy (low/medium urgency, simple resolution)
    {
        "id": "T001",
        "message": "I was charged twice for my subscription this month. Please help!",
        "severity": "high",
        "correct_type": "billing",
        "correct_category": "duplicate_charge",
        "correct_solutions": ["refund_duplicate_charge", "investigate_fraud"],
        "correct_primary_solution": "refund_duplicate_charge",
        "needs_escalation": False,
        "task_level": 1,
    },
    {
        "id": "T002",
        "message": "Why is my credit card being charged monthly? I thought it was a one-time purchase.",
        "severity": "medium",
        "correct_type": "billing",
        "correct_category": "subscription_issue",
        "correct_solutions": ["cancel_subscription", "update_subscription"],
        "correct_primary_solution": "cancel_subscription",
        "needs_escalation": False,
        "task_level": 1,
    },
    
    # ACCOUNT - Easy (simple reset/verification)
    {
        "id": "T003",
        "message": "I can't log into my account. It says invalid password but I'm sure it's correct.",
        "severity": "high",
        "correct_type": "account",
        "correct_category": "password",
        "correct_solutions": ["reset_password_link", "send_recovery_email"],
        "correct_primary_solution": "reset_password_link",
        "needs_escalation": False,
        "task_level": 1,
    },
    {
        "id": "T004",
        "message": "Where do I find the export feature?",
        "severity": "low",
        "correct_type": "feature",
        "correct_category": "how_to",
        "correct_solutions": ["explain_feature", "send_tutorial"],
        "correct_primary_solution": "explain_feature",
        "needs_escalation": False,
        "task_level": 1,
    },
    
    # BUG - Easy (standard fixes)
    {
        "id": "T005",
        "message": "The app crashes when I try to upload a file",
        "severity": "high",
        "correct_type": "bug",
        "correct_category": "app_crash",
        "correct_solutions": ["update_app_version", "clear_cache_restart"],
        "correct_primary_solution": "update_app_version",
        "needs_escalation": False,
        "task_level": 1,
    },
    
    # BILLING - Medium (requires investigation, might escalate)
    {
        "id": "T006",
        "message": "My invoice shows $150 but I thought my plan was $99/month",
        "severity": "medium",
        "correct_type": "billing",
        "correct_category": "wrong_amount",
        "correct_solutions": ["correct_invoice", "refund_difference"],
        "correct_primary_solution": "correct_invoice",
        "needs_escalation": False,
        "task_level": 2,
    },
    {
        "id": "T007",
        "message": "I see a charge from 3 months ago that I don't recognize. Might be fraud.",
        "severity": "high",
        "correct_type": "billing",
        "correct_category": "fraud",
        "correct_solutions": ["escalate_security", "freeze_account"],
        "correct_primary_solution": "escalate_security",
        "needs_escalation": True,
        "task_level": 2,
    },
    
    # ACCOUNT - Medium (2FA / complex account issues)
    {
        "id": "T008",
        "message": "I changed my email but now I can't receive verification codes",
        "severity": "high",
        "correct_type": "account",
        "correct_category": "email",
        "correct_solutions": ["update_email_settings", "verify_new_email"],
        "correct_primary_solution": "update_email_settings",
        "needs_escalation": False,
        "task_level": 2,
    },
    {
        "id": "T009",
        "message": "My account was hacked. Someone changed my password and email.",
        "severity": "high",
        "correct_type": "account",
        "correct_category": "security",
        "correct_solutions": ["escalate_security", "freeze_account"],
        "correct_primary_solution": "escalate_security",
        "needs_escalation": True,
        "task_level": 2,
    },
    
    # BUG - Medium
    {
        "id": "T010",
        "message": "Dashboard graphs not loading properly. Shows blank instead of data.",
        "severity": "medium",
        "correct_type": "bug",
        "correct_category": "ui_glitch",
        "correct_solutions": ["clear_cache_restart", "escalate_engineering"],
        "correct_primary_solution": "clear_cache_restart",
        "needs_escalation": False,
        "task_level": 2,
    },
    
    # FEATURE - Medium (business questions)
    {
        "id": "T011",
        "message": "We need custom API integration for our system. Is this possible?",
        "severity": "medium",
        "correct_type": "feature",
        "correct_category": "api",
        "correct_solutions": ["escalate_sales", "schedule_consultation"],
        "correct_primary_solution": "escalate_sales",
        "needs_escalation": True,
        "task_level": 2,
    },
    
    # HARD - Complex escalation scenarios
    {
        "id": "T012",
        "message": "Export to PDF feature broken after update. Critical for our reports.",
        "severity": "high",
        "correct_type": "bug",
        "correct_category": "critical",
        "correct_solutions": ["escalate_engineering", "create_hotfix"],
        "correct_primary_solution": "escalate_engineering",
        "needs_escalation": True,
        "task_level": 3,
    },
    {
        "id": "T013",
        "message": "Multiple users on my account reporting suspicious activity",
        "severity": "high",
        "correct_type": "account",
        "correct_category": "security",
        "correct_solutions": ["escalate_security", "freeze_account"],
        "correct_primary_solution": "escalate_security",
        "needs_escalation": True,
        "task_level": 3,
    },
    {
        "id": "T014",
        "message": "Data missing after system outage. Need immediate recovery.",
        "severity": "high",
        "correct_type": "bug",
        "correct_category": "missing_data",
        "correct_solutions": ["sync_data", "escalate_engineering"],
        "correct_primary_solution": "escalate_engineering",
        "needs_escalation": True,
        "task_level": 3,
    },
]


def get_random_ticket(task_level: int = 1):
    """Get a random ticket for the given task level."""
    import random
    filtered = [t for t in TICKETS if t["task_level"] <= task_level]
    return random.choice(filtered) if filtered else TICKETS[0]


def get_ticket_by_id(ticket_id: str):
    """Get specific ticket by ID."""
    for ticket in TICKETS:
        if ticket["id"] == ticket_id:
            return ticket
    return None


def get_valid_solutions_for_issue(issue_type: str, category: str) -> list:
    """
    Get valid solution IDs for a specific issue type and category.
    
    Returns:
        List of valid solution IDs
    """
    if issue_type in RESOLUTION_POLICIES:
        if category in RESOLUTION_POLICIES[issue_type]:
            return RESOLUTION_POLICIES[issue_type][category]
    return []

