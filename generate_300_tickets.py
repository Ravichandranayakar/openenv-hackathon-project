import json
import random
import os

# Define the acceptable primary solutions for each category and severity
TEMPLATES = {
    "billing": [
        # Low severity
        ({"severity": "low", "task_level": 1, "correct_primary_solution": "explain_feature", "correct_solutions": ["explain_feature"]},
         ["How do I view my past invoices?", "Where is the billing history located?", "I need help finding my receipt from last month."]),
        ({"severity": "low", "task_level": 1, "correct_primary_solution": "update_subscription", "correct_solutions": ["update_subscription"]},
         ["I want to switch from monthly to annual billing.", "How do I upgrade my plan?", "I need to add a new credit card."]),
        # Medium severity
        ({"severity": "medium", "task_level": 2, "correct_primary_solution": "cancel_subscription", "correct_solutions": ["cancel_subscription"]},
         ["Please cancel my subscription immediately.", "I no longer need this service, how do I stop the billing?", "Cancel my account plan."]),
        ({"severity": "medium", "task_level": 2, "correct_primary_solution": "correct_invoice", "correct_solutions": ["correct_invoice", "refund_difference"]},
         ["My invoice has the wrong company name on it.", "The tax amount on my receipt is incorrect.", "You billed me $50 but my plan is $40."]),
        # High severity
        ({"severity": "high", "task_level": 2, "correct_primary_solution": "refund_duplicate_charge", "correct_solutions": ["refund_duplicate_charge"]},
         ["I was charged twice yesterday!", "My credit card shows two identical transactions.", "Duplicate charge on my account."]),
        # Critical severity
        ({"severity": "critical", "task_level": 3, "correct_primary_solution": "escalate_security", "correct_solutions": ["escalate_security", "freeze_account"], "needs_escalation": True},
         ["Someone is using my credit card on your platform without my permission!", "URGENT: Fraudulent charges on my account!", "I did not authorize these transactions! Stop them!"])
    ],
    "account": [
        # Low severity
        ({"severity": "low", "task_level": 1, "correct_primary_solution": "reset_password_link", "correct_solutions": ["reset_password_link"]},
         ["I forgot my password.", "Can you send a password reset email?", "How do I change my password?"]),
        ({"severity": "low", "task_level": 1, "correct_primary_solution": "update_email_settings", "correct_solutions": ["update_email_settings"]},
         ["I want to change my primary email address.", "How do I update my profile picture?", "Stop sending me marketing emails."]),
        # Medium severity
        ({"severity": "medium", "task_level": 2, "correct_primary_solution": "explain_feature", "correct_solutions": ["explain_feature", "update_email_settings"]},
         ["How do I set up Two-Factor Authentication?", "I need to add a team member to my account.", "Where do I configure role permissions?"]),
        # High severity
        ({"severity": "high", "task_level": 2, "correct_primary_solution": "update_email_settings", "correct_solutions": ["update_email_settings", "escalate_security"]},
         ["I lost my 2FA backup codes and can't log in.", "My phone was stolen, I need to bypass 2FA.", "Can't access my old email to verify my account."]),
        # Critical severity
        ({"severity": "critical", "task_level": 3, "correct_primary_solution": "escalate_security", "correct_solutions": ["escalate_security", "freeze_account"], "needs_escalation": True},
         ["My account has been HACKED!", "Someone changed my login email and locked me out!", "Unauthorized login detected from another country!"])
    ],
    "technical": [
        # Low severity
        ({"severity": "low", "task_level": 1, "correct_primary_solution": "explain_feature", "correct_solutions": ["explain_feature"]},
         ["Where is the export to CSV button?", "Does your API support pagination?", "How do I filter the dashboard by date?"]),
        # Medium severity
        ({"severity": "medium", "task_level": 2, "correct_primary_solution": "clear_cache_restart", "correct_solutions": ["clear_cache_restart"]},
         ["The dashboard is loading very slowly today.", "Images are not rendering on my profile.", "I'm getting a 404 error on the reports page."]),
        ({"severity": "medium", "task_level": 2, "correct_primary_solution": "sync_data", "correct_solutions": ["sync_data"]},
         ["My recent transactions aren't showing up.", "The mobile app isn't syncing with the web app.", "Data export is missing yesterday's entries."]),
        # High severity
        ({"severity": "high", "task_level": 2, "correct_primary_solution": "update_app_version", "correct_solutions": ["update_app_version", "escalate_engineering"]},
         ["The app crashes every time I try to upload a PDF.", "Fatal error when opening the settings menu.", "iOS app keeps force closing."]),
        # Critical severity
        ({"severity": "critical", "task_level": 3, "correct_primary_solution": "escalate_engineering", "correct_solutions": ["escalate_engineering"], "needs_escalation": True},
         ["PRODUCTION IS DOWN. We are getting 500 errors everywhere!", "API is completely unresponsive.", "Major data loss in our enterprise environment!"])
    ]
}

def generate_tickets():
    import importlib.util
    import sys
    
    # Load existing tickets
    file_path = r"c:\Users\Ravichandran\openenv-hackathon-project\my_env\server\data\tickets.py"
    spec = importlib.util.spec_from_file_location("tickets", file_path)
    tickets_module = importlib.util.module_from_spec(spec)
    sys.modules["tickets"] = tickets_module
    spec.loader.exec_module(tickets_module)
    
    existing_tickets = tickets_module.TICKETS
    existing_ids = set(t["id"] for t in existing_tickets)
    
    num_to_generate = 300 - len(existing_tickets)
    if num_to_generate <= 0:
        print("Already have 300 or more tickets.")
        return
        
    print(f"Generating {num_to_generate} new tickets...")
    
    new_tickets = []
    
    categories = ["billing", "account", "technical"]
    
    prefixes = ["Hello,", "Hi support,", "URGENT:", "Please help:", "Question:", "Issue:", "Bug report:", ""]
    suffixes = [" Thanks.", " Please fix this ASAP.", " Let me know.", " Any ideas?", " Appreciate the help.", ""]
    
    for i in range(num_to_generate):
        ticket_num = len(existing_tickets) + i + 1
        ticket_id = f"T{ticket_num:03d}"
        
        # Pick category
        cat = random.choice(categories)
        
        # Pick template
        template_metadata, messages = random.choice(TEMPLATES[cat])
        
        # Generate message
        base_msg = random.choice(messages)
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        
        message = f"{prefix} {base_msg} {suffix}".strip()
        
        # Optional: slight mutations
        mutations = [
            ("!", "!!"), ("?", "??"), ("my", "our"), ("I", "We"), 
            ("help", "assist"), ("ASAP", "immediately")
        ]
        if random.random() > 0.5:
            m_old, m_new = random.choice(mutations)
            message = message.replace(m_old, m_new)
            
        ticket = {
            "id": ticket_id,
            "message": message,
            "severity": template_metadata["severity"],
            "correct_type": cat,
            "correct_category": "general" if cat != "technical" else "bug",
            "correct_solutions": template_metadata["correct_solutions"],
            "correct_primary_solution": template_metadata["correct_primary_solution"],
            "needs_escalation": template_metadata.get("needs_escalation", False),
            "task_level": template_metadata["task_level"]
        }
        new_tickets.append(ticket)
        
    all_tickets = existing_tickets + new_tickets
    
    # Write back to file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # We will reconstruct the file
    # Find the end of imports / start of TICKETS
    import_part = ""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    header_lines = []
    for line in lines:
        if line.startswith("TICKETS = ["):
            break
        header_lines.append(line)
        
    footer_lines = []
    in_footer = False
    for line in lines:
        if line.startswith("def get_random_ticket"):
            in_footer = True
        if in_footer:
            footer_lines.append(line)
            
    # Serialize TICKETS list
    import pprint
    
    with open(file_path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line)
            
        f.write("TICKETS = [\n")
        for t in all_tickets:
            # Write dict formatting
            f.write("  {\n")
            f.write(f'    "id": "{t["id"]}",\n')
            f.write(f'    "message": "{t["message"]}",\n')
            f.write(f'    "severity": "{t["severity"]}",\n')
            f.write(f'    "correct_type": "{t["correct_type"]}",\n')
            f.write(f'    "correct_category": "{t["correct_category"]}",\n')
            
            # format lists properly
            sols_str = ", ".join([f'"{s}"' for s in t["correct_solutions"]])
            f.write(f'    "correct_solutions": [{sols_str}],\n')
            
            f.write(f'    "correct_primary_solution": "{t["correct_primary_solution"]}",\n')
            f.write(f'    "needs_escalation": {str(t["needs_escalation"])},\n')
            f.write(f'    "task_level": {t["task_level"]},\n')
            f.write("  },\n")
        f.write("]\n\n\n")
        
        for line in footer_lines:
            f.write(line)
            
    print(f"Successfully generated and wrote {len(all_tickets)} tickets to tickets.py")

if __name__ == "__main__":
    generate_tickets()
