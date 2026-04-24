# data/generate_train.py
#
# Generates 1,000 synthetic training examples (200 per class).
# Method: hand-crafted templates with parameterised slot filling.
# NO LLM used for training data — fully reproducible, zero API cost.
# Noise variants (lowercase, informal phrasing) are added to close
# the gap between clean templates and real-world ticket style.

import json
import random
from collections import Counter
from pathlib import Path

random.seed(42)

# ── Template bank ──────────────────────────────────────────────────────────────

BILLING = [
    "I was charged twice for my subscription this month.",
    "My invoice shows an incorrect amount for {month}.",
    "I need a refund for the duplicate charge on my account.",
    "Why was I billed for a plan I already downgraded from?",
    "My payment failed but you still charged my card.",
    "I cancelled my subscription but was charged again in {month}.",
    "The price on my bill doesn't match what was advertised.",
    "I have an unexpected charge of ₹{amount} on my statement.",
    "My free trial ended early and I was billed before the period finished.",
    "Please send me a receipt for my last payment.",
    "I updated my credit card but the old one was charged.",
    "I'm being billed for users I removed from my account.",
    "My annual plan was auto-renewed without any notice.",
    "There's a tax discrepancy on my invoice from {month}.",
    "I need to dispute a charge of ₹{amount} from last {month}.",
    "The promo discount was not applied to my bill.",
    "I'm on the free plan but I see a ₹{amount} charge on my bank statement.",
    "My billing cycle date changed without my consent.",
    "I was overcharged compared to the pricing shown on your website.",
    "Can you explain the line item called platform fee on my invoice?",
]

TECHNICAL_ISSUE = [
    "The export to CSV button does nothing when I click it.",
    "I get a 500 error every time I try to save my project.",
    "The dashboard won't load — it just spins forever.",
    "My integrations stopped syncing after yesterday's update.",
    "I can't log in even with the correct password.",
    "The mobile app crashes every time I open the settings page.",
    "Notifications are not being delivered to my email address.",
    "The search feature returns no results for items I know exist.",
    "The API is returning a 401 even with a valid token.",
    "My uploaded files disappeared after the recent system migration.",
    "Two-factor authentication codes are not being accepted.",
    "The bulk import feature fails every time at step 3.",
    "I'm getting CORS errors when calling your API from my app.",
    "The PDF report download always comes out corrupted.",
    "Video playback inside your platform freezes after 30 seconds.",
    "My webhook endpoint completely stopped receiving events.",
    "The dark mode preference is not being saved between sessions.",
    "I keep getting logged out after just a few minutes.",
    "Images I upload are not displaying in the preview panel.",
    "The audit log is missing entries for the past five days.",
]

FEATURE_REQUEST = [
    "It would be great if I could export data to Excel format.",
    "Please add a dark mode option to the web application.",
    "Can you add support for multiple languages in the interface?",
    "I'd like to see a native mobile app for iOS.",
    "Would love the ability to bulk delete records at once.",
    "Please add a Slack integration for real-time notifications.",
    "Can you add an option to schedule reports to be sent by email?",
    "Keyboard shortcuts for common actions would save me a lot of time.",
    "A two-pane view for comparing documents side by side would help.",
    "I'd like an API endpoint for bulk data import.",
    "Please add a calendar view for managing tasks and deadlines.",
    "Role-based access controls would be very useful for our team.",
    "Can you add colour-coding for different project categories?",
    "I'd like the ability to set up recurring tasks automatically.",
    "SSO support for enterprise accounts would make onboarding easier.",
    "An undo button after deleting items is badly needed.",
    "Can you provide a Zapier integration for automating workflows?",
    "I'd like to customise the columns shown in the list view.",
    "A public API for read-only access to my data would be very helpful.",
    "Please add an AI-powered summary for long comment threads.",
]

COMPLAINT = [
    "Your customer service is absolutely terrible. I've been waiting 5 days for a reply.",
    "This product is not as advertised. I feel completely misled.",
    "I've submitted three support tickets and nobody has responded.",
    "The constant bugs are making this tool unusable for my business.",
    "Your last update broke my entire workflow and no one seems to care.",
    "I've been a paying customer for two years and I'm treated like this?",
    "Your documentation is confusing, outdated, and full of errors.",
    "This is the third time this week I've had the exact same issue.",
    "Your uptime SLA is not being met and I'm losing real revenue.",
    "I asked for a refund two weeks ago and I still haven't received anything.",
    "The quality has dropped significantly since your last major update.",
    "I'm extremely frustrated with the complete lack of communication.",
    "This is unacceptable behaviour from a company I trusted with my data.",
    "I've escalated this issue four times now with absolutely no resolution.",
    "Your pricing increase was not communicated to customers at all.",
    "I've been promised a fix for months and nothing has changed.",
    "I feel like smaller customers are completely ignored by your team.",
    "The outage last week cost my team several hours of productive work.",
    "Your chatbot gave me completely wrong information three times in a row.",
    "I am seriously considering cancelling due to the consistently poor service.",
]

OTHER = [
    "How do I change my account username in the settings?",
    "Where can I find the full documentation for your API?",
    "What are your business hours for customer support?",
    "Can I transfer my account to a different email address?",
    "What is your data retention policy for inactive accounts?",
    "How do I add a team member to my existing account?",
    "Where can I download the desktop application for Windows?",
    "Do you offer a discount for registered non-profit organisations?",
    "What payment methods do you currently accept?",
    "How do I permanently delete my account and all associated data?",
    "Can I use your service if my company is based in a restricted region?",
    "What is the maximum file upload size on the Pro plan?",
    "How does your pricing model work for large enterprise teams?",
    "Do you have a student or educational institution discount?",
    "Where can I find your full privacy policy document?",
    "I'd like to know more about your enterprise plan pricing.",
    "Can I request a product demo before committing to a paid plan?",
    "What is the difference between the Basic and Pro plan features?",
    "How do I update my notification preferences in the account settings?",
    "Can you explain how the monthly credit system works?",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

MONTHS  = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]
AMOUNTS = ["199", "299", "499", "799", "999", "1299", "2499", "4999"]

PREFIXES = ["Hi, ", "Hello, ", "Hey there, ", "Good day, ", "Hi support team, ", ""]
SUFFIXES = [" Please help.", " This is urgent.", " Thank you.", " Need this resolved.", " Appreciate your help.", ""]

# Noise functions simulate real ticket messiness
NOISE_FNS = [
    lambda t: t.lower(),
    lambda t: t.replace("I ", "i "),
    lambda t: t.rstrip(".") + "!!",
    lambda t: t,  # identity — keep original
    lambda t: t,  # identity — weight towards clean
]


def fill(text: str) -> str:
    return (text
            .replace("{month}", random.choice(MONTHS))
            .replace("{amount}", random.choice(AMOUNTS)))


def apply_noise(text: str) -> str:
    return random.choice(NOISE_FNS)(text)


def expand(templates: list, label: str, n: int = 200) -> list:
    data = []
    per   = n // len(templates)
    extra = n % len(templates)

    for i, tmpl in enumerate(templates):
        count = per + (1 if i < extra else 0)
        for j in range(count):
            text = fill(tmpl)
            if j == 0:
                # First copy: clean, no prefix/suffix
                final = text
            else:
                # Subsequent copies: add surface variation + optional noise
                prefix = random.choice(PREFIXES)
                suffix = random.choice(SUFFIXES)
                final  = apply_noise(prefix + text + suffix).strip()
            data.append({"text": final, "label": label})

    return data


# ── Build & save ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_data = []
    all_data += expand(BILLING,          "billing",          200)
    all_data += expand(TECHNICAL_ISSUE,  "technical_issue",  200)
    all_data += expand(FEATURE_REQUEST,  "feature_request",  200)
    all_data += expand(COMPLAINT,        "complaint",        200)
    all_data += expand(OTHER,            "other",            200)

    random.shuffle(all_data)

    out = Path(__file__).parent / "train_data.json"
    with open(out, "w") as f:
        json.dump(all_data, f, indent=2)

    counts = Counter(d["label"] for d in all_data)
    print(f"Saved {len(all_data)} training examples to {out}")
    for label, count in sorted(counts.items()):
        print(f"  {label:20s}: {count}")