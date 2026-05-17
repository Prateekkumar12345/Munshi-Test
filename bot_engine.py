import json
import os
import re
from datetime import datetime, timedelta
from typing import Optional

from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_MODEL = "gpt-4.1-mini"


# DATE EXTRACTION
class DateTimeExtractor:
    RELATIVE_KEYWORDS = {
        "today": 0,
        "aaj": 0,
        "tomorrow": 1,
        "kal": 1,
        "parso": 2,
    }

    CLOCK_RE = re.compile(
        r"(\d{1,2})(?:(\d{2}))?\s*(am|pm|baj|bje|baj|\*)?",
        re.IGNORECASE,
    )

    @staticmethod
    def parse_clock(message: str):
        match = DateTimeExtractor.CLOCK_RE.search(message)
        if not match:
            return None

        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        amp = match.group(3)

        if amp:
            amp = amp.lower()
            if amp == "pm" and hour != 12:
                hour += 12
            if amp == "am" and hour == 12:
                hour = 0

        return hour, minute

    @staticmethod
    def extract_date_from_message(message: str):
        now = datetime.now()
        today = now.date()

        message_lower = message.lower()

        final_date = None

        for word, offset in DateTimeExtractor.RELATIVE_KEYWORDS.items():
            if word in message_lower:
                final_date = today + timedelta(days=offset)
                break

        if "next week" in message_lower:
            final_date = today + timedelta(days=7)

        if "next month" in message_lower:
            final_date = today + relativedelta(months=1)

        if final_date:
            return {"deadline": final_date.isoformat()}
        return {"deadline": None}


# COMMAND PARSER
class CommandParser:
    def __init__(self):
        self.datetime_extractor = DateTimeExtractor()

    def parse(self, message: str):
        message = message.strip()
        message_lower = message.lower()
        datetime_info = self.datetime_extractor.extract_date_from_message(message)

        def build(intent, id=None, worker_slug=None, depart_slug=None):
            return {
                "intent": intent,
                "id": id,
                "worker_slug": worker_slug,
                "depart_slug": depart_slug,
                "deadline": datetime_info.get("deadline"),
                "message": None,
            }

        if message_lower.startswith("/tasks"):
            return build("/tasks")
        if message_lower.startswith("/present"):
            return build("/present")
        if message_lower.startswith("/absent"):
            return build("/absent")
        if message_lower.startswith("/help"):
            return build("/help")
        if message_lower.startswith("/report"):
            return build("/report")
        if message_lower.startswith("/members"):
            return build("/members")
        if message_lower.startswith("/issues"):
            return build("/issues")
        if message_lower.startswith("/issue"):
            return build("/issue")
        if message_lower.startswith("/resolve"):
            task_id = re.search(r"\d+", message)
            return build(
                "resolve",
                int(task_id.group()) if task_id else None,
            )
        if message_lower.startswith("/complete"):
            task_id = re.search(r"\d+", message)
            return build(
                "complete",
                int(task_id.group()) if task_id else None,
            )

        return None


# INTENT CLASSIFIER
class IntentClassifier:
    VALID_INTENTS = {
        "tasks",
        "assign",
        "depart_assign",
        "mgrassign",
        "mgrself",
        "update",
        "issue",
        "issues",
        "resolve",
        "members",
        "report",
        "help",
        "present",
        "absent",
        "complete",
        "general_chat",
    }

    VALID_DEPARTMENTS = {
        "operations",
        "sales",
        "purchase",
        "it",
    }

    def __init__(self):
        self.datetime_extractor = DateTimeExtractor()
        print("✅ Hybrid Intent Classifier Loaded")

    # LIGHTWEIGHT EXTRACTION
    def extract_task_id(self, message: str):
        patterns = [
            r"task\s*(?:id)?\s*(\d+)",
            r"id\s*(\d+)",
            r"#(\d+)",
            r"(\d+)\s*wala\s*task",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def extract_mentions(self, message: str):
        mentions = re.findall(r"@(\w+)", message)
        if mentions:
            return f"@{mentions[0]}"
        return None

    # LLM CLASSIFICATION
    def llm_classify(self, message: str):
        prompt = f"""Classify the user message into one of these intents:

1. /tasks - User wants to see their tasks.
   Examples: mera kaam dikhao, my tasks, pending tasks, kya kaam hai

2. /assign - Assigning NEW work to a specific person.

3. /depart_assign - Assigning work to a department WITHOUT naming a person.
   Departments: operations, sales, purchase, it
   Examples: warehouse khali karo, invoice bhejo, server theek karo, raw material order karo

4. /mgrassign - Manager assigning EXISTING TASK to another person.
   Examples: task 32 ajay ko do, @rahul task 5 complete karo, id 4 priya ko assign karo

5. /mgrself - Manager taking task themselves.
   Examples: task 32 main karunga, main ye task kar lunga, i will do task 8

6. /complete - Task completed.
   Examples: task 4 complete, ho gaya, kar diya, done

7. /update - Updating existing task.
   Examples: task 4 delayed, task 8 pending, update task 2

8. /issue - Reporting an issue.

9. /issues - Viewing issues.
   Examples: show issues, active issues

10. /resolve - Resolving issues.
    Examples: resolve issue 4, issue theek ho gaya

11. /present - Attendance present.
    Examples: aa gaya hu, present hu, i am here

12. /absent - Attendance absent.
    Examples: nahi aaunga, absent hu, leave chahiye

13. /members - Viewing members.

14. /report - Generating report.

15. /help - Help request.

16. general_chat - Greetings, casual conversation.

IMPORTANT:
- Understand CONTEXT. Do not depend only on keywords.
- Hindi + Hinglish + English supported.
- Return ONLY JSON.

User message: {message}

Output format:
{{"intent": "string", "worker_slug": "string or null", "depart_slug": "operations|sales|purchase|it|null"}}

Rules:
- /assign => worker_slug required
- /depart_assign => depart_slug required
- /mgrassign => worker_slug required
- /mgrself => worker_slug null
- Others => both null"""

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are an intent classifier. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ]
        )

        result = json.loads(response.choices[0].message.content)
        return result

    def classify(self, message: str):
        # Extract task ID and mentions first
        task_id = self.extract_task_id(message)
        mention = self.extract_mentions(message)

        # Get LLM classification
        llm_result = self.llm_classify(message)

        intent = llm_result.get("intent")
        worker_slug = llm_result.get("worker_slug")
        depart_slug = llm_result.get("depart_slug")

        # Override with extracted mentions if present
        if mention and intent in ["assign", "mgrassign"]:
            worker_slug = mention

        # Get deadline
        datetime_info = self.datetime_extractor.extract_date_from_message(message)

        return {
            "intent": intent,
            "id": task_id,
            "worker_slug": worker_slug,
            "depart_slug": depart_slug,
            "deadline": datetime_info.get("deadline"),
            "message": None,
        }