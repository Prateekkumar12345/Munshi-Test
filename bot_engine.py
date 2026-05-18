"""
bot_engine.py
Production Hybrid Intent Classification Engine

Architecture:
1. Slash command parser
2. Lightweight deterministic extraction
3. LLM semantic intent classification
4. Validation layer
5. Structured JSON response

Supports:
- English
- Hindi
- Hinglish

Author: OpenAI Hybrid Architecture
"""

import json
import os
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================
# ENV
# ============================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

CHAT_MODEL = "gpt-4.1-mini"

# ============================================================
# DATE EXTRACTOR
# ============================================================


class DateTimeExtractor:

    RELATIVE_KEYWORDS = {
        "today": 0,
        "aaj": 0,
        "आज": 0,

        "tomorrow": 1,
        "kal": 1,
        "कल": 1,

        "parso": 2,
        "परसों": 2,
        "day after tomorrow": 2,
    }

    CLOCK_RE = re.compile(
        r"(\d{1,2})(?::(\d{2}))?\s*(am|pm|baje|bje|baj|बजे)?",
        re.IGNORECASE,
    )

    @staticmethod
    def parse_clock(message: str):

        match = DateTimeExtractor.CLOCK_RE.search(message)

        if not match:
            return None

        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0

        ampm = match.group(3)

        if ampm:

            ampm = ampm.lower()

            if ampm == "pm" and hour != 12:
                hour += 12

            elif ampm == "am" and hour == 12:
                hour = 0

            elif ampm in ["baje", "bje", "baj", "बजे"]:

                # heuristic
                if 1 <= hour <= 6:
                    hour += 12

        return hour, minute

    @staticmethod
    def extract_date_from_message(message: str):

        now = datetime.now()
        today = now.date()

        ml = message.lower()

        final_date = None

        # relative keywords
        for word, offset in DateTimeExtractor.RELATIVE_KEYWORDS.items():

            if word in ml:
                final_date = today + timedelta(days=offset)
                break

        # next week
        if "next week" in ml or "agle hafte" in ml:
            final_date = today + timedelta(days=7)

        # next month
        if "next month" in ml or "agle mahine" in ml:
            final_date = today + relativedelta(months=1)

        # explicit date dd-mm-yyyy
        explicit_date = re.search(
            r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})",
            message,
        )

        if explicit_date:

            day, month, year = explicit_date.groups()

            try:

                final_date = datetime(
                    int(year),
                    int(month),
                    int(day)
                ).date()

            except Exception:
                pass

        if not final_date:
            return {
                "deadline": None,
                "type": None,
            }

        clock = DateTimeExtractor.parse_clock(message)

        if clock:

            hour, minute = clock

            dt = datetime(
                final_date.year,
                final_date.month,
                final_date.day,
                hour,
                minute,
                0,
            )

            return {
                "deadline": dt.isoformat(),
                "type": "datetime",
            }

        return {
            "deadline": final_date.strftime("%Y-%m-%d"),
            "type": "date",
        }


# ============================================================
# GENERAL CHAT RESPONSE
# ============================================================


def get_general_chat_response(message: str):

    return (
        "Main task management, attendance, assignment aur reports mein help kar sakta hoon. "
        "Commands dekhne ke liye /help type karo."
    )


# ============================================================
# COMMAND PARSER
# ============================================================


class CommandParser:

    def __init__(self):

        self.datetime_extractor = DateTimeExtractor()

    def parse(self, message: str):

        message = message.strip()

        ml = message.lower()

        datetime_info = self.datetime_extractor.extract_date_from_message(message)

        def build(
            intent,
            id=None,
            worker_slug=None,
            depart_slug=None,
        ):

            return {
                "intent": intent,
                "id": id,
                "worker_slug": worker_slug,
                "depart_slug": depart_slug,
                "deadline": datetime_info.get("deadline"),
                "message": None,
            }

        # ====================================================
        # COMMANDS
        # ====================================================

        if ml.startswith("/tasks"):
            return build("/tasks")

        if ml.startswith("/present"):
            return build("/present")

        if ml.startswith("/absent"):
            return build("/absent")

        if ml.startswith("/help"):
            return build("/help")

        if ml.startswith("/report"):
            return build("/report")

        if ml.startswith("/members"):
            return build("/members")

        if ml.startswith("/issues"):
            return build("/issues")

        if ml.startswith("/issue"):
            return build("/issue")

        if ml.startswith("/resolve"):

            task_id = re.search(r"\d+", message)

            return build(
                "/resolve",
                int(task_id.group()) if task_id else None,
            )

        if ml.startswith("/complete"):

            task_id = re.search(r"\d+", message)

            return build(
                "/complete",
                int(task_id.group()) if task_id else None,
            )

        if ml.startswith("/update"):

            task_id = re.search(r"\d+", message)

            return build(
                "/update",
                int(task_id.group()) if task_id else None,
            )

        return None


# ============================================================
# INTENT CLASSIFIER
# ============================================================


class IntentClassifier:

    VALID_INTENTS = {
        "/tasks",
        "/assign",
        "/depart_assign",
        "/mgrassign",
        "/mgrself",
        "/update",
        "/issue",
        "/issues",
        "/resolve",
        "/members",
        "/report",
        "/help",
        "/present",
        "/absent",
        "/complete",
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

    # ========================================================
    # ENTITY EXTRACTION
    # ========================================================

    def extract_task_id(self, message: str):

        patterns = [

            r"task\s*(?:id)?\s*(\d+)",

            r"id\s*(\d+)",

            r"#(\d+)",

            r"(\d+)\s*wala\s*task",

            r"task\s*number\s*(\d+)",

            r"task\s*no\s*(\d+)",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                message,
                re.IGNORECASE,
            )

            if match:
                return int(match.group(1))

        return None

    def extract_mentions(self, message: str):

        mentions = re.findall(
            r"@(\w+)",
            message,
        )

        if mentions:
            return f"@{mentions[0]}"

        return None

    # ========================================================
    # LLM CLASSIFICATION
    # ========================================================

    def llm_classify(self, message: str):

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content": self.build_system_prompt(),
                },
                {
                    "role": "user",
                    "content": message,
                },
            ],
        )

        raw = response.choices[0].message.content

        try:
            return json.loads(raw)

        except Exception:

            return {
                "intent": "general_chat",
                "worker_slug": None,
                "depart_slug": None,
            }

    # ========================================================
    # MAIN PIPELINE
    # ========================================================

    def classify(self, message: str):

        datetime_info = self.datetime_extractor.extract_date_from_message(message)

        task_id = self.extract_task_id(message)

        mention = self.extract_mentions(message)

        llm_result = self.llm_classify(message)

        intent = llm_result.get(
            "intent",
            "general_chat",
        )

        # validate intent
        if intent not in self.VALID_INTENTS:
            intent = "general_chat"

        worker_slug = llm_result.get("worker_slug")

        depart_slug = llm_result.get("depart_slug")

        # mention override
        if mention and not worker_slug:
            worker_slug = mention

        # department validation
        if depart_slug:

            if depart_slug not in self.VALID_DEPARTMENTS:
                depart_slug = None

        result = {
            "intent": intent,
            "id": task_id,
            "worker_slug": worker_slug,
            "depart_slug": depart_slug,
            "deadline": datetime_info.get("deadline"),
            "message": None,
        }

        # general response
        if intent == "general_chat":

            result["message"] = get_general_chat_response(message)

        return result

    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    def build_system_prompt(self):

        return """
    You are an enterprise multilingual intent classification engine.

You understand:
- English
- Hindi
- Hinglish

You ONLY return valid JSON.

========================================================
CORE UNDERSTANDING RULE
========================================================

You MUST understand SEMANTIC MEANING, not just keywords.

VERY IMPORTANT:

There is a BIG difference between:

1. INSTRUCTING someone to do work
2. CONFIRMING work is already completed

Examples:

"complete the work"
-> instruction
-> /assign or /depart_assign

"finish the task"
-> instruction
-> /assign or /depart_assign

"task complete ho gaya"
-> already completed
-> /complete

"done"
-> already completed
-> /complete

"kar diya"
-> already completed
-> /complete

========================================================
SUPPORTED INTENTS
========================================================

1. /tasks
User wants to view their tasks.

Examples:
- mera kaam dikhao
- my tasks
- pending tasks
- mujhe kya karna hai
- task list dikhao

========================================================

2. /assign

Assigning NEW work to a specific person.

This means:
- user is instructing a person to do work
- NO existing task reference

Examples:
- ajay ko warehouse khali karne bolo
- @rahul invoice bhejdo
- priya client ko call kare
- ajay complete the work
- complete the work ajay
- rahul ye task finish karo

IMPORTANT:
If user is telling someone to complete/finish work,
it is STILL assignment.

Rules:
- worker_slug required
- depart_slug null

========================================================

3. /depart_assign

Assigning work to a department WITHOUT naming a person.

Departments:
- operations
- sales
- purchase
- it

Examples:
- warehouse khali karo
- invoice bhejo
- server theek karo
- raw material order karo
- complete the dispatch work
- finish warehouse cleaning

Rules:
- depart_slug required
- worker_slug null

========================================================

4. /mgrassign

Manager assigning EXISTING TASK to another person.

This ONLY applies when:
- existing task reference present
AND
- another person mentioned

Task references:
- task 5
- id 7
- #9
- 4 wala task

Examples:
- task 32 ajay ko do
- task 5 @rahul ko assign karo
- id 4 priya ko de do
- task 9 ajay complete karo

Rules:
- worker_slug required
- existing task reference required

========================================================

5. /mgrself

Manager taking task themselves.

Examples:
- task 32 main karunga
- main ye task kar lunga
- i will do task 8
- task 9 mai khud karunga

Rules:
- worker_slug null

========================================================

6. /complete

ONLY when the user is CONFIRMING
that work is ALREADY FINISHED.

This is NOT an instruction.

Examples:
- ho gaya
- kar diya
- done
- completed
- task complete ho gaya
- work finished
- task khatam ho gaya
- complete kar diya
- dispatch done

IMPORTANT:
If user is INSTRUCTING someone to complete work,
then it is NOT /complete.

Examples:
- complete the work
- finish the task
- ajay complete this
- warehouse complete karo

These are assignments.

========================================================

7. /update

Updating status of existing work/task.

Examples:
- task delayed hai
- task pending hai
- update task 4
- task 5 hold pe hai

========================================================

8. /issue

Reporting a new issue/problem.

Examples:
- machine kharab hai
- issue hai
- server down hai
- printer kaam nahi kar raha

========================================================

9. /issues

Viewing issues.

Examples:
- active issues
- show issues
- issues dikhao

========================================================

10. /resolve

Resolving issue.

Examples:
- resolve issue 4
- issue theek ho gaya
- problem solved

========================================================

11. /present

Attendance present.

Examples:
- aa gaya hu
- present hu
- i am here
- office aa gaya

========================================================

12. /absent

Attendance absent.

Examples:
- absent hu
- nahi aaunga
- leave chahiye
- aaj nahi aa paunga

========================================================

13. /members

Viewing members/team.

Examples:
- members dikhao
- team members
- employee list

========================================================

14. /report

Generating reports.

Examples:
- report generate karo
- daily report
- weekly report

========================================================

15. /help

Help request.

Examples:
- help
- commands batao
- kaise use kare

========================================================

16. general_chat

Casual conversation only.

Examples:
- hello
- hi
- kaise ho
- good morning

========================================================
DEPARTMENT ROUTING RULES
========================================================

operations:
- warehouse
- dispatch
- logistics
- delivery
- inventory
- production
- machine
- packaging
- loading/unloading

sales:
- invoice
- customer
- client
- quotation
- payment
- order

purchase:
- vendor
- supplier
- procurement
- raw material
- buying
- sourcing

it:
- server
- laptop
- computer
- software
- internet
- wifi
- printer

========================================================
OUTPUT FORMAT
========================================================

Return ONLY valid JSON.

{
  "intent": "string",
  "worker_slug": "string or null",
  "depart_slug": "operations|sales|purchase|it|null"
}

========================================================
STRICT RULES
========================================================

- /assign => worker_slug required
- /mgrassign => worker_slug required
- /depart_assign => depart_slug required
- /mgrself => worker_slug null
- all others => both null

Never return explanations.
Never return markdown.
Only return JSON.
"""