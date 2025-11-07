"""
Enhanced Parsing Utilities for Research API
"""
import re
import uuid
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

def parse_main_and_sub_questions(text: str) -> dict:
    """Parse the text response to extract multiple main questions and their sub-questions."""
    logger.info("Starting to parse main and sub-questions")

    lines = text.strip().split('\n')
    main_questions = []
    sub_questions_map = {}
    current_question = None
    in_sub_questions = False

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        logger.debug(f"Processing line {i + 1}: {line}")

        # Check for main question headers (MAIN QUESTION 1:, MAIN QUESTION 2:, etc.)
        if re.match(r"^MAIN\s*QUESTION\s*\d+:", line, re.I):
            question_text = re.sub(r"^MAIN\s*QUESTION\s*\d+:\s*", "", line, flags=re.I).strip()
            if question_text:
                current_question = question_text
                main_questions.append(question_text)
                sub_questions_map[len(main_questions) - 1] = []
                in_sub_questions = False
                logger.debug(f"Found main question: {question_text}")
            continue

        # Check for sub-questions section
        if re.match(r"^SUB-QUESTIONS:\s*$", line, re.I):
            in_sub_questions = True
            logger.debug("Entering sub-questions section")
            continue

        # Collect sub-questions
        if in_sub_questions and line.startswith("- "):
            sub_q = line[2:].strip()
            if current_question:
                sub_questions_map[len(main_questions) - 1].append(sub_q)
                logger.debug(f"Found sub-question: {sub_q}")

        # Reset sub-questions flag if another section starts
        elif line.endswith(":") and not re.match(r"^SUB-QUESTIONS:\s*$", line, re.I):
            in_sub_questions = False
            logger.debug("Exiting sub-questions section")

    # Convert to the expected format
    all_sub_questions = [
        {
            "text": sub_q,
            "main_question_index": main_idx,
            "main_question_text": main_questions[main_idx],
            "id": str(uuid.uuid4())
        }
        for main_idx, sub_list in sub_questions_map.items()
        for sub_q in sub_list
    ]

    result = {
        "main_questions": main_questions,
        "sub_questions": all_sub_questions,
        "sub_questions_by_main": sub_questions_map
    }

    logger.info(f"Parsed {len(main_questions)} main questions and {len(all_sub_questions)} sub-questions")
    return result

def parse_subquestion_mappings(text: str, sub_questions: list = None) -> list:
    """
    Parse the LLM response for sub-question mappings with ID linking.
    """
    logger.info("Parsing sub-question mappings")

    sections = re.split(r"(?=MAIN\s*QUESTION\s*\d+:)", text, flags=re.I)
    mappings = []

    for i, section in enumerate(sections[1:], start=1):
        try:
            lines = [l.strip() for l in section.split('\n') if l.strip()]
            if not lines:
                continue

            logger.debug(f"Processing section {i}: {lines[0]}")

            main_question_text = re.sub(r"^MAIN\s*QUESTION\s*\d+:\s*", "", lines[0], flags=re.I).strip()

            sub_questions_list = []
            data_requirements = []
            analysis_approach = []
            current_section = None

            for line in lines[1:]:
                if re.match(r"^SUB-QUESTIONS:\s*$", line, re.I):
                    current_section = "sub_questions"
                    logger.debug("Entering sub-questions section")
                    continue
                elif re.match(r"^DATA REQUIREMENTS:\s*$", line, re.I):
                    current_section = "data"
                    logger.debug("Entering data requirements section")
                    continue
                elif re.match(r"^ANALYSIS APPROACH:\s*$", line, re.I):
                    current_section = "analysis"
                    logger.debug("Entering analysis approach section")
                    continue

                if current_section == "sub_questions" and line.startswith("- "):
                    sub_questions_list.append(line[2:].strip())
                    logger.debug(f"Found sub-question: {line[2:].strip()}")
                elif current_section == "data" and line.startswith("- "):
                    data_requirements.append(line[2:].strip())
                    logger.debug(f"Found data requirement: {line[2:].strip()}")
                elif current_section == "analysis" and line.startswith("- "):
                    analysis_approach.append(line[2:].strip())
                    logger.debug(f"Found analysis approach: {line[2:].strip()}")

            for sub_q in sub_questions_list:
                sub_id = next((sq["id"] for sq in sub_questions if sq["text"].lower() == sub_q.lower()), str(uuid.uuid4()))
                mappings.append({
                    "sub_question_id": sub_id,
                    "sub_question": sub_q,
                    "data_requirements": "\n".join(f"- {r}" for r in data_requirements) or "- None specified",
                    "analysis_approach": "\n".join(f"- {a}" for a in analysis_approach) or "- None specified"
                })
                logger.info(f"Added mapping for sub-question: {sub_q} (ID: {sub_id})")

        except Exception as e:
            logger.error(f"Error parsing section {i}: {e}")

    logger.info(f"Successfully parsed {len(mappings)} sub-question mappings")
    return mappings