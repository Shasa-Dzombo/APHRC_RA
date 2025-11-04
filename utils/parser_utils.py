"""
Parsing utilities for research workflow responses
"""
import re
import uuid
from typing import Dict, List, Any


def parse_main_and_sub_questions(text: str) -> dict:
    lines = text.strip().split('\n')
    main_questions = []
    sub_questions_map = {}
    current_main_index = -1
    collecting_sub_questions = False

    for line in lines:
        line = line.strip()
        if re.match(r"main\s*question", line, re.I):
            current_main_index = len(main_questions)
            collecting_sub_questions = False
            continue

        elif re.match(r"sub-questions", line, re.I):
            collecting_sub_questions = True
            if current_main_index not in sub_questions_map:
                sub_questions_map[current_main_index] = []
            continue

        elif current_main_index >= 0 and len(main_questions) == current_main_index and line and not collecting_sub_questions:
            main_questions.append(line)

        elif collecting_sub_questions and re.match(r"^- ", line):
            sub_questions_map[current_main_index].append(line[2:].strip())

    all_sub_questions = []
    for main_idx, sub_list in sub_questions_map.items():
        for sub_q in sub_list:
            if main_questions and 0 <= main_idx < len(main_questions):
                all_sub_questions.append({
                    "text": sub_q,
                    "main_question_index": main_idx,
                    "main_question_text": main_questions[main_idx]
                })

    return {
        "main_questions": main_questions,
        "sub_questions": all_sub_questions,
        "sub_questions_by_main": sub_questions_map
    }


def parse_subquestion_mappings(text: str, sub_questions: list = None) -> list:
    """
    Parse the LLM response for each SUB-QUESTION section instead of MAIN QUESTION.
    Links parsed data and analysis back to existing sub-question IDs.
    """
    print("\nParsing LLM response (subquestion mappings)...")

    # Split on SUB-QUESTION headers instead of MAIN QUESTION
    sections = re.split(r"(?=SUB[-\s]*QUESTION)", text, flags=re.I)
    mappings = []

    for section in sections[1:]:
        try:
            lines = [l.strip() for l in section.split("\n") if l.strip()]
            if not lines:
                continue

            # Extract sub-question text
            first_line = re.sub(r"(?i)sub[-\s]*question\s*\d*:?", "", lines[0]).strip()
            sub_question_text = first_line

            # Match to known sub-question list if provided
            sub_id = None
            if sub_questions:
                for sq in sub_questions:
                    if sub_question_text.lower() in sq["text"].lower() or sq["text"].lower() in sub_question_text.lower():
                        sub_id = sq["id"]
                        break

            if not sub_id:
                print(f"Warning: Could not find ID for sub-question: {sub_question_text[:80]}")
                continue

            data_requirements = []
            analysis_approach = []
            current_section = None

            # Parse DATA REQUIREMENTS and ANALYSIS APPROACH blocks
            for line in lines[1:]:
                if re.search(r"data\s*requirements", line, re.I):
                    current_section = "data"
                    continue
                elif re.search(r"analysis\s*approach", line, re.I):
                    current_section = "analysis"
                    continue

                if current_section == "data" and line.startswith("-"):
                    data_requirements.append(line[1:].strip())
                elif current_section == "analysis" and line.startswith("-"):
                    analysis_approach.append(line[1:].strip())

            # Build mapping
            mapping = {
                "sub_question_id": sub_id,
                "sub_question": sub_question_text,
                "data_requirements": "\n".join("- " + r for r in data_requirements) if data_requirements else "- None specified",
                "analysis_approach": "\n".join("- " + a for a in analysis_approach) if analysis_approach else "- None specified",
            }
            mappings.append(mapping)
            print(f"Linked sub-question: {sub_question_text[:80]} (ID: {sub_id})")

        except Exception as e:
            print(f"Error parsing section: {e}")
            continue

    print(f"\nSuccessfully parsed {len(mappings)} sub-question mappings.")
    return mappings