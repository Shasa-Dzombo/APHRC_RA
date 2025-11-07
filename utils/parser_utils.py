"""
Parsing utilities for research workflow responses
"""
import re
import uuid
from typing import Dict, List, Any

import logging as logger


def parse_main_and_sub_questions(text: str) -> dict:
    """Parse the text response to extract multiple main questions and their sub-questions."""
    logger.info("Starting to parse main and sub-questions")
    
    lines = text.strip().split('\n')
    main_questions = []
    sub_questions_map = {}
    
    current_main_index = -1
    collecting_sub_questions = False
    
    for line in lines:
        line = line.strip()
        
        # Check for main question headers (MAIN QUESTION 1:, MAIN QUESTION 2:, etc.)
        if re.match(r"^MAIN\s*QUESTION\s*\d+:", line, re.I):
            # Extract the main question text
            question_text = re.sub(r"^MAIN\s*QUESTION\s*\d+:\s*", "", line, flags=re.I).strip()
            if question_text:
                current_main_index = len(main_questions)
                main_questions.append(question_text)
                sub_questions_map[current_main_index] = []
                collecting_sub_questions = False
                logger.debug(f"Found main question: {question_text}")
            continue
        
        # Check for sub-questions header
        elif re.match(r"SUB-QUESTIONS:", line, re.I):
            collecting_sub_questions = True
            if current_main_index not in sub_questions_map:
                sub_questions_map[current_main_index] = []
            continue
        
        # Collect main question text (if not already captured)
        elif current_main_index >= 0 and len(main_questions) == current_main_index + 1 and line and not collecting_sub_questions:
            # If main question was empty, use this line
            if not main_questions[current_main_index]:
                main_questions[current_main_index] = line
                logger.debug(f"Updated main question: {line}")
        
        # Collect sub-questions
        elif collecting_sub_questions and line.startswith("- "):
            if current_main_index in sub_questions_map:
                sub_q = line[2:].strip()
                sub_questions_map[current_main_index].append(sub_q)
                logger.debug(f"Found sub-question: {sub_q}")
    
    # Convert to the expected format
    all_sub_questions = []
    for main_idx, sub_list in sub_questions_map.items():
        for sub_q in sub_list:
            all_sub_questions.append({
                "text": sub_q,
                "main_question_index": main_idx,
                "main_question_text": main_questions[main_idx] if main_idx < len(main_questions) else ""
            })
    
    result = {
        "main_questions": main_questions,
        "sub_questions": all_sub_questions,
        "sub_questions_by_main": sub_questions_map
    }
    
    logger.info(f"Parsed {len(main_questions)} main questions and {len(all_sub_questions)} sub-questions")
    return result

def parse_subquestion_mappings(text: str, sub_questions: list = None) -> list:
    """
    Parse the LLM response for each SUB-QUESTION section instead of MAIN QUESTION.
    Links parsed data and analysis back to existing sub-question IDs.
    """
    logger.info("Parsing sub-question mappings")
    
    # Split on MAIN QUESTION headers since that's what we're getting
    sections = re.split(r"(?=MAIN\s*QUESTION\s*\d+:)", text, flags=re.I)
    mappings = []

    for section in sections[1:]:  # Skip first empty section
        try:
            lines = [l.strip() for l in section.split("\n") if l.strip()]
            if not lines:
                continue

            # Extract question text from first line
            question_text = re.sub(r"^MAIN\s*QUESTION\s*\d+:\s*", "", lines[0], flags=re.I).strip()
            if not question_text:
                question_text = lines[1].strip()  # Get from next line if not on header line

            # Find matching sub-question ID
            sub_id = None
            if sub_questions:
                for sq in sub_questions:
                    if (question_text.lower() in sq["text"].lower() or 
                        sq["text"].lower() in question_text.lower()):
                        sub_id = sq["id"]
                        break

            # Initialize sections
            data_requirements = []
            analysis_approach = []
            current_section = None
            
            # Log found question
            logger.info("Processing question: %s", question_text)

            # Parse DATA REQUIREMENTS and ANALYSIS APPROACH blocks
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

                if current_section == "data" and line.startswith("-"):
                    data_requirements.append(line[1:].strip())
                elif current_section == "analysis" and line.startswith("-"):
                    analysis_approach.append(line[1:].strip())

            # Build mapping with fallback for empty sections
            mapping = {
                "sub_question_id": sub_id or str(uuid.uuid4()),  # Generate ID if none found
                "sub_question": question_text,
                "data_requirements": "\n".join("- " + r for r in data_requirements) if data_requirements else "- None specified",
                "analysis_approach": "\n".join("- " + a for a in analysis_approach) if analysis_approach else "- None specified",
            }
            mappings.append(mapping)
            logger.info(f"Added mapping for question: {question_text} (ID: {mapping['sub_question_id']})")

        except Exception as e:
            logger.error(f"Error parsing section: {e}")
            continue

    print(f"\nSuccessfully parsed {len(mappings)} sub-question mappings.")
    return mappings