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
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for main question headers (MAIN QUESTION 1:, MAIN QUESTION 2:, etc.)
        if re.match(r"^MAIN\s*QUESTION\s*\d+:", line, re.I):
            # Extract main question text
            question_text = re.sub(r"^MAIN\s*QUESTION\s*\d+:\s*", "", line, flags=re.I).strip()
            if not question_text and lines.index(line) + 1 < len(lines):
                # Get the question from the next line if the header line is empty
                next_line = lines[lines.index(line) + 1].strip()
                if next_line and not next_line.startswith("SUB-QUESTIONS:"):
                    question_text = next_line
            
            if question_text:
                current_question = question_text
                main_questions.append(question_text)
                sub_questions_map[len(main_questions) - 1] = []
                in_sub_questions = False
                logger.debug(f"Found main question: {question_text}")
            continue
        
        # Check for sub-questions section
        if line.startswith("SUB-QUESTIONS:"):
            in_sub_questions = True
            continue
            
        # Collect sub-questions
        if line.startswith("- ") and current_question is not None and in_sub_questions:
            sub_q = line[2:].strip()  # Remove the "- " prefix
            current_index = len(main_questions) - 1
            if current_index >= 0:
                sub_questions_map[current_index].append(sub_q)
                logger.debug(f"Found sub-question: {sub_q}")
                
        # Reset sub-questions flag if we hit another section
        elif line.endswith(":") and not line.startswith("SUB-QUESTIONS:"):
            in_sub_questions = False
    
    # Convert to the expected format
    all_sub_questions = []
    for main_idx, sub_list in sub_questions_map.items():
        for sub_q in sub_list:
            all_sub_questions.append({
                "text": sub_q,
                "main_question_index": main_idx,
                "main_question_text": main_questions[main_idx],
                "id": str(uuid.uuid4())
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
    Parse the LLM response for sub-question mappings with ID linking.
    """
    logger.info("Parsing sub-question mappings")
    
    # Split on MAIN QUESTION headers since that's what we're getting
    sections = re.split(r"(?=MAIN\s*QUESTION\s*\d+:)", text, flags=re.I)
    mappings = []
    
    current_data_reqs = []
    current_analysis = []

    for section in sections[1:]:  # Skip first empty section
        try:
            lines = [l.strip() for l in section.split('\n') if l.strip()]
            if not lines:
                continue

            # Extract main question text from first line
            main_question_text = re.sub(r"^MAIN\s*QUESTION\s*\d+:\s*", "", lines[0], flags=re.I).strip()
            if not main_question_text:
                main_question_text = lines[1].strip()  # Get from next line if not on header line

            # Initialize sections for this main question
            sub_questions_list = []
            collecting_sub_questions = False
            
            # Parse each line looking for sub-questions and their data
            current_section = None
            data_requirements = []
            analysis_approach = []
            
            for line in lines[1:]:
                if re.match(r"SUB-QUESTIONS:", line, re.I):
                    collecting_sub_questions = True
                    continue
                    
                elif re.match(r"DATA REQUIREMENTS:", line, re.I):
                    collecting_sub_questions = False
                    current_section = "data"
                    continue
                    
                elif re.match(r"ANALYSIS APPROACH:", line, re.I):
                    collecting_sub_questions = False
                    current_section = "analysis"
                    continue
                    
                if collecting_sub_questions and line.startswith("- "):
                    sub_questions_list.append(line[2:].strip())
                    
                elif current_section == "data" and line.startswith("- "):
                    data_requirements.append(line[2:].strip())
                    
                elif current_section == "analysis" and line.startswith("- "):
                    analysis_approach.append(line[2:].strip())
            
            # Create mapping for each sub-question
            for sub_q in sub_questions_list:
                sub_id = None
                if sub_questions:
                    for sq in sub_questions:
                        if (sub_q.lower() in sq["text"].lower() or 
                            sq["text"].lower() in sub_q.lower()):
                            sub_id = sq["id"]
                            break
                
                mapping = {
                    "sub_question_id": sub_id or str(uuid.uuid4()),  # Generate ID if none found
                    "sub_question": sub_q,
                    "data_requirements": "\n".join("- " + r for r in data_requirements) if data_requirements else "- None specified",
                    "analysis_approach": "\n".join("- " + a for a in analysis_approach) if analysis_approach else "- None specified"
                }
                mappings.append(mapping)
                logger.info(f"Added mapping for sub-question: {sub_q} (ID: {mapping['sub_question_id']})")

        except Exception as e:
            logger.error(f"Error parsing section: {e}")
            continue

    logger.info(f"Successfully parsed {len(mappings)} sub-question mappings")
    return mappings