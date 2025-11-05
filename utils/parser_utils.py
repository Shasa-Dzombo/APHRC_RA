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
    Parse the LLM response for sub-question mappings with ID linking.
    """
    logger.info("Parsing sub-question mappings")
    
    # Split on MAIN QUESTION headers since that's what we're getting
    sections = re.split(r"(?=MAIN\s*QUESTION\s*\d+:)", text, flags=re.I)
    mappings = []

    for section in sections[1:]:  # Skip first empty section
        try:
            lines = [l.strip() for l in section.split('\n') if l.strip()]
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

    logger.info(f"Successfully parsed {len(mappings)} sub-question mappings")
    return mappings

def parse_data_gaps(text: str) -> list:
    """Parse the text response to extract data gaps with multiple fallback strategies."""
    logger.info("Parsing data gaps from response")
    gaps = []
    
    # Strategy 1: Format with clear headers
    missing_var_pattern = r"MISSING VARIABLE:([^\n]+)(?:[\s\S]*?)GAP DESCRIPTION:([^\n]+)(?:[\s\S]*?)SUGGESTED SOURCES:([^\n]+)(?:[\s\S]*?)(?:SUB-QUESTION:([^\n]+)|$)"
    matches = re.finditer(missing_var_pattern, text, re.IGNORECASE)
    
    for match in matches:
        var_name = match.group(1).strip()
        description = match.group(2).strip()
        sources = match.group(3).strip()
        sub_question = match.group(4).strip() if match.group(4) else "General research question"
        
        if var_name.lower() != "variable":
            gaps.append({
                "missing_variable": var_name,
                "gap_description": description,
                "suggested_sources": sources,
                "sub_question": sub_question
            })
    
    if gaps:
        logger.info(f"Found {len(gaps)} data gaps using header format")
        return gaps
    
    # Strategy 2: Bullet points or numbered lists
    list_pattern = r"(?:\d+\.|\*|\-)[\s\t]*(?:Missing|Needed|Required):?\s*([a-zA-Z0-9_]+)(?:[\s\S]*?)(?:Description|Gap):?\s*([^\n]+)(?:[\s\S]*?)(?:Sources|Data sources):?\s*([^\n]+)"
    matches = re.finditer(list_pattern, text, re.IGNORECASE)
    
    for match in matches:
        var_name = match.group(1).strip()
        description = match.group(2).strip() if match.group(2) else "No description provided"
        sources = match.group(3).strip() if match.group(3) else "No sources specified"
        
        if var_name.lower() != "variable" and len(var_name) > 2:
            gaps.append({
                "missing_variable": var_name,
                "gap_description": description,
                "suggested_sources": sources,
                "sub_question": "From list item"
            })
    
    if gaps:
        logger.info(f"Found {len(gaps)} data gaps using list format")
        return gaps
    
    # Strategy 3: Sentence extraction
    sentences = re.split(r'[.!?]\s+', text)
    for sentence in sentences:
        if "missing" in sentence.lower() and any(term in sentence.lower() for term in ["data", "variable", "information"]):
            var_match = re.search(r'missing\s+([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+){0,2})', sentence.lower())
            if var_match:
                var_name = var_match.group(1)
                if var_name not in ["variable", "data", "information", "variables"] and len(var_name) > 3:
                    gaps.append({
                        "missing_variable": var_name,
                        "gap_description": sentence.strip(),
                        "suggested_sources": "Not specified in text",
                        "sub_question": "Extracted from context"
                    })
    
    # Fallback: Example data gaps
    if not gaps:
        gaps = [
            {
                "missing_variable": "demographic_data",
                "gap_description": "Detailed demographic breakdown of the population being studied",
                "suggested_sources": "National census, demographic health surveys",
                "sub_question": "General research question"
            }
        ]
        logger.warning("No data gaps found, using fallback examples")
    
    logger.info(f"Returning {len(gaps)} data gaps total")
    return gaps