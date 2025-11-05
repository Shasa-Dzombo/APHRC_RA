"""
Utility functions for formatting responses
"""
import re

def format_answer_content(content: str) -> str:
    """
    Format the answer content by:
    1. Removing any generated questions
    2. Structuring the content with proper line breaks
    3. Keeping only relevant sections
    4. Ensuring consistent spacing
    """
    # Remove everything between MAIN QUESTION and DIRECT ANSWER (including both patterns)
    content = re.sub(r'\*\*MAIN QUESTION[^*]*?\*\*DIRECT ANSWER:', '**DIRECT ANSWER:**', content, flags=re.DOTALL)
    
    # Remove any remaining main questions and sections
    content = re.sub(r'\*\*MAIN QUESTION.*?(?=\*\*|$)', '', content, flags=re.DOTALL)
    
    # Remove any sub-questions sections
    content = re.sub(r'\*\*SUB-QUESTIONS:\*\*.*?(?=\*\*|$)', '', content, flags=re.DOTALL)
    
    # Define expected sections
    sections = [
        "**DIRECT ANSWER:**",
        "**DATA-DRIVEN INSIGHTS:**",
        "**ANALYTICAL METHODOLOGY:**",
        "**RESEARCH IMPLICATIONS:**",
        "**LIMITATIONS & RECOMMENDATIONS:**"
    ]
    
    formatted_parts = []
    
    # Process each section
    for i in range(len(sections)):
        current_section = sections[i]
        next_section = sections[i + 1] if i + 1 < len(sections) else None
        
        if current_section in content:
            start = content.index(current_section)
            end = content.index(next_section) if next_section and next_section in content else None
            
            # Extract section content
            section_content = content[start:end].strip() if end else content[start:].strip()
            
            # Format the section content
            lines = section_content.split('\n')
            # Keep the header line
            header = lines[0]
            # Join content lines with proper spacing, removing empty lines
            content_lines = [line.strip() for line in lines[1:] if line.strip()]
            # Join content lines into a single paragraph
            content_text = ' '.join(content_lines)
            
            # Combine header with formatted content
            formatted_section = f"{header}\n{content_text}"
            formatted_parts.append(formatted_section)
    
    # Join all sections with double line breaks
    return '\n\n'.join(formatted_parts)