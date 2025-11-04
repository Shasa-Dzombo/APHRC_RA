"""
Prompts for generating research questions from dataset analysis
"""

DATASET_QUESTIONS_PROMPT = '''Based on the dataset: {table_name}
Variables: {variables}
Description: {description}

Please generate a research plan following this EXACT format:

MAIN QUESTION:
[Single clear research question]

SUB-QUESTIONS:
- [First sub-question]
- [Second sub-question]
- [Third sub-question]

Each question should be focused on analyzing relationships and patterns in the available variables.
'''