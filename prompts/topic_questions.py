"""
Prompts for generating research questions from topic-based research
"""

TOPIC_QUESTIONS_PROMPT = '''Based on the research topic: {title}
Description: {description}
Area of Study: {area_of_study}
Geography: {geography}

Please generate a research plan following this EXACT format:

MAIN QUESTION:
[Single clear research question that addresses the core research objective]

SUB-QUESTIONS:
- [First sub-question that explores a specific aspect]
- [Second sub-question that investigates related factors]
- [Third sub-question that examines implications or outcomes]

Focus on questions that can lead to meaningful insights and actionable recommendations.
'''