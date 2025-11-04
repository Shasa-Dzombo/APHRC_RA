"""
Prompts for AI-generated research insights and analysis
"""

AI_INSIGHTS_PROMPT = '''As a research assistant, provide preliminary insights and potential answers to the following research questions. 
Use available web information and acknowledge data limitations.

Research Questions:
{main_questions}

Identified Data Gaps:
{data_gaps}

Web Information:
{web_results}

Please provide:
1. Preliminary answers to each research question
2. Key statistics and findings from available sources
3. Clear identification of limitations and uncertainties
4. Suggestions for further research needed
5. Citations for any specific claims or statistics

Format the response with clear headers and sections.
'''