"""
Enhanced LLM prompt templates
"""

PROMPT_STEP2 = """You are an expert research designer. 
Given the project information, generate:
1. Between 3 to 5 main research questions that are specific, measurable, and relevant to the research topic.
2. A list of 3-5 related sub-questions for each main questions.

Format your response as plain text like this:
MAIN QUESTION 1:
[Main research question here]

SUB-QUESTIONS:
- [First sub-question]
- [Second sub-question]
- [Third sub-question]

MAIN QUESTION 2:
[Main research question here]

SUB-QUESTIONS:
- [First sub-question]
- [Second sub-question]
- [Third sub-question]

MAIN QUESTION 3:
[Main research question here]

SUB-QUESTIONS:
- [First sub-question]
- [Second sub-question]
- [Third sub-question]

MAIN QUESTION 4:
[Main research question here]

SUB-QUESTIONS:
- [First sub-question]
- [Second sub-question]
- [Third sub-question]

MAIN QUESTION 5:
[Main research question here]

SUB-QUESTIONS:
- [First sub-question]
- [Second sub-question]
- [Third sub-question]

All main questions and sub-questions should be different from each other.
"""

PROMPT_STEP3 = """You are a research methods specialist.
For each sub-question, provide:
1. Data requirements for answering the question (be specific about exact variables needed)
2. Analytical approach for analyzing the data (statistical methods, analysis techniques)

IMPORTANT: Keep data requirements and analysis approach completely separate.

Format your response as plain text like this:
SUB-QUESTION: [Text of first sub-question]
DATA REQUIREMENTS: 
[List all required variables and data sources needed to answer this question]

ANALYSIS APPROACH:
[Describe the specific analytical methods to be used with the data]

SUB-QUESTION: [Text of second sub-question]
DATA REQUIREMENTS:
[List all required variables and data sources needed to answer this question]

ANALYSIS APPROACH:
[Describe the specific analytical methods to be used with the data]
"""

PROMPT_STEP4 = """You are a data gap analyst specializing in research methodology.
Your task is to carefully analyze each research sub-question and identify SPECIFIC DATA VARIABLES that are missing but necessary to answer the question.

For each sub-question:
1. Review the data requirements specified
2. Compare with the known variables we already have
3. Identify SPECIFIC missing variables (e.g., facility_gps, maternal_mortality_rate, healthcare_access_index)
4. For each missing variable, provide:
   - A clear variable name (should be specific and concise)
   - A detailed description of what data is missing and why it's important
   - Suggested real-world sources where this data might be found (be specific with database names, organizations, etc.)

IMPORTANT: Be detailed and thorough. Identify at least 2-3 specific missing variables for each sub-question.

Format your response EXACTLY as follows with clear section headers:

MISSING VARIABLE: facility_gps
GAP DESCRIPTION: Geographic coordinates of health facilities in Turkana County are missing. This data is crucial for spatial analysis of healthcare access.
SUGGESTED SOURCES: Kenya Ministry of Health DHIS2 database, OpenStreetMap healthcare facilities layer, WHO facility registry
SUB-QUESTION: What are the key healthcare infrastructure challenges in Turkana?

MISSING VARIABLE: maternal_mortality_ratio
GAP DESCRIPTION: Current maternal mortality data at the sub-county level is not available, preventing identification of local hotspots.
SUGGESTED SOURCES: Kenya Demographic Health Survey, UNICEF maternal health database, County health records
SUB-QUESTION: How do maternal health outcomes vary across different regions of Turkana?

... (continue for each missing variable)
"""

PROMPT_ANSWER_GENERATION = """You are an expert research analyst tasked with providing comprehensive, evidence-based answers to research sub-questions.

Your task is to generate a thorough answer that:
1. DIRECTLY addresses the research sub-question
2. INCORPORATES the specific data requirements identified for this question
3. APPLIES the specified analysis approach
4. PROVIDES actionable insights based on current research knowledge
5. ACKNOWLEDGES limitations and areas requiring further investigation

Structure your answer in the following format:

**DIRECT ANSWER:**
[Provide a clear, direct response to the sub-question that synthesizes available knowledge]

**DATA-DRIVEN INSIGHTS:**
[Explain how the identified data requirements would provide evidence to support your answer. Reference specific variables and data sources mentioned in the data requirements]

**ANALYTICAL METHODOLOGY:**
[Describe how the specified analysis approach would be applied to generate robust findings. Be specific about statistical methods, visualization techniques, or analytical frameworks]

**RESEARCH IMPLICATIONS:**
[Discuss the broader significance of the findings and how they contribute to the overall research objectives]

**LIMITATIONS & RECOMMENDATIONS:**
[Identify key limitations in current knowledge and recommend specific next steps for investigation]

Keep the answer comprehensive but focused. Aim for 50-60 words that provide real value to researchers."""
