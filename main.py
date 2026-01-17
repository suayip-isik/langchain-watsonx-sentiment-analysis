import os
from dotenv import load_dotenv

from langchain_ibm import WatsonxLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# Load environment variables from .env file
load_dotenv()

# Set up credentials from .env file
credentials = {
    "url": os.getenv("WATSONX_URL"),
    "apikey": os.getenv("WATSONX_APIKEY"),
}

# Set up project_id from .env file
project_id = os.getenv("PROJECT_ID")

# Initialize the IBM LLM with updated model
llm = WatsonxLLM(
    model_id="ibm/granite-4-h-small",  # Updated to non-deprecated model
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=project_id,
    params={
        "max_new_tokens": 150
    }
)

## Define Prompt Templates

# Prompt for extracting keywords
keyword_prompt = PromptTemplate(
    input_variables=["text"],
    template="Extract the most important keywords from the following text:\n{text}\n\nKeywords:"
)

# Prompt for generating sentiment summary
sentiment_prompt = PromptTemplate(
    input_variables=["keywords"],
    template="Using the following keywords, summarize the sentiment of the feedback:\nKeywords: {keywords}\n\nSentiment Summary:"
)

# Prompt for refining the summary
refine_prompt = PromptTemplate(
    input_variables=["sentiment_summary"],
    template="Refine the following sentiment summary to make it more concise and precise:\n{sentiment_summary}\n\nRefined Summary:"
)

## Define Chains using LCEL (LangChain Expression Language)

# Helper function to extract text content from LLM response
def extract_text(response):
    """Extract text content from LLM response (handles both AIMessage and string)"""
    if hasattr(response, 'content'):
        return response.content
    return str(response)

# Step 1: Extract keywords from text
def extract_keywords(input_dict):
    """Extract keywords from text"""
    result = (keyword_prompt | llm).invoke({"text": input_dict["text"]})
    keywords = extract_text(result)
    return {"text": input_dict["text"], "keywords": keywords}

# Step 2: Generate sentiment summary from keywords
def generate_sentiment(input_dict):
    """Generate sentiment summary from keywords"""
    result = (sentiment_prompt | llm).invoke({"keywords": input_dict["keywords"]})
    sentiment_summary = extract_text(result)
    return {**input_dict, "sentiment_summary": sentiment_summary}

# Step 3: Refine the sentiment summary
def refine_summary(input_dict):
    """Refine the sentiment summary"""
    result = (refine_prompt | llm).invoke({"sentiment_summary": input_dict["sentiment_summary"]})
    refined_summary = extract_text(result)
    return {"refined_summary": refined_summary}

# Create the complete workflow using LCEL
workflow = (
    RunnableLambda(extract_keywords)
    | RunnableLambda(generate_sentiment)
    | RunnableLambda(refine_summary)
)

# Example Input Text
feedback_text = """
I really enjoy the features of this app, but it crashes frequently, making it hard to use. 
The customer support is helpful, but response times are slow.
I tried to reachout to the support team, but they never responded
For me, the customer support was very much helpful. Ihis is very helpful app. Thank you for grate services. 
"""

# Run the Workflow
result = workflow.invoke({"text": feedback_text})

# Display the Output
print("Refined Sentiment Summary:")
print(result.get("refined_summary", result))  # Extract refined_summary from result dictionary