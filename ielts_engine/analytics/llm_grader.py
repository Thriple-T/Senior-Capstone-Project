import os
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

# Used Hugging Face Serverless Inference API
base_url = "https://router.huggingface.co/v1"
# Ensure any user has HF_TOKEN set in their environment variables to avoid hardcoding sensitive information
api_key = os.environ.get("HF_TOKEN", "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")

if not api_key:
    print("[WARNING] HF_TOKEN environment variable is missing. LLM grading will fail.")

# Initialize the Instructor client with Qwen3-VL-8B-Instruct via the openai wrapper
try:
    client = instructor.from_openai(OpenAI(
        base_url=base_url,
        api_key=api_key,
    ))
except AttributeError:
    # Fallback to older instructor syntax if installed
    client = instructor.patch(OpenAI(
        base_url=base_url,
        api_key=api_key,
    ))

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct:novita"

class IELTSAssessment(BaseModel):
    Task_Achievement: float = Field(..., description="Band score for Task Achievement (0-9)")
    Coherence_Cohesion: float = Field(..., description="Band score for Coherence and Cohesion (0-9)")
    Lexical_Resource: float = Field(..., description="Band score for Lexical Resource (0-9)")
    Grammar_Range: float = Field(..., description="Band score for Grammatical Range and Accuracy (0-9)")
    total_band: float = Field(..., description="Overall Band Score for the essay (average of the 4 criteria, rounded to nearest 0.5)")
    feedback_summary: str = Field(..., description="A 2-3 sentence summary of feedback explaining the scores.")

def get_llm_assessment(prompt_text: str, essay_text: str) -> IELTSAssessment:
    system_prompt = (
        "You are an expert IELTS Writing examiner. You grade essays based on four criteria: "
        "Task Achievement, Coherence & Cohesion, Lexical Resource, and Grammatical Range & Accuracy. "
        "You output your assessment in a strict structured format. Provide band scores between 0.0 and 9.0."
    )
    
    user_prompt = f"Prompt: {prompt_text}\n\nEssay: {essay_text}\n\nPlease evaluate this essay and provide scores and feedback."
    
    assessment = client.chat.completions.create(
        model=MODEL_NAME,
        response_model=IELTSAssessment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1
    )
    
    return assessment

if __name__ == "__main__":
    sample_prompt = "Some people believe that university education should be free for everyone. To what extent do you agree or disagree?"
    sample_essay = "The debate regarding the accessibility of higher education is a highly contentious one."
    print("Testing LLM Grader...")
    try:
        res = get_llm_assessment(sample_prompt, sample_essay)
        print(res.model_dump_json(indent=2))
    except Exception as e:
        print(f"Failed to connect to LLM or parse output: {e}\nCheck your base_url or model name.")
