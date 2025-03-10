import openai
import os
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Database Setup
DATABASE_URL = "postgresql://your_user:your_password@localhost/your_database"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define Request Model
class MessageRequest(BaseModel):
    lead_name: str
    company_name: str
    job_title: str
    pain_point: str
    tone: str = "professional"
    model: str = "gpt-4"
    response_length: str = "medium"

# Tone and Length Dictionaries
tone_styles = {
    "casual": "Keep it light and friendly. Use conversational language and a warm tone.",
    "professional": "Keep it structured, concise, and informative. Maintain a formal tone.",
    "persuasive": "Use urgency and strong CTA. Highlight FOMO (fear of missing out)."
}

length_styles = {
    "short": "Keep the message under 80 words. Make it concise and to the point.",
    "medium": "Balance between informative and hook-based for engagement.",
    "long": "Write a more detailed message around 150-200 words for high-value prospects."
}

# Function to generate AI sales messages
def generate_sales_message(request: MessageRequest):
    prompt = f"""
    You are an AI sales assistant crafting outbound messages for B2B lead generation.
    Your goal is to generate a {request.tone} message that is relevant and engaging.
    
    Writing Style: {tone_styles.get(request.tone, 'professional')}
    Response Length: {length_styles.get(request.response_length, 'medium')}
    
    Lead Info:
    - Name: {request.lead_name}
    - Company: {request.company_name}
    - Job Title: {request.job_title}
    - Pain Point: {request.pain_point}
    
    Generate a message within the specified length.
    """
    
    response = openai.ChatCompletion.create(
        model=request.model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response["choices"][0]["message"]["content"]

# FastAPI App Initialization
app = FastAPI()

@app.post("/generate-sales-message/")
def api_generate_sales_message(request: MessageRequest):
    """ API endpoint for generating AI-powered sales messages. """
    return {"message": generate_sales_message(request)}

# Define A/B Test Model
class ABTestResult(Base):
    __tablename__ = "ab_test_results"
    id = Column(Integer, primary_key=True, index=True)
    lead_name = Column(String(100))
    company_name = Column(String(100))
    job_title = Column(String(100))
    pain_point = Column(Text)
    variation_A = Column(Text)
    variation_B = Column(Text)
    selected_version = Column(String(1), default="")
    response_metric = Column(Integer, default=0)  # Clicks, replies, etc.

Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# A/B Testing Endpoint
@app.post("/ab-test-sales-messages/")
def ab_test_sales_messages(request: MessageRequest, db: Session = Depends(get_db)):
    """ Generate two variations of AI-powered sales messages for A/B testing and store in DB. """
    variation_A = generate_sales_message(request)
    variation_B = generate_sales_message(request)
    
    db_record = ABTestResult(
        lead_name=request.lead_name,
        company_name=request.company_name,
        job_title=request.job_title,
        pain_point=request.pain_point,
        variation_A=variation_A,
        variation_B=variation_B,
        response_metric=0
    )
    
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
    return {"variation_A": variation_A, "variation_B": variation_B, "test_id": db_record.id}

# API to update A/B Test Result
class ABTestUpdateRequest(BaseModel):
    test_id: int
    selected_version: str  # 'A' or 'B'
    response_metric: int  # Engagement score

@app.put("/update-ab-test-result/")
def update_ab_test_result(request: ABTestUpdateRequest, db: Session = Depends(get_db)):
    """ Automatically adjust AI messaging based on A/B test results. """
    test_entry = db.query(ABTestResult).filter(ABTestResult.id == request.test_id).first()
    
    if not test_entry:
        return {"error": "Test ID not found"}
    
    test_entry.selected_version = request.selected_version
    test_entry.response_metric = request.response_metric  # Update engagement score
    db.commit()
    db.refresh(test_entry)
    
    # Use AI to refine the next message based on the winning version
    adjusted_prompt = f"""
    A/B Test Performance Analysis:
    - Variation A: {test_entry.variation_A}
    - Variation B: {test_entry.variation_B}
    - Winning Message: {test_entry.selected_version} with {test_entry.response_metric} engagement.
    
    Learn from the winning message and generate an improved version based on its structure.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": adjusted_prompt}]
    )
    
    improved_message = response["choices"][0]["message"]["content"]
    
    return {
        "message": "A/B test result updated successfully",
        "updated_record": test_entry,
        "improved_message": improved_message
    }
