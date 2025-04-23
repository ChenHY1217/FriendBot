import json

def evaluate_emotional_intelligence(user_input, model_response, base_response, client):
    """
    Evaluates the emotional intelligence of two different responses (3-layer architecture vs base model)
    using GPT-4o as the evaluator.
    
    Args:
        user_input: The original user input
        model_response: Response from the 3-layer architecture
        base_response: Response from base GPT-4o
        client: OpenAI client instance
    
    Returns:
        dict: Evaluation scores and feedback
    """
    
    evaluation_prompt = """
    You are an expert evaluator of emotional intelligence in AI responses to relationship advice scenarios. 
    Analyze the following user input and two different AI responses, then score them based on the rubric below.

    Note the response is supposed to be from the standpoint of a relationship expert friend.
    
    USER INPUT: {user_input}
    
    MODEL RESPONSE (3-LAYER ARCHITECTURE): {model_response}
    
    BASE RESPONSE (BASE MODEL): {base_response}
    
    EVALUATION RUBRIC (score each category from 1-10):
    
    1. Emotional Recognition (1-10):
       - How accurately does the response identify and acknowledge the user's emotional state?
       - Does it recognize both explicit and implicit emotions?
       - Does it detect emotional nuances and subtleties?
    
    2. Empathetic Response (1-10):
       - How well does the response demonstrate genuine understanding of the user's feelings?
       - Does it validate emotions without judgment?
       - Does it show appropriate emotional resonance?
    
    3. Relationship Insight (1-10):
       - How well does the response demonstrate understanding of relationship dynamics?
       - Does it identify underlying patterns or issues?
       - Does it offer perspective that shows relationship expertise?
    
    4. Balance of Support vs. Advice (1-10):
       - How well does the response balance emotional support with practical guidance?
       - Does it prioritize emotional validation before offering solutions?
       - Does it offer advice in a non-prescriptive, thoughtful manner?
    
    5. Communication Style (1-10):
       - How natural, warm and friend-like is the response?
       - Does it avoid clinical or overly formal language?
       - Does it match the emotional tone and energy of the user?
    
    Please provide:
    1. Numerical scores for each category for both responses
    2. A brief explanation (2-3 sentences) for each score
    3. An overall winner with justification
    4. Specific suggestions for improving the lower-scoring response
    
    Format your response as a JSON object with the following structure:
    {{
      "model_response": {{
        "emotional_recognition": {{"score": X, "explanation": "..."}},
        "empathetic_response": {{"score": X, "explanation": "..."}},
        "relationship_insight": {{"score": X, "explanation": "..."}},
        "balance": {{"score": X, "explanation": "..."}},
        "communication_style": {{"score": X, "explanation": "..."}},
        "total": X
      }},
      "base_response": {{
        "emotional_recognition": {{"score": X, "explanation": "..."}},
        "empathetic_response": {{"score": X, "explanation": "..."}},
        "relationship_insight": {{"score": X, "explanation": "..."}},
        "balance": {{"score": X, "explanation": "..."}},
        "communication_style": {{"score": X, "explanation": "..."}},
        "total": X
      }},
      "winner": "model_response or base_response",
      "justification": "...",
      "improvement_suggestions": "..."
    }}
    """
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert evaluator of emotional intelligence in AI responses."},
                {
                    "role": "user",
                    "content": evaluation_prompt.format(
                        user_input=user_input,
                        model_response=model_response,
                        base_response=base_response
                    )
                }
            ],
            response_format={"type": "json_object"}
        )
        
        evaluation = json.loads(completion.choices[0].message.content)
        return evaluation
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return {"error": f"Evaluation failed: {str(e)}"}
