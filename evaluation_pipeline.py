import json
import os
import csv
import pandas as pd
from dotenv import load_dotenv
import openai
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from evaluation import evaluate_emotional_intelligence
from prompts import IntentPrompt, ResponsePrompt

# Load environment variables
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Functions from mainTest.py
def clean_input(noisy_input, openai_client):
    """Cleans user-provided text"""
    system_prompt = """
    You are an assistant that helps clean and understand user input. The input will be from user's who are seeking relationship advice. Your task is to fix spelling errors and remove any unneccesary noise. However, try your best to preserve what the user is trying to say. In other words, do not change the content of the message, just clean it up.

    Return the cleaned input.
    """

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": noisy_input}
        ],
    )

    try:
        return completion.choices[0].message.content
    except Exception:
        return "Error: Unable to parse response."

def extractIntent(input_text, client):
    """Extracts intent from user input"""
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": IntentPrompt},
            {"role": "user", "content": input_text}
        ],
    )

    try:
        return json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError:
        return "Error: Unable to parse response."

def generateResponse(input_text, emotions, intent, conversation_history, client):
    """Generates a response using 3-layer architecture"""
    final_input = f"User Input: {input_text}\nEmotions: {json.dumps(emotions)}\nIntent: {intent} \nPast conversation history: {conversation_history}"

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": ResponsePrompt},
            {"role": "user", "content": final_input}
        ],
    )

    try:
        return completion.choices[0].message.content
    except Exception:
        return "Error: Unable to parse response."

def extract_emotions_with_bert(text, model, tokenizer, emotion_labels):
    """Extract emotions using pre-trained BERT model"""
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze(0)

    top5_indices = torch.argsort(probs, descending=True)[:5]
    top5_labels = [emotion_labels[i] for i in top5_indices]
    top5_probs = [probs[i].item() for i in top5_indices]

    extracted_emotions = {}
    for label, prob in zip(top5_labels, top5_probs):
        extracted_emotions[label] = prob
    
    return extracted_emotions

def get_base_response(input_text, client):
    """Get baseline response from GPT-4o without the 3-layer architecture"""
    response_object = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a friend that is experienced in dating. The Question is coming from someone seeking advice. Give a response to each question."},
            {"role": "user", "content": input_text}
        ],
    )
    return response_object.choices[0].message.content

def process_question(question, model, tokenizer, emotion_labels, client):
    """Process a single question through both pipelines and evaluate"""
    # Clean input if needed (assuming questions are already clean)
    cleaned_input = question
    conversation_history = []

    # 3-Layer Architecture process
    # 1. Extract emotions using BERT
    extracted_emotions = extract_emotions_with_bert(cleaned_input, model, tokenizer, emotion_labels)
    
    # 2. Extract intent using GPT
    extracted_intent = extractIntent(cleaned_input, conversation_history, client)
    
    # 3. Generate response using our architecture
    model_response = generateResponse(cleaned_input, extracted_emotions, extracted_intent, conversation_history, client)
    
    # Get baseline response
    base_response = get_base_response(cleaned_input, client)
    
    # Evaluate emotional intelligence
    evaluation = evaluate_emotional_intelligence(cleaned_input, model_response, base_response, client)
    
    # Determine winner
    winner_response = model_response if evaluation["winner"] == "Response A (3-layer architecture)" else base_response
    winner_name = "3-layer" if evaluation["winner"] == "Response A (3-layer architecture)" else "baseline"
    winner_score = evaluation["response_a"]["total"] if evaluation["winner"] == "Response A (3-layer architecture)" else evaluation["response_b"]["total"]
    
    return {
        "question": question,
        "winner": winner_name,
        "winning_response": winner_response,
        "winning_score": winner_score,
        "3layer_score": evaluation["response_a"]["total"],
        "baseline_score": evaluation["response_b"]["total"]
    }

def main():
    # Load BERT model for emotion detection
    MODEL_NAME = "bhadresh-savani/bert-base-go-emotion"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    emotion_labels = [
        "Admiration", "Amusement", "Anger", "Annoyance", "Approval", "Caring", "Confusion",
        "Curiosity", "Desire", "Disappointment", "Disapproval", "Disgust", "Embarrassment",
        "Excitement", "Fear", "Gratitude", "Grief", "Joy", "Love", "Nervousness", "Optimism",
        "Pride", "Realization", "Relief", "Remorse", "Sadness", "Surprise", "Neutral"
    ]

    # Read questions from CSV
    df = pd.read_csv('data/detected_emotions_output (1).csv')
    questions = df['Question'].tolist()
    
    results = []
    three_layer_wins = 0
    baseline_wins = 0
    ties = 0
    
    # Process each question
    for i, question in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
        result = process_question(question, model, tokenizer, emotion_labels, client)
        results.append(result)
        
        # Count wins
        if result["winner"] == "3-layer":
            three_layer_wins += 1
        elif result["winner"] == "baseline":
            baseline_wins += 1
        else:
            ties += 1
        
        # Print progress stats
        print(f"Current score: 3-layer: {three_layer_wins}, Baseline: {baseline_wins}, Ties: {ties}")
    
    # Write results to CSV
    with open('evaluation_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['question', 'winner', 'winning_response', 'winning_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'question': result['question'],
                'winner': result['winner'],
                'winning_response': result['winning_response'],
                'winning_score': result['winning_score']
            })
    
    # Write detailed results with scores
    with open('evaluation_detailed_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['question', 'winner', '3layer_score', 'baseline_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'question': result['question'],
                'winner': result['winner'],
                '3layer_score': result['3layer_score'],
                'baseline_score': result['baseline_score']
            })
    
    # Print final stats
    total_questions = len(questions)
    print("\n=== FINAL EVALUATION RESULTS ===")
    print(f"Total questions processed: {total_questions}")
    print(f"3-Layer Architecture wins: {three_layer_wins} ({(three_layer_wins/total_questions)*100:.2f}%)")
    print(f"Baseline GPT-4o wins: {baseline_wins} ({(baseline_wins/total_questions)*100:.2f}%)")
    print(f"Ties: {ties} ({(ties/total_questions)*100:.2f}%)")
    print(f"Results saved to evaluation_results.csv and evaluation_detailed_results.csv")

if __name__ == "__main__":
    main()
