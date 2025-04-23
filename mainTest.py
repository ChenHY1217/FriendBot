# Main code for running FriendBot IVA Project
import json
import os
from dotenv import load_dotenv
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
from prompts import IntentPrompt, ResponsePrompt, BenchmarkPrompt # Importing the prompts from prompts.py

# Can change temperature to reduce randomness in output from GPT-4o-mini

###################################################################################################
# This is the test code for FriendBot IVA Project. It is responsible for benchmarking             
# the performance of the 3-layer architecutre for emotional intelligence in responses.
# We will compare the performance of the 3-layer architecture with just a generic LLM like GPT-4o.
###################################################################################################

# Function to clean up noisy user input using GPT-4o-mini
def clean_input(noisy_input, openai_client):
    """
    Cleans user-provided text by fixing spelling, grammatical errors, and removing unnecessary noise, 
    while maintaining the original meaning of the message.
    
    Args:
        noisy_input (str): Original user message.
        client (openai.Client): OpenAI client instance.
    
    Returns:
        str: Cleaned and standardized input text.
    """
    system_prompt = """
    You are an assistant that helps clean and understand user input. The input will be from user's who are seeking relationship advice. Your task is to fix spelling errors and remove any unneccesary noise. However, try your best to preserve what the user is trying to say. In other words, do not change the content of the message, just clean it up.

    Return the cleaned input.

    """

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": noisy_input,
            }
        ],
    )

    try:
        response_object = completion.choices[0].message.content
        return response_object
    except json.JSONDecodeError:
        # Fallback if response isn't valid JSON
        return "Error: Unable to parse response."
    
# Function to extract intent from user input using GPT-4o-mini
def extractIntent(input, conversation_history, client):

    system_prompt = f"""
    {IntentPrompt}

    Past conversation history: {conversation_history}
    """
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": input,
            }
        ],
    )

    try:
        response_text = completion.choices[0].message.content
        response_object = json.loads(response_text)
        return response_object
    except json.JSONDecodeError:
        # Fallback if response isn't valid JSON
        return "Error: Unable to parse response."
    
# Function to generate a response using GPT-4o model
def generateResponse(input, emotions, intent, conversation_history, client):

    finalInput = f"User Input: {input}\nEmotions: {json.dumps(emotions)}\nIntent: {intent} \nPast conversation history: {conversation_history}"

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": BenchmarkPrompt},
            {
                "role": "user",
                "content": finalInput,
            }
        ],
    )

    try:
        response_object = completion.choices[0].message.content
        return response_object
    except json.JSONDecodeError:
        # Fallback if response isn't valid JSON
        return "Error: Unable to parse response."


# Improved extractIntent with strict JSON schema enforcement
def extract_intent(input_text, conversation_history, client):
    schema = {
        "relationship_content": "boolean",
        "intent_category": {
            "intent": "one of the eight labels",
            "confidence": "float"
        }
    }
    options = [
        "Seeking advice/guidance",
        "Venting/expressing emotions",
        "Information seeking",
        "Crisis communication",
        "Reflection on past experiences",
        "Future planning/relationship goals",
        "Conflict resolution",
        "Communication improvement"
    ]
    system_prompt = (
        "You are an assistant that categorizes user relationship questions. "
        "Respond with JSON exactly matching this schema: \n" +
        json.dumps(schema, indent=2) +
        "\nWhere 'intent' must be one of: " + ", ".join(options) +
        "\nDo not include any additional keys or text."
    )
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ],
    )
    text = completion.choices[0].message.content
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": text}

# Helper to extract primary intent string from model response
def extract_primary_intent(pred_obj):
    if not isinstance(pred_obj, dict) or "error" in pred_obj:
        return None
    if not pred_obj.get("relationship_content", True):
        return "Information seeking"
    ic = pred_obj.get("intent_category", {})
    intent = ic.get("intent")
    if isinstance(intent, list) and intent:
        intent = intent[0]
    if isinstance(intent, str):
        return intent.strip()
    return None

# Wrapper for normalized intent with fallback
def get_normalized_intent(question, client):
    raw = extract_intent(question, [], client)
    primary = extract_primary_intent(raw)
    intent_map = {
        "seeking advice/guidance": "Seeking advice/guidance",
        "venting/expressing emotions": "Venting/expressing emotions",
        "information seeking": "Information seeking",
        "crisis communication": "Crisis communication",
        "reflection on past experiences": "Reflection on past experiences",
        "future planning/relationship goals": "Future planning/relationship goals",
        "conflict resolution": "Conflict resolution",
        "communication improvement": "Communication improvement"
    }
    if primary and primary.lower() in intent_map:
        return intent_map[primary.lower()]
    # Urgent fallback
    if isinstance(raw, dict):
        emo = raw.get("emotional_undertones", {}).get("emotion", "").lower()
        if any(term in emo for term in ("worry", "anxiety", "grief")):
            return "Crisis communication"
    return "Seeking advice/guidance"

if __name__ == "__main__":

    load_dotenv()  # Load environment variables from .env file
    # print("ENV KEYS:", os.environ.get("OPENAI_API_KEY"))
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    conversation_history = []  # Initialize conversation history

    INTRO = """
    Welcome to FriendBot! I am here to help you with your relationship concerns. I will analyze your input and provide you with insights and advice, or just be a listening ear. Think of me as a friend you can vent to and Let's get started!
    """
    print(INTRO)  # Print the introduction message
    print("")  # TESTING PURPOSES ONLY

    ########################################################################################################
    # 3 Layer Architecture for input processing resulting in improved emotional intelligence in responses
    ########################################################################################################

    # Cleaning user input
    NOISY_INPUT = "I am sooo saaddd!!! I dont kno wat to dooo... My bf is cheatin on meee :("
    cleanedInput = clean_input(NOISY_INPUT, client)

    print("Cleaned Input: ", cleanedInput) # TESTING PURPOSES ONLY
    conversation_history.append(f"User: {cleanedInput}")  # Append cleaned input to conversation history

    ########################################################################################################
    # Layer 1 - Extracting Emotions using Pre-trained BERT model (trained on GoEmotions dataset)
    ########################################################################################################

    # Testing different existing pre-trained models from HuggingFace
    # codewithdark/bert-GoEmotions
    # Load model and tokenizer
    # MODEL_NAME = "codewithdark/bert-Gomotions"
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # # Emotion labels (adjust based on your dataset)
    # emotion_labels = [
    #     "Admiration", "Amusement", "Anger", "Annoyance", "Approval", "Caring", "Confusion",
    #     "Curiosity", "Desire", "Disappointment", "Disapproval", "Disgust", "Embarrassment",
    #     "Excitement", "Fear", "Gratitude", "Grief", "Joy", "Love", "Nervousness", "Optimism",
    #     "Pride", "Realization", "Relief", "Remorse", "Sadness", "Surprise", "Neutral"
    # ]

    # # Example text
    # text = cleanedInput
    # inputs = tokenizer(text, return_tensors="pt")

    # # Predict
    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     probs = torch.sigmoid(outputs.logits).squeeze(0)  # Convert logits to probabilities

    # # Get top 5 predictions
    # top5_indices = torch.argsort(probs, descending=True)[:5]  # Get indices of top 5 labels
    # top5_labels = [emotion_labels[i] for i in top5_indices]
    # top5_probs = [probs[i].item() for i in top5_indices]

    # # Print results
    # print("Top 5 Predicted Emotions:")
    # extractedEmotions = {}
    # for label, prob in zip(top5_labels, top5_probs):
    #     extractedEmotions[label] = prob
    #     print(f"{label}: {prob:.4f}")

    # print("") # TESTING PURPOSES ONLY

    # bhadresh-savani/bert-base-go-emotion
    # Load the model and tokenizer directly for more control over outputs
    model_name = "bhadresh-savani/bert-base-go-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Define the emotion labels for this model
    emotion_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
                     'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
                     'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
                     'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
                     'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    
    # Process the input text
    text = cleanedInput
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()
        
    # Convert logits to probabilities using softmax
    probabilities = torch.nn.functional.softmax(logits, dim=0)
    
    # Get top 5 emotions and their probabilities
    top_k = 5
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    # Create dictionary with top emotions and their scores
    extractedEmotions = {}
    for i in range(top_k):
        emotion = emotion_labels[top_indices[i]]
        score = top_probs[i].item()
        extractedEmotions[emotion] = score
    
    print("Top 5 Predicted Emotions:")
    for emotion, score in extractedEmotions.items():
        print(f"{emotion}: {score:.4f}")
    
    print("") # TESTING PURPOSES ONLY

    ########################################################################################################
    # Layer 2 - Extracting Intent using OpenAI 4o-mini model
    ########################################################################################################

    extractedIntent = extractIntent(cleanedInput, conversation_history, client)

    print("Intent Response: ", extractedIntent) # TESTING PURPOSES ONLY
    print("type of response of intent prompt: ", type(extractedIntent)) # TESTING PURPOSES ONLY
    print("") # TESTING PURPOSES ONLY

    ########################################################################################################
    # Layer 3 - Generating Response using OpenAI 4o model
    ########################################################################################################

    response = generateResponse(cleanedInput, extractedEmotions, extractedIntent, conversation_history, client)

    # print("Response: ", response) # TESTING PURPOSES ONLY
    # print("type of response: ", type(response)) # TESTING PURPOSES ONLY
    # print("") # TESTING PURPOSES ONLY

    ################################################################################
    # Benchmarking / Evaluating the performance of the 3-layer architecture
    ################################################################################

    # For benchmarking, we can compare the performance of the 3-layer architecture with just a generic LLM like GPT-4o.

    baseResponseObject = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": BenchmarkPrompt},
            {
                "role": "user",
                "content": cleanedInput,
            }
        ],
    )

    baseResponse = baseResponseObject.choices[0].message.content
    print("Base Response: ", baseResponse) # TESTING PURPOSES ONLY

    print("") # TESTING PURPOSES ONLY

    modelResponse = response
    print("Model Response: ", modelResponse) # TESTING PURPOSES ONLY

    ################################################################################
    # testing the performance of the final responses
    #################################################################################
    from evaluation import evaluate_emotional_intelligence
    
    print("\nEvaluating emotional intelligence of responses...")
    evaluation = evaluate_emotional_intelligence(cleanedInput, modelResponse, baseResponse, client)
    
    print("\n=== EMOTIONAL INTELLIGENCE EVALUATION ===")
    print(json.dumps(evaluation, indent=2))
    
    print("Model Response (3-Layer) Total Score:", evaluation["model_response"]["total"])
    print("Base Response (Base) Total Score:", evaluation["base_response"]["total"])
    print("Winner:", evaluation["winner"])


    # Load model
    model_name = "bhadresh-savani/bert-base-go-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Define labels and mapping
    ################################################################################
    # testing the performance of the emotion layer
    ################################################################################
    # Emotion labels from your BERT model
    
    emotion_labels = [
        "Admiration", "Amusement", "Anger", "Annoyance", "Approval", "Caring", "Confusion",
        "Curiosity", "Desire", "Disappointment", "Disapproval", "Disgust", "Embarrassment",
        "Excitement", "Fear", "Gratitude", "Grief", "Joy", "Love", "Nervousness", "Optimism",
        "Pride", "Realization", "Relief", "Remorse", "Sadness", "Surprise", "Neutral"
    ]
    emotion_map = {
        "Admiration": "Joy", "Amusement": "Joy", "Anger": "Anger", "Annoyance": "Anger",
        "Approval": "Trust", "Caring": "Trust", "Confusion": "Fear", "Curiosity": "Anticipation",
        "Desire": "Anticipation", "Disappointment": "Sadness", "Disapproval": "Anger",
        "Disgust": "Disgust", "Embarrassment": "Shame", "Excitement": "Joy", "Fear": "Fear",
        "Gratitude": "Trust", "Grief": "Sadness", "Joy": "Joy", "Love": "Joy",
        "Nervousness": "Fear", "Optimism": "Anticipation", "Pride": "Joy", "Realization": "Surprise",
        "Relief": "Joy", "Remorse": "Sadness", "Sadness": "Sadness", "Surprise": "Surprise",
        "Neutral": "Neutral"
    }

    df = pd.read_csv("data/dating_emotion_dataset.csv", encoding="latin-1")
    full_correct = half_correct = wrong = total = 0

    for _, row in df.iterrows():
        text = row["Question"]
        raw = [e.strip() for e in row["Emotions"].split(",")]
        true_emotions = [e.split("(")[0].strip() for e in raw]

        # Model inference
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze()
            probs = torch.sigmoid(logits)

        # Top-3 predictions
        top3 = torch.argsort(probs, descending=True)[:3]
        preds = [emotion_map[emotion_labels[i]] for i in top3]

        # Count matches
        matches = sum(1 for p in preds if p in true_emotions)

        if matches >= len(true_emotions):
            full_correct += 1
        elif matches > 0:
            half_correct += 1
        else:
            wrong += 1
        total += 1

    # Report
    print(f"Emotion Evaluation (Layer 1):")
    print(f"  Full correct (all true emotions in top-3): {full_correct}/{total}")
    print(f"  Half correct (one true emotion in top-3): {half_correct}/{total}")
    print(f"  Wrong (no true emotions in top-3): {wrong}/{total}\n")



    #intent
    load_dotenv()
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    df_intent = pd.read_csv("data/dating_intent_dataset.csv", encoding="latin-1")

    correct_raw, correct_norm, total = 0, 0, 0

    for _, row in df_intent.iterrows():
        question = row["Question"]
        true_intent = row["Intent"].strip()

        # Original prediction
        raw_obj = extractIntent(question, [], client)
        raw_intent = extract_primary_intent(raw_obj) or ""
        if raw_intent.lower() == true_intent.lower():
            correct_raw += 1

        # Our normalized prediction
        norm_intent = get_normalized_intent(question, client)
        if norm_intent.lower() == true_intent.lower():
            correct_norm += 1

        total += 1

    raw_accuracy = correct_raw / total if total else 0.0
    norm_accuracy = correct_norm / total if total else 0.0

    print(f"Mapped Intent Classification Accuracy: {norm_accuracy:.2%} ({correct_norm}/{total})")
    print(f"Total examples evaluated: {total}")
