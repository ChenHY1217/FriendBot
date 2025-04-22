# Main code for running FriendBot IVA Project
import json
import os
from dotenv import load_dotenv
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from prompts import IntentPrompt, ResponsePrompt # Importing the prompts from prompts.py

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
def extract_intent(user_input, openai_client):
    """
    Extracts the intent, emotional undertone, and relationship dynamics from the user's message 
    using GPT-based intent classification.
    
    Args:
        user_input (str): Cleaned input from the user.
        conversation_history (list): List of previous conversation turns for context.
        client (openai.Client): OpenAI client instance.
    
    Returns:
        dict: Structured JSON containing intent, emotions, dynamics, urgency level, and therapeutic approaches.
    """
    
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": IntentPrompt},
            {
                "role": "user",
                "content": user_input,
            }
        ],
    )

    try:
        response_object = json.loads(completion.choices[0].message.content)
        return response_object
    except json.JSONDecodeError:
        # Fallback if response isn't valid JSON
        return "Error: Unable to parse response."
    
# Function to generate a response using GPT-4o model
def generate_response(user_input, emotions, intent, openai_client):
    """
    Generates an emotionally intelligent and contextually relevant response for the user's message 
    based on extracted emotions and intent.
    
    Args:
        user_input (str): User's cleaned input.
        emotions (dict): Extracted emotions from the user's input.
        intent (dict): Extracted intent details.
        client (openai.Client): OpenAI client instance.
    
    Returns:
        str: Generated FriendBot response.
    """
    final_input = f"User Input: {user_input}\nEmotions: {json.dumps(emotions)}\nIntent: {intent}"

    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": ResponsePrompt},
            {
                "role": "user",
                "content": final_input,
            }
        ],
    )

    try:
        response_object = completion.choices[0].message.content
        return response_object
    except json.JSONDecodeError:
        # Fallback if response isn't valid JSON
        return "Error: Unable to parse response."

if __name__ == "__main__":

    load_dotenv()  # Load environment variables from .env file
    print("ENV KEYS:", os.environ.get("OPENAI_API_KEY"))
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

    ########################################################################################################
    # Layer 1 - Extracting Emotions using Pre-trained BERT model (trained on GoEmotions dataset)
    ########################################################################################################

    # Testing different existing pre-trained models from HuggingFace
    # codewithdark/bert-GoEmotions
    # Load model and tokenizer
    MODEL_NAME = "codewithdark/bert-Gomotions"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Emotion labels (adjust based on your dataset)
    emotion_labels = [
        "Admiration", "Amusement", "Anger", "Annoyance", "Approval", "Caring", "Confusion",
        "Curiosity", "Desire", "Disappointment", "Disapproval", "Disgust", "Embarrassment",
        "Excitement", "Fear", "Gratitude", "Grief", "Joy", "Love", "Nervousness", "Optimism",
        "Pride", "Realization", "Relief", "Remorse", "Sadness", "Surprise", "Neutral"
    ]

    # Example text
    text = cleanedInput
    inputs = tokenizer(text, return_tensors="pt")

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze(0)  # Convert logits to probabilities

    # Get top 5 predictions
    top5_indices = torch.argsort(probs, descending=True)[:5]  # Get indices of top 5 labels
    top5_labels = [emotion_labels[i] for i in top5_indices]
    top5_probs = [probs[i].item() for i in top5_indices]

    # Print results
    print("Top 5 Predicted Emotions:")
    extractedEmotions = {}
    for label, prob in zip(top5_labels, top5_probs):
        extractedEmotions[label] = prob
        print(f"{label}: {prob:.4f}")

    print("") # TESTING PURPOSES ONLY

    ########################################################################################################
    # Layer 2 - Extracting Intent using OpenAI 4o-mini model
    ########################################################################################################

    extractedIntent = extract_intent(cleanedInput, client)

    print("Intent Response: ", extractedIntent) # TESTING PURPOSES ONLY
    print("type of response of intent prompt: ", type(extractedIntent)) # TESTING PURPOSES ONLY
    print("") # TESTING PURPOSES ONLY

    ########################################################################################################
    # Layer 3 - Generating Response using OpenAI 4o model
    ########################################################################################################

    response = generate_response(cleanedInput, extractedEmotions, extractedIntent, client)

    print("Response: ", response) # TESTING PURPOSES ONLY
    print("type of response: ", type(response)) # TESTING PURPOSES ONLY
    print("") # TESTING PURPOSES ONLY


    ################################################################################
    # Benchmarking / Evaluating the performance of the 3-layer architecture
    ################################################################################

    # For benchmarking, we can compare the performance of the 3-layer architecture with just a generic LLM like GPT-4o.

    baseResponseObject = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a friend that can give relationship advice."},
            {
                "role": "user",
                "content": cleanedInput,
            }
        ],
    )

    baseResponse = baseResponseObject.choices[0].message.content
    print("Base Response: ", baseResponse) # TESTING PURPOSES ONLY

    modelResponse = response
    print("Model Response: ", modelResponse) # TESTING PURPOSES ONLY

    ################################################################################
    # testing the performance of the intent layer
    ################################################################################

    # Emotion labels from your BERT model
    emotion_labels = [
        "Admiration", "Amusement", "Anger", "Annoyance", "Approval", "Caring", "Confusion",
        "Curiosity", "Desire", "Disappointment", "Disapproval", "Disgust", "Embarrassment",
        "Excitement", "Fear", "Gratitude", "Grief", "Joy", "Love", "Nervousness", "Optimism",
        "Pride", "Realization", "Relief", "Remorse", "Sadness", "Surprise", "Neutral"
    ]

    # Remapping: model_label → simplified label (based on dataset clusters)
    emotion_map = {
        "Admiration": "Joy",
        "Amusement": "Joy",
        "Anger": "Anger",
        "Annoyance": "Anger",
        "Approval": "Trust",
        "Caring": "Trust",
        "Confusion": "Fear",
        "Curiosity": "Anticipation",
        "Desire": "Anticipation",
        "Disappointment": "Sadness",
        "Disapproval": "Anger",
        "Disgust": "Disgust",
        "Embarrassment": "Shame",
        "Excitement": "Joy",
        "Fear": "Fear",
        "Gratitude": "Trust",
        "Grief": "Sadness",
        "Joy": "Joy",
        "Love": "Joy",
        "Nervousness": "Fear",
        "Optimism": "Anticipation",
        "Pride": "Joy",
        "Realization": "Surprise",
        "Relief": "Joy",
        "Remorse": "Sadness",
        "Sadness": "Sadness",
        "Surprise": "Surprise",
        "Neutral": "Neutral"
    }

    # Debug CWD and file visibility
    print("CWD:", os.getcwd())
    print("FILES:", os.listdir())

    # Load dataset
    df = pd.read_csv("data/dating_emotion_dataset.csv", encoding="latin-1")

    correct = 0
    total = 0

    for _, row in df.iterrows():
        text = row["Question"]
        true_emotions_raw = [e.strip() for e in row["Emotions"].split(",")]

        # Map true emotions into simplified labels
        true_emotions = []
        for emo in true_emotions_raw:
            if "(" in emo:  # e.g., "Sadness (worry)"
                emo = emo.split("(")[0].strip()
            true_emotions.append(emo)

        # Tokenize and run through model
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).squeeze(0)

        # Get top-3 predictions and map them
        top3_indices = torch.argsort(probs, descending=True)[:3]
        top3_raw = [emotion_labels[i] for i in top3_indices]
        top3_remapped = [emotion_map[emo] for emo in top3_raw]

        # Check if any mapped prediction is in the mapped true set
        if any(pred in true_emotions for pred in top3_remapped):
            correct += 1
        total += 1

    # Compute accuracy
    accuracy = correct / total
    print(f"\nLayer 1 Emotion Classification Accuracy (remapped, top-3 in true set): "
        f"{accuracy:.2%} ({correct}/{total} correct)")
