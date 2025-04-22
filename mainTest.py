# Main code for running FriendBot IVA Project
import json
import os
from dotenv import load_dotenv
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
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
def extractIntent(input, conversation_history, client):

    system_prompt = f"""
    {IntentPrompt}

    Past conversation history: {conversation_history}
    """
    
    completion = client.chat.completions.create(
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
def generateResponse(input, emotions, intent, conversation_history, client):

    finalInput = f"User Input: {input}\nEmotions: {json.dumps(emotions)}\nIntent: {intent} \nPast conversation history: {conversation_history}"

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

    extractedIntent = extractIntent(cleanedInput, conversation_history, client)

    print("Intent Response: ", extractedIntent) # TESTING PURPOSES ONLY
    print("type of response of intent prompt: ", type(extractedIntent)) # TESTING PURPOSES ONLY
    print("") # TESTING PURPOSES ONLY

    ########################################################################################################
    # Layer 3 - Generating Response using OpenAI 4o model
    ########################################################################################################

    response = generateResponse(cleanedInput, extractedEmotions, extractedIntent, conversation_history, client)

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
    # Evaluate the emotional intelligence of responses
    ################################################################################
    from evaluation import evaluate_emotional_intelligence
    
    print("\nEvaluating emotional intelligence of responses...")
    evaluation = evaluate_emotional_intelligence(cleanedInput, modelResponse, baseResponse, client)
    
    print("\n=== EMOTIONAL INTELLIGENCE EVALUATION ===")
    print(json.dumps(evaluation, indent=2))
    
    print("\nResponse A (3-Layer) Total Score:", evaluation["response_a"]["total"])
    print("Response B (Base) Total Score:", evaluation["response_b"]["total"])
    print("Winner:", evaluation["winner"])




