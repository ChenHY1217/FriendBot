# Main code for running FriendBot IVA Project
import json
import os
from dotenv import load_dotenv
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from prompts import IntentPrompt, ResponsePrompt # Importing the prompts from prompts.py

# Can change temperature to reduce randomness in output from GPT-4o-mini



# Function to clean up noisy user input using GPT-4o-mini
def clean_input(noisyInput, client):
    # Prompt designed to get GPT to clean and understand user input
    system_prompt = """
    You are an assistant that helps clean and understand user input. The input will be from user's who are seeking relationship advice. Your task is to fix spelling errors and remove any unneccesary noise. However, try your best to preserve what the user is trying to say. In other words, do not change the content of the message, just clean it up.

    Return the cleaned input.

    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": noisyInput,
            }
        ],
    )

    try:
        responseObject = completion.choices[0].message.content
        return responseObject
    except json.JSONDecodeError:
        # Fallback if response isn't valid JSON
        return "Error: Unable to parse response."
    
# Function to extract intent from user input using GPT-4o-mini
def extract_intent(input, conversation_history, client):

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
        responseObject = json.loads(completion.choices[0].message.content)
        return responseObject
    except json.JSONDecodeError:
        # Fallback if response isn't valid JSON
        return "Error: Unable to parse response."
    
# Function to generate a response using GPT-4o model
def generate_response(input, emotions, intent, conversation_history, client):

    finalInput = f"User Input: {input}\nEmotions: {json.dumps(emotions)}\nIntent: {intent} \nPast conversation history: {conversation_history}"

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": ResponsePrompt},
            {
                "role": "user",
                "content": finalInput,
            }
        ],
    )

    try:
        responseObject = completion.choices[0].message.content
        return responseObject
    except json.JSONDecodeError:
        # Fallback if response isn't valid JSON
        return "Error: Unable to parse response."


if __name__ == "__main__":

    load_dotenv()  # Load environment variables from .env file
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    conversation_history = []  # Initialize conversation history
    conversation_history_base = []  # Initialize conversation history for base response

    intro = """
    Welcome to FriendBot! I am here to help you with your relationship concerns. I will analyze your input and provide you with insights and advice, or just be a listening ear. Think of me as a friend you can vent to and Let's get started!
    """
    print(intro)  # Print the introduction message
    print("")  # TESTING PURPOSES ONLY

    ########################################################################################################
    # 3 Layer Architecture for input processing resulting in improved emotional intelligence in responses
    ########################################################################################################

    # Cleaning user input
    # sample noisy input ==> "I am sooo saaddd!!! I dont kno wat to dooo... My bf is cheatin on meee :("
    while True:

        noisyInput = input("User: ")  # Get user input
        cleanedInput = clean_input(noisyInput, client)

        print("Cleaned Input: ", cleanedInput) # TESTING PURPOSES ONLY
        conversation_history.append(f"User: {cleanedInput}")  # Append cleaned input to conversation history
        conversation_history_base.append(f"User: {cleanedInput}")  # Append cleaned input to base conversation history

        ########################################################################################################
        # Layer 1 - Extracting Emotions using Pre-trained BERT model (trained on GoEmotions dataset)
        ########################################################################################################

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

        extractedIntent = extract_intent(cleanedInput, conversation_history, client)

        print("Intent Response: ", extractedIntent) # TESTING PURPOSES ONLY
        print("type of response of intent prompt: ", type(extractedIntent)) # TESTING PURPOSES ONLY
        print("") # TESTING PURPOSES ONLY

        ########################################################################################################
        # Layer 3 - Generating Response using OpenAI 4o model
        ########################################################################################################

        response = generate_response(cleanedInput, extractedEmotions, extractedIntent, conversation_history, client)
        baseResponse = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a friend that is experienced in dating. The Question is coming from someone seeking advice. Give a response to each question."},
                {
                    "role": "user",
                    "content": cleanedInput,
                }
            ],
        ).choices[0].message.content

        print("Response: ", response) # TESTING PURPOSES ONLY
        print("type of response: ", type(response)) # TESTING PURPOSES ONLY
        print("") # TESTING PURPOSES ONLY

        conversation_history.append(f"FriendBot: {response}")  # Append response to conversation history
        conversation_history_base.append(f"GPT 4o: {baseResponse}")

        print("----------------------------------------------------------------")  # TESTING PURPOSES ONLY
        print("GPT 4o: ", baseResponse)  # Print the response from the model
        print("")
        print("FriendBot: ", response)  # Print the response from the model


        