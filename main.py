# Main code for running FriendBot IVA Project
from dotenv import load_dotenv
import os
import openai
import json
from prompts import IntentPrompt, ResponsePrompt # Importing the prompts from prompts.py

# Can change temperature to reduce randomness in output from GPT-4o-mini



# Function to clean up noisy user input using GPT-4o-mini
def cleanInput(noisyInput, client):
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
def extractIntent(input, client):
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": IntentPrompt},
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
def generateResponse(input, emotions, intent, client):

    finalInput = f"User Input: {input}\nEmotions: {emotions}\nIntent: {intent}"

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

    # 3 Layer Architecture for input processing resulting in improved emotional intelligence in responses

    # Cleaning user input
    noisyInput = "I am sooo saaddd!!! I dont kno wat to dooo... My bf is cheatin on meee :("
    cleanedInput = cleanInput(noisyInput, client)

    print("Cleaned Input: ", cleanedInput) # TESTING PURPOSES ONLY
    print("") # TESTING PURPOSES ONLY

    # Layer 1 - Extracting Emotions using Pre-trained BERT model (trained on GoEmotions dataset)





    # Layer 2 - Extracting Intent using OpenAI 4o-mini model
    extractedIntent = extractIntent(cleanedInput, client)

    print("Intent Response: ", extractedIntent) # TESTING PURPOSES ONLY
    print("type fo response of intent prompt: ", type(extractedIntent)) # TESTING PURPOSES ONLY
    print("") # TESTING PURPOSES ONLY

    # Layer 3 - Generating Response using OpenAI 4o model
    response = generateResponse(cleanedInput, extractedEmotions, extractedIntent, client)

    print("Response: ", response) # TESTING PURPOSES ONLY
    print("type of response: ", type(response)) # TESTING PURPOSES ONLY
    print("") # TESTING PURPOSES ONLY




