# Main code for running FriendBot IVA Project
from dotenv import load_dotenv
import os
import openai
import json

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

if __name__ == "__main__":

    load_dotenv()  # Load environment variables from .env file
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 3 Layer Architecture for input processing resulting in improved emotional intelligence in responses

    # Cleaning user input
    noisyInput = "I am sooo saaddd!!! I dont kno wat to dooo... My bf is cheatin on meee :("
    cleanedInput = cleanInput(noisyInput, client)

    print("Cleaned Input: ", cleanedInput) # TESTING PURPOSES ONLY

    # Layer 1 - Extracting Emotions using Pre-trained BERT model (trained on GoEmotions dataset)





    # Layer 2 - Extracting Intent using OpenAI 






