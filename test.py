from prompts import IntentPrompt, ResponsePrompt  # Importing the prompts from prompts.py

input_text = "I am sooo saaddd!!! I dont kno wat to dooo... My bf is cheatin on meee :("

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
    
