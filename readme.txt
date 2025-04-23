Check out github https://github.com/ChenHY1217/FriendBot

Before running:

run in terminal to install dependencies: pip install python-dotenv openai torch transformers pandas scikit-learn
create .env file and add to it: OPENAI_API_KEY = "your_api_key_here"

Below are the instructions for running the main code / FriendBot:

py main.py

This will run FriendBot, capable of multi-turn conversations with storing of conversation history. For each input, the user will receive two responses side by side, one from Base Model GPT-4o and one from our 3-layer Architecture.

Below are the instructions for running mainTest, a single test for a specific input:

py mainTest.py

It has a default NOISY_INPUT value for user input. Feel free to change it to test other inputs.

It will run what is in main.py but only once. However, this version will have all the intermediate information like predicted emotions and extracted intent. Then it will do an Emotional Intelligence Evaluation / Comparison of the two model responses and declare the "winner".

Below are the instructions for running the evaluation pipeline for final response evaluation based on 50 question dataset:

py evaluation_pipeline.py

This runs the comparison process between the two models for each question in the given dataset (dating_emotion_dataset.csv). It tallies the win rates of each model and prints final statistics at the end. It also outputs two csv files (evaluation_results.csv and evaluation_detailed_results.csv) with details on each comparison.

For the Emotion_BERT.ipynb file:

Feel free to run the notebook. It is used in validation process for testing accuracy of BERT model.