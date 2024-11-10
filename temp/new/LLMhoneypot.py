import google.generativeai as genai
from dotenv import load_dotenv
import argparse
import requests
from datetime import datetime
import yaml
import os

#load environment variables from .env file
load_dotenv()
api_key = os.getenv("API_KEY")

#configure the generative AI with the API key
genai.configure(api_key=api_key)

today = datetime.now()
#open history file for logging
last_login_placeholder = "Last login: {last_login}"
def generate_last_login():
    try:
        # Fetch IP and location data
        response = requests.get("https://ipinfo.io")
        data = response.json()
        
        # Get IP and other location details
        ip_address = data.get("ip", "unknown")
        
        # Get current date and time
        current_time = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
        
        # Create the last login string
        # last_login_string = f"Last login: {current_time} from {ip_address}"
        return last_login_placeholder.format(last_login=f"{current_time} from {ip_address}")

    except requests.RequestException:
        return last_login_placeholder.format(last_login=f"{current_time} from {'192.168.1.1'}")

history = open("history.txt", "a+", encoding="utf-8")

if os.stat('history.txt').st_size == 0:
    #if empty then load personality settings from a YAML file
    with open('personalitySSH.yml', 'r', encoding="utf-8") as file:
        identity = yaml.safe_load(file)

    identity = identity['personality'] #get personality
    prompt = identity['prompt'] #get prompt
    
    prompt += (
    f"Example 1.\n{generate_last_login()}\nbrian@biolab:~$ \n"
    f"Example 2.\n{generate_last_login()}\nkatherin@aicenter:~$ \n"
    f"Example 3.\n{generate_last_login()}\nwalter@strato:~$ "
    )
else:
    #if not empty then write a message indicating session continuation
    history.write("\nHere the session stopped. Now you will start it again from the beginning with the same user. You must respond just with starting message and nothing more.\n")
    history.seek(0)
    prompt = history.read()

def main():
    #create a parser for cli args
    parser = argparse.ArgumentParser(description="Simple command line with Gemini (Google AI)")
    args, unknown = parser.parse_known_args()

    #create the initial prompt for the AI model in this case is gemini-1.5-flash
    initial_prompt = f"You are a Linux OS terminal. Your personality is: {prompt}\n" \
                     f"Always respond with realistic terminal output based on user commands.\n"

    messages = [{"role": "system", "content": initial_prompt}]
    #write into history
    logs = open("history.txt", "a+", encoding="utf-8")
    #if history empty
    if os.stat('history.txt').st_size == 0:
        for msg in messages:
            history.write(msg["content"])
    else:
        history.write("The session continues in the following lines.\n\n")
    # history.close()

    model = genai.GenerativeModel("gemini-1.5-flash") #set the model

    # Initialize dynamic user and host variables
    user = "user"  # Default user
    host = "awesome-server"  # Default host

    while True:
        try:
            # Print the dynamic prompt before the user input without the extra symbols
            # print(f"[{user}@{host}:~]$", end=' ')  # Adjust the prompt display
            user_input = input()  # Get user input without the extra symbols

            # Generate a response based on user input
            prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]) + f"\nuser: {user_input}\n"

            response = model.generate_content(prompt_text)
            msg = response.text.strip()
            msg = msg.replace("```", "").strip()  # Remove the code block indicators


            if not msg:
                print("No response generated. Please try a different command.")
                continue

            # Append the assistant's response to messages
            message = {"content": msg, "role": 'assistant'}
            messages.append(message)

            # Extract username and host from the assistant's response if included
            if "@" in msg:
                parts = msg.split()
                for part in parts:
                    if "@" in part:
                        user_host = part.split("@")
                        if len(user_host) == 2:
                            user, host = user_host  # Update user and host

            # Log interaction
            logs = open("history.txt", "a+", encoding="utf-8")
            logs.write(f"assistant: {message['content']}\n")
            logs.close()

            # Print the assistant's response
            print(message['content'], end=' ')

            # Log user input
            logs = open("history.txt", "a+", encoding="utf-8")
            logs.write(f"user: {user_input}\n")
            logs.close()

        except KeyboardInterrupt:
            print("")
            break


if __name__ == "__main__":
    main()


