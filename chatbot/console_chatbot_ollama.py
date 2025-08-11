import ollama
import random

def chat_to_ollama(history_contents):
    response = ollama.Client(
        host="http://127.0.0.1:11434"
    ).chat(
        # model='qwen3:4b',
        model="gemma3:4b",
        stream=True,
        messages=history_contents,
        options={
            "temperature": 0.5,
            "num_ctx": 2048,
            "top_k": 30,
            "top_p": 0.8,
            "do_sample": True,
            "repeat_penalty": 1.2,
            # "seed": random.randint(0, 1000000000)
            })
    return response

def main():
    # Initialize chat history
    history = [
        {"role": "system", "content": "你是一个AI助手。"},
        {"role": "assistant", "content": "你好，有什么我可以帮你的吗？"}
    ]
    
    # Print initial message
    print(f"AI: {history[-1]['content']}")
    
    while True:
        # Get user input
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        # Add user message to history
        user_message = {"role": "user", "content": query}
        history.append(user_message)
        
        try:
            # Get AI response
            response = chat_to_ollama(history)
            assistant_response = ""
            
            print("\nAI: ", end="", flush=True)
            for chunk in response:
                if chunk.get('message') and chunk['message'].get('content'):
                    content = chunk['message']['content']
                    assistant_response += content
                    print(content, end="", flush=True)
            
            # Add AI response to history
            history.append({"role": "assistant", "content": assistant_response})
            print("\n")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            history.append({"role": "assistant", "content": f"Error: {str(e)}"})

if __name__ == "__main__":
    main()
