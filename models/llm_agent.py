import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
class GeminiModel:
    def __init__(self):
        # Ref: https://ai.google.dev/gemini-api/docs/openai
        self.client = OpenAI(api_key=os.environ.get("GEMINI_API_KEY"), 
                             base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    
    def get_response(self, system_prompt: str, user_prompt: str, model_name: str = "gemini-1.5-flash") -> str:
        """
        Get response from Gemini model.
        """
        # model = genai.GenerativeModel("gemini-1.5-flash")
        response = self.client.chat.completions.create(
            model="gemini-2.0-flash",
            n=1,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ]
        )
        return response.choices[0].message.content
    
class GroqModel:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )

    def get_response(self, system_prompt: str, user_prompt: str, model_name: str = "llama3-70b-8192") -> str:
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    
class OpenAiModel:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) 
        
    def get_response(self, system_prompt: str, user_prompt: str, model_name: str = "gpt-4o-mini", temperature: float = 1.0, num_samples: int = 1) -> str:
        """
        Get response from OpenAI model.
        """
        response = self.client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            n=num_samples,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        if num_samples == 1:
            return response.choices[0].message.content
        else:
            return [choice.message.content for choice in response.choices]

class LLM_Agent:
    def __init__(self, model) -> None:
        self.model = model
        
    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Get response from the LLM model.
        """
        return self.model.get_response(system_prompt, user_prompt)

if __name__ == '__main__':
    # Example usage, using the Gemini Model. 
    gemini_agent = LLM_Agent(GeminiModel())
    response = gemini_agent.get_response(
        system_prompt="You are a helpful assistant.",
        user_prompt="What is the capital of France?"
    )
    print(response)
    
    # Example usage, using the OpenAI Model. 
    openai_agent = LLM_Agent(OpenAiModel())
    response = openai_agent.get_response(
        system_prompt="You are a helpful assistant.",
        user_prompt="What is the capital of France?"
    )
    print(response)