import json
import time
from openai import OpenAI
import time

with open(r"jsons\config.json", 'r', encoding='utf-8') as f:
    config = json.load(f)

    
class Agent():
    def __init__(self, model_name, agent_name, temperature):
        self.Model_name = model_name
        self.Agent_name = agent_name
        self.temperature = temperature
        self.memory_lst = []
        
    def add_system(self, content):
        self.memory_lst.append({"role": "system", "content": f"{content}"})

    def add_assistant(self, content):
        self.memory_lst.append({"role": "assistant", "content": f"{content}"})

    def add_user(self, content):
        self.memory_lst.append({"role": "user", "content": f"{content}"})
    
    def ask(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
        
        
class GPT_Agent(Agent):
    def __init__(self, model_name, agent_name, temperature):
        super().__init__(model_name, agent_name, temperature)
        self.client = OpenAI(api_key = config['openai_api_key'], base_url = "https://api.openai.com/v1/responses") 
        
    def ask(self):
        try:
            start_time = time.time()
            response = self.client.responses.create(
                model = self.Model_name,
                input = self.memory_lst,
                temperature = self.temperature,
                reasoning={
                    "effort": "none"
                }
            )
            end_time = time.time()
            return response, end_time - start_time
        except Exception as e:
            print(f"Error with model {self.Model_name}: {e}")
            time.sleep(1)
            return self.ask()
        


class Qwen_Agent(Agent):
    def __init__(self, model_name, agent_name, temperature):
        super().__init__(model_name, agent_name, temperature)
        self.client = OpenAI(api_key=config['aliyun_api_key'], base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        
    def ask(self):
        try:
            start_time = time.time()
            response = self.client.responses.create(
                model = self.Model_name,
                input = self.memory_lst,
                temperature = self.temperature,
                reasoning={
                    "effort": "none"
                }
            )
            end_time = time.time()
            return response, end_time - start_time
        except Exception as e:
            print(f"Error with model {self.Model_name}: {e}")
            time.sleep(1)
            return self.ask()


        
class Gemini_Agent(Agent):
    def __init__(self, model_name, agent_name, temperature):
        super().__init__(model_name, agent_name, temperature)
        self.client = OpenAI(api_key = config['gemini_api_key'], base_url = "https://generativelanguage.googleapis.com/v1beta/")
        
    def ask(self):
        try:
            start_time = time.time()
            completion = self.client.chat.completions.create(
                model = self.Model_name,
                messages = self.memory_lst,
                temperature = self.temperature,
                reasoning_effort = 'low'
            )
            end_time = time.time()
            return completion, end_time - start_time
        except Exception as e:
            print(f"Error with model {self.Model_name}: {e}")
            time.sleep(1)
            return self.ask()


class Deepseek_Agent(Agent):
    def __init__(self, model_name, agent_name, temperature):
        super().__init__(model_name, agent_name, temperature)
        self.client = OpenAI(api_key = config['deepseek_api_key'], base_url = "https://api.deepseek.com") 
        
    def ask(self):
        try:
            start_time = time.time()
            completion = self.client.chat.completions.create(
                model = self.Model_name,
                messages = self.memory_lst,
                temperature = self.temperature,
                extra_body={"thinking": {"type": "disabled"}}
            )
            end_time = time.time()
            return completion, end_time - start_time
        except Exception as e:
            print(f"Error with model {self.Model_name}: {e}")
            time.sleep(1)
            return self.ask()