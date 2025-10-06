import os
import json
import functools
import copy
from typing import Dict, List, Tuple, Optional
# from ..utils import store_all_logic
from tasks import MGSM_Task, Task
from collections import Counter
from .agent_module import Agent
import time
import requests
import openai
import traceback
import io
import copy
import anthropic
from dotenv import load_dotenv
from requests.exceptions import Timeout
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# # Direct vLLM imports
# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

class VLLMAgent:
    def __init__(self, model_name, temperature=0.5, max_new_tokens=4096, top_p=1.0, args=None, port=8000, use_direct_api=True, **kwargs) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.args = args or {}
        self.use_direct_api = use_direct_api
        self.use_transformers = False
        
        if "Llama-3.1-8B-Instruct" in model_name:
            self.use_transformers = True
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print(f"Successfully loaded {model_name} with transformers")

        if self.use_transformers == False and self.use_direct_api:
            # Try vLLM first
            try:
                print(f"Initializing vLLM engine directly with model: {model_name}")
                self.llm = LLM(
                    model=model_name,
                    trust_remote_code=True,
                    gpu_memory_utilization=0.9,
                    **kwargs
                )
                self.sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens
                )
                print(f"vLLM initialization successful for {model_name}")
                
                # Load tokenizer for chat template
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    print(f"Loaded tokenizer for {model_name}")
                except Exception as e:
                    print(f"Warning: Could not load tokenizer for {model_name}: {e}")
                    self.tokenizer = None
                    
            except Exception as e:
                print(f"vLLM initialization failed: {e}")
                print("Falling back to transformers library...")
                self.use_transformers = True
                
                try:
                    # Load both model and tokenizer in transformers fallback
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                    print(f"Successfully loaded {model_name} with transformers")
                except Exception as transformers_error:
                    print(f"Transformers fallback also failed: {transformers_error}")
                    raise RuntimeError(f"Both vLLM and transformers failed to load {model_name}. vLLM error: {e}, Transformers error: {transformers_error}")
        elif self.use_transformers == False:
            # Use HTTP client (original approach)
            self.client = openai.OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
            print(f"VLLM Server is running on port {port}")
            self.tokenizer = None

    def _format_messages_to_prompt(self, messages: List[dict]) -> str:
        """Convert OpenAI-style messages to a model-specific prompt string."""
        
        # Try to use the model's chat template first
        if self.tokenizer and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                print(f"Warning: Failed to apply chat template: {e}")
        
        # Fall back to model-specific formats
        model_lower = self.model_name.lower()
        
        if "qwen" in model_lower:
            return self._format_qwen_style(messages)
        elif "llama" in model_lower:
            return self._format_llama_style(messages)
        elif "phi" in model_lower:
            return self._format_phi_style(messages)
        elif "mistral" in model_lower:
            return self._format_mistral_style(messages)
        elif "yi" in model_lower:
            return self._format_yi_style(messages)
        else:
            # Generic fallback
            return self._format_generic_style(messages)

    def _format_qwen_style(self, messages: List[dict]) -> str:
        """Format messages for Qwen models."""
        prompt_parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)

    def _format_llama_style(self, messages: List[dict]) -> str:
        """Format messages for Llama models."""
        # Different Llama models use different formats
        if "llama-3" in self.model_name.lower():
            # Llama 3 format
            prompt_parts = []
            for message in messages:
                role = message["role"]
                content = message["content"]
                if role == "system":
                    prompt_parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>")
                elif role == "user":
                    prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
                elif role == "assistant":
                    prompt_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")
            
            prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return "".join(prompt_parts)
        else:
            # Llama 2 format
            prompt_parts = ["<s>"]
            
            system_message = None
            conversation_parts = []
            
            for message in messages:
                role = message["role"]
                content = message["content"]
                if role == "system":
                    system_message = content
                elif role == "user":
                    conversation_parts.append(("user", content))
                elif role == "assistant":
                    conversation_parts.append(("assistant", content))
            
            # Build conversation
            for i, (role, content) in enumerate(conversation_parts):
                if role == "user":
                    if i == 0 and system_message:
                        # First user message with system message
                        prompt_parts.append(f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{content} [/INST]")
                    else:
                        prompt_parts.append(f"[INST] {content} [/INST]")
                elif role == "assistant":
                    prompt_parts.append(f" {content} </s><s>")
            
            # If the last message is from user, add space for assistant response
            if conversation_parts and conversation_parts[-1][0] == "user":
                prompt_parts.append(" ")
            
            return "".join(prompt_parts)

    def _format_phi_style(self, messages: List[dict]) -> str:
        """Format messages for Phi models."""
        prompt_parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}<|end|>")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}<|end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}<|end|>")
        
        prompt_parts.append("<|assistant|>\n")
        return "\n".join(prompt_parts)

    def _format_generic_style(self, messages: List[dict]) -> str:
        """Generic fallback format."""
        prompt_parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def inference(self, messages: List[dict], num_of_samples: int = 1) -> List[str]:
        if self.use_direct_api:
            if self.use_transformers:
                # Use transformers for inference
                prompt = self._format_messages_to_prompt(messages)
                input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                results = []
                for _ in range(num_of_samples):
                    outputs = self.model.generate(
                        **input_ids,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True if self.temperature > 0 else False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode only the new tokens
                    new_tokens = outputs[0][input_ids["input_ids"].shape[1]:]
                    decoded_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    results.append(decoded_output.strip())
                
                return results
            else:
                # Direct vLLM API
                prompt = self._format_messages_to_prompt(messages)
                
                # Update sampling params for this call
                sampling_params = SamplingParams(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_new_tokens,
                    n=num_of_samples
                )
                
                outputs = self.llm.generate([prompt], sampling_params)
                results = []
                for output in outputs:
                    for completion in output.outputs:
                        results.append(completion.text.strip())
                return results
        else:
            # HTTP client approach
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=self.top_p,
                n=num_of_samples,
                **self.args
            )
            return [choice.message.content for choice in response.choices]

class BasePolicy:
    def __init__(self):
        pass

    def forward(self, messages, num_of_samples): 
        raise NotImplementedError

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_with_backoff(client, **kwargs):
    kwargs["timeout"] = 60
    response = client.chat.completions.create(**kwargs)
    return response

class ClaudePolicy(BasePolicy):
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key=None, key_path='key.env'):
        super().__init__()
        self.model = model
        if api_key is None:
            load_dotenv('key.env')
            api_key = os.getenv('ANTHROPIC_API_KEY')
        self.client = anthropic.Anthropic(api_key=api_key)
        print(f"Teacher Model is {self.model}")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def call_llm(self, messages, num_of_samples):
        candidates = []
        for _ in range(num_of_samples):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=messages,
                timeout=60
            )
            candidates.append(response.content[0].text)
        return candidates

    def forward(self, messages, num_of_samples) -> Tuple[str, bool]:
        raw_actions = self.call_llm(messages, num_of_samples)
        return raw_actions

class GPTPolicy(BasePolicy):
    def __init__(self, model: str = "gpt-4o", api_key=None, key_path='key.env'):
        super().__init__()
        self.model = model
        if api_key is None:
            load_dotenv('key.env')
            api_key = os.getenv('OPENAI_API_KEY')
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        print(f"Teacher Model is {self.model}")

    def call_llm(self,**kwargs):
        response = chat_with_backoff(self.client, **kwargs)
        candidates = []
        for candidate in response.choices:
            z = candidate.message.content
            # pred = re.sub("\n"," ", z)
            # candidates.append(pred.strip())
            candidates.append(z)
        return candidates

    def forward(self, messages, num_of_samples) -> Tuple[str, bool]:
        # Retrieve Action from GPT
        kwargs = {
            "messages": messages,
            "model": self.model,
            "n": num_of_samples
        }
        if self.model == "o3-mini":
            kwargs["reasoning_effort"] = "high"
        raw_actions = self.call_llm(**kwargs)
        return raw_actions

class VLLMPolicy(BasePolicy):
    def __init__(self, model: str = "", response_limit: int = 4096, temperature: float = 0.5, port: int = 8000, use_direct_api: bool = True):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.use_direct_api = use_direct_api
        self.agent = VLLMAgent(
            model_name=model, 
            temperature=temperature, 
            max_new_tokens=response_limit, 
            top_p=1.0, 
            args=None, 
            port=port,
            use_direct_api=use_direct_api
        )

    def forward(self, messages, num_of_samples) -> Tuple[str, bool]:
        raw_actions = self.agent.inference(messages, num_of_samples)
        return raw_actions

class MetaAgent:
    def __init__(self, 
                 model: str = "gpt-4o",
                 task: Task = MGSM_Task,
                 api_key = None,
                 key_path = 'key.env',
                 debug_max: int = 3,
                 num_of_samples: int = 5,
                 temperature: float = 0.5,
                 port: int = 8000,
                 use_direct_api: bool = True):
        
        # Initialize appropriate policy
        if "gpt" in model or "o1" in model or "o3" in model:
            self.policy = GPTPolicy(model=model, api_key=api_key, key_path=key_path)
        elif "claude" in model:
            self.policy = ClaudePolicy(model=model, api_key=api_key, key_path=key_path)
        else:
            self.policy = VLLMPolicy(model=model, temperature=temperature, port=port, use_direct_api=use_direct_api)
        self.task = task
        self.exec_agent = Agent()
        self.debug_max = debug_max
        self.num_of_samples = num_of_samples
        
    def improve_system(self,
                      messages: List[Dict]) -> Tuple[str, str, str, float, float]:
        """
        Improve the current system through meta-learning
        
        Args:
            messages: Conversation history
            eval_history: History of previous systems and corresponding evaluations
            
        Returns:
            Tuple containing:
                - Model output
                - New system code
                - Evaluation feedback  
                - Validation accuracy
                - Test accuracy
        """
        meta_outputs = self.policy.forward(messages, self.num_of_samples)
        
        best_result = {
            'test_acc': 0,
            'system_code': "",
            'model_output': "",
            'eval_feedback': "Valid Accuracy: 0 \n All 0 error.",
            'valid_acc': 0
        }

        model_outputs, new_systems, eval_feedbacks, valid_accs, test_accs = [], [], [], [], [] 
        eval_feedback = ""
        
        for output in meta_outputs:
            debug_messages = [messages[-1]]
            original_output = output
            # Try to evaluate with debug retries
            for _ in range(self.debug_max):
                print('Meta Agent Output:')
                print(output)
                print('Meta Agent Output End')
                try:
                    system_code = self._extract_system_code(output)
                    eval_feedback, valid_acc, test_acc = self.evaluate_on_task(system_code)
                    if valid_acc <= 0.05:
                        print('All 0 Error')
                        break
                    
                    code_start = original_output.rfind("```python")
                    code_end = original_output.find("```", code_start + len("```python"))
                    if code_start != -1 and code_end != -1:
                        updated_output = (
                            original_output[:code_start + len("```python")] + 
                            "\n" + system_code + "\n" + 
                            original_output[code_end:]
                        )
                    else:
                        updated_output = original_output + '\n' + 'system_code'
                    
                    model_outputs.append(updated_output)
                    new_systems.append(system_code)
                    eval_feedbacks.append(eval_feedback)
                    valid_accs.append(valid_acc) 
                    test_accs.append(test_acc)
                    if valid_acc > best_result['valid_acc']:

                        best_result.update({
                            'test_acc': test_acc,
                            'system_code': system_code,
                            'model_output': updated_output,
                            'eval_feedback': eval_feedback,
                            'valid_acc': valid_acc
                        })
                    break
                    
                except Exception as e:
                    print('Start Error Handling')
                    exception_stringio = io.StringIO()
                    traceback.print_exc(file=exception_stringio)
                    error = "Error " + exception_stringio.getvalue()

                    # Get the actual line of code that caused the error
                    tb = traceback.extract_tb(e.__traceback__)
                    for frame in tb:
                        if frame.filename == "<string>":
                            error_line_no = frame.lineno
                            code_lines = system_code.split('\n')
                            if 0 <= error_line_no - 1 < len(code_lines):
                                error_line = code_lines[error_line_no - 1].strip()
                                error += f"\nError occurred in line {error_line_no}: {error_line}"
                            break
                    print(error)
                    exception_stringio.close()
                    debug_messages.append({"role": "assistant", "content": output})
                    debug_messages.append({
                        "role": "user", 
                        "content": f"""Error during evaluation:
{error}

WARNING: DO NOT USE ANY TRY-EXCEPT BLOCKS IN YOUR SOLUTION.
Your task is to fix the root cause of the error, not to catch it.

Requirements:
1. Analyze the error message in detail
2. Explain the specific changes needed to fix the core issue
3. Provide a clean implementation that solves the problem directly
4. Do not include any error handling or try-except blocks

Please strictly follow the following output format:

[Your analysis here]

Code:
```python
def solver(agent, task: str):
    \"""
    Fill in your code here.
    \"""
    return return_dict
``` """})
                    try:
                        output = self.policy.forward(debug_messages, 1)[0]
                    except Exception as e:
                        print(f'Error in policy.forward: {str(e)}')
                        break
                    continue
        if eval_feedback == "":
            return self.improve_system(messages)
        return model_outputs, new_systems, eval_feedbacks, valid_accs, test_accs, best_result
    
    def _extract_system_code(self, meta_output: str) -> str:
        """Extract system code from meta agent output."""
        # Find all python code blocks
        current_pos = len(meta_output)
        while True:
            code_start = meta_output.rfind("```python", 0, current_pos)
            if code_start == -1:
                raise ValueError("Could not find solver function definition in code section. You should implement the solver(agent,task) function.")
                
            code = meta_output[code_start + len("```python"):]
            code_end = code.find("```")
            if code_end == -1:
                # raise ValueError("Unclosed code block")
                current_pos = code_start
                continue
                
            code_block = code[:code_end].strip()
            
            # Check if this block contains the solver function
            if "def solver" in code_block:
                return code_block
                
            # Move search position before this code block
            current_pos = code_start
    
    def evaluate_on_task(self, code_str: str):
        """
        Evaluate the current solver on the goal task samples and return the evaluation feedback.

        Args:
            solver (str): String representation of the solver function code

        Returns:
            feedback (str): Evaluation feedback including valid set accuracy, test set accuray, test sample inputs, model outputs and valid sample answer.
        """
        module_namespace = self._execute_code_string(code_str)
        solver = module_namespace['solver']
        
        # Create a wrapped solver that has access to the module namespace
        def wrapped_solver(agent, task):
            # Update globals with the module namespace to make all definitions available
            global_dict = globals().copy()
            global_dict.update(module_namespace)
            # Call the original solver with the updated globals
            with modified_globals(global_dict):
                return solver(agent, task)
        feedback, valid_acc, test_acc = self.task.evaluate(functools.partial(wrapped_solver, self.exec_agent))
        if test_acc > self.task.last_test_acc:
            # store_all_logic(f"../{self.task.task_name}_{round(test_acc, 4)}")
            self.task.last_test_acc = test_acc
        return feedback, valid_acc, test_acc

    def _execute_code_string(self, code_str: str, func_name: str = "solver") -> callable:
        """
        Execute a string representation of code and return the resulting namespace.
        
        Args:
            code_str (str): String representation of the complete code including imports
            func_name (str): Name to give to the function (default: "solver")
        
        Returns:
            callable: The compiled function object
        """
        # Create namespace dictionary to store the function
        namespace = {}
        # Execute the function string in the namespace
        try:
            exec(code_str, globals(), namespace)
            if 'solver' not in namespace:
                raise ValueError("Could not find solver function in the provided code")
            return namespace
        except Exception as e:
            raise ValueError(f"Failed to convert string to function: {str(e)}")


class modified_globals:
    """Context manager for temporarily modifying globals"""
    def __init__(self, new_globals):
        self.new_globals = new_globals
        self.original_globals = None

    def __enter__(self):
        self.original_globals = globals().copy()
        globals().clear()
        globals().update(self.new_globals)

    def __exit__(self, exc_type, exc_val, exc_tb):
        globals().clear()
        globals().update(self.original_globals)

