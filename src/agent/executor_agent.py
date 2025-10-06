import os
import io
import re
import ast
import sys
import json
import typing
import inspect
import functools
import itertools
import traceback
import importlib
import subprocess
import contextlib
import collections
import openai
import concurrent.futures
import multiprocessing
import queue
import time
import threading
import anthropic
from dotenv import load_dotenv
from utils.sanitize import has_probable_infinite_loop, sanitize
from utils.helper import extract_test_cases_from_jsonl, test_case_2_test_function
code_execution_lock = threading.Lock()
code_execution_semaphore = threading.BoundedSemaphore(4)
execution_queue = queue.Queue()



class AgentBase:
    def action_call_llm(agent, *args, **kwargs):
        raise NotImplementedError("action_call_llm hasn't been implemented")

class Agent(AgentBase):
    def __init__(agent, api_key=None, key_path='key.env'):
        # Load configurations
        if api_key is None:
            load_dotenv('key.env')
            api_key = os.getenv('OPENAI_API_KEY')
        openai.api_key = api_key
        agent.client = openai.OpenAI(api_key=api_key)
        agent.token_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0
        }
    
    def call_json_format_llm(
        agent,
        *,
        messages: typing.List[typing.Dict[str, str]], 
        model: typing.Literal["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"] = "gpt-3.5-turbo", 
        temperature: float = 1.0, 
        max_completion_tokens: int = 4096, 
        num_of_response: int = 1,
        agent_role: str = "task solver", 
        return_dict_keys: typing.List[str] = [], 
        instructions: str = "", 
    ):  
        # Check if messages is a list
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        # Check each message in messages
        for i, msg in enumerate(messages):
            # Check if message is a dict
            if not isinstance(msg, dict):
                raise ValueError(f"message at index {i} must be a dictionary")
            
            # Check if message has exactly 2 keys: 'role' and 'content'
            msg_keys = set(msg.keys())
            expected_keys = {'role', 'content'}
            if msg_keys != expected_keys:
                extra_keys = msg_keys - expected_keys
                missing_keys = expected_keys - msg_keys
                error_msg = f"message at index {i} must contain exactly two keys: 'role' and 'content'."
                if extra_keys:
                    error_msg += f" Found extra keys: {extra_keys}."
                if missing_keys:
                    error_msg += f" Missing required keys: {missing_keys}."
                raise ValueError(error_msg)
            # Check if values are strings
            if not isinstance(msg['role'], str):
                raise ValueError(f"'role' at index {i} must be a string")
            if not isinstance(msg['content'], str):
                raise ValueError(f"'content' at index {i} must be a string")
        system_prompt = (
            f"You are a helpful {agent_role}.\n"
            f"Reply in JSON format, ONLY using the keys {return_dict_keys}.\n"
            f"{instructions}"
        ).strip()
        _messages = [{"role": "developer", "content": system_prompt}, *messages]
        return_dicts = agent.call_llm(model=model,
                                    messages=_messages, 
                                    temperature=temperature,
                                    max_completion_tokens=max_completion_tokens,
                                    num_of_response=num_of_response,
                                    response_format="json")
        
        for key in return_dict_keys:
            for return_dict in return_dicts:
                if key not in return_dict:
                    return_dict[key] = f"NO {key} IN DICTIONARY"
        return return_dicts
        
    def call_llm(
        agent, 
        *,
        model: typing.Literal["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"] = "gpt-4o-mini", 
        messages: typing.List[typing.Dict[str, str]], 
        temperature: float = 1.0, 
        max_completion_tokens: int = 4096, 
        num_of_response: int = 1,
        response_format: typing.Literal["text", "json", "json_object"] = "text", 
        agent_role: str = "task solver", 
        instructions: str = "",
        tools=None, 
        tool_choice=None,
    ):
        """
        Sends a request to the OpenAI LLM with a system prompt and user message, and returns the response.

        Args:
            agent (Agent): The OpenAI client instance used to interact with the LLM.
            messages (List[Dict[str, str]]): A list of message dictionaries (conversation history).
            response_format (str): The desired format of the LLM's output.
            model (str): Specifies which LLM model to use.
            temperature (float): A float value controlling the randomness of the model's responses. Higher values (e.g., 1.0) increase creativity, while lower values (e.g., 0.1) make the responses more focused and deterministic.
            max_completion_tokens: An integer defining the maximum number of tokens in the completion response, up to 4096.
            num_of_response (int): The number of chat completion choices to generate for each input message.

        Returns:
            response (dict): The response from the OpenAI LLM.
        """
        if messages[0]['role'] != "system":
            system_prompt = (
                f"You are a helpful {agent_role}.\n"
                f"{instructions}"
            ).strip()
            messages = [{"role": "system", "content": system_prompt}, *messages]
        try:
            if response_format == "json":
                response_format = "json_object"
            
            import copy
            messages = copy.deepcopy(messages)
            for message in messages:
                message["content"] = str(message["content"])
            
            kwargs = {
                "n": num_of_response,
                "model": model,
                "messages": messages,
                "response_format": {"type": response_format if response_format == "json_object" else "text"}, 
                "temperature": temperature,
                "max_completion_tokens": max_completion_tokens
            }

            if tools is not None:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice

            response = agent.client.chat.completions.create(**kwargs).to_dict() # to Python dictionary
            
            usage = response["usage"]
            agent.token_usage["prompt_tokens"] += usage["prompt_tokens"]
            agent.token_usage["completion_tokens"] += usage["completion_tokens"]

            def try_parse_json(content):
                try:
                    return json.loads(content)
                except:
                    return {"JSONDecodeError": content}

            if response_format == "text":
                return [choice["message"]["content"] for choice in response["choices"]]
            else:
                return [try_parse_json(choice["message"]["content"]) for choice in response["choices"]]
        except Exception as e:
            print(e)
            raise e
    
    def execute_code(agent, code: str, timeout=5):
        """
        Execute code string in a safe namespace and return the result.
        
        Args:
            code (str): Python code string to execute
            
        Returns:
            The result of executing the code
        """
        unsafe_patterns = [
            "import os", "import sys", "import subprocess",
            "eval", "exec", "open", "__import__",
            "system", "popen", "spawn", "while", "input()",
            "matplotlib", "plt", "figure", "plot" # Block matplotlib to avoid main thread issues
        ]
        if any(pattern in code for pattern in unsafe_patterns):
            raise ValueError("Unsafe operation detected in code")
        if has_probable_infinite_loop(code):
            raise ValueError("Potential infinite loop detected in code")
        if not code_execution_semaphore.acquire(timeout=timeout):
            raise TimeoutError("Failed to acquire execution semaphore")
        try:
            # print(code)
            namespace = {}
            stdout = io.StringIO()
            stderr = io.StringIO()
            start_time = time.time()
            with code_execution_lock:
                if time.time() - start_time > timeout:
                    raise TimeoutError("Timeout while waiting for lock")
                
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec(code, namespace)
                    
                    if "solution" not in namespace or not callable(namespace["solution"]):
                        # Get last line of code to extract result if no solution() function
                        code_lines = code.strip().split('\n')
                        last_line = code_lines[-1].strip()
                        
                        # Try to get result from last line
                        if '=' in last_line:
                            result_var = last_line.split('=')[0].strip()
                            result = namespace[result_var]
                        else:
                            # Last line is the result itself
                            result = eval(last_line, namespace)
                            
                        return result
                    
                    result = namespace["solution"]()
                    return result
                
        except Exception as e:
            error_msg = f"Error when executing code: {str(e)}"
            if stderr.getvalue():
                error_msg += f"\nStderr: {stderr.getvalue()}"
            raise Exception(error_msg) from e
        
        finally:
            stdout.close()
            stderr.close()
            code_execution_semaphore.release()
    
    def exec_test(agent, solution, entry_point):

        test_cases = extract_test_cases_from_jsonl(entry_point, dataset_name="mbpp")
                
        fail_cases = []
        for test_case in test_cases:
            test_code = test_case_2_test_function(solution, test_case, entry_point)
            try:
                exec(test_code, globals())
            except AssertionError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                with open("tester.txt", "a") as f:
                    f.write("test_error of " + entry_point + "\n")
                error_infomation = {
                    "test_fail_case": {
                        "test_case": test_case,
                        "error_type": "AssertionError",
                        "error_message": str(e),
                        "traceback": tb_str,
                    }
                }
                fail_cases.append(error_infomation)
            except Exception as e:
                with open("tester.txt", "a") as f:
                    f.write(entry_point + " " + str(e) + "\n")
                return {"exec_fail_case": str(e)}
        if fail_cases != []:
            return fail_cases
        else:
            return "no error"
    
    def test_on_public_test(agent, task, solution, entry_point, test_loop = 3):
        count = 0
        for _ in range(test_loop):
            result = agent.exec_test(solution, entry_point)
            if result == "no error":
                return {"result": True, "solution": solution, 'feedback': result, "count": count}
            elif "exec_fail_case" in result:
                result = result["exec_fail_case"]
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=task,
                    solution=solution,
                    exec_pass=f"executed unsuccessfully, error: \n {result}",
                    test_fail="executed unsucessfully",
                )
                response = agent.call_json_format_llm(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    agent_role="Expert Python Programmer",
                    return_dict_keys=["reflection", "code"],
                    instructions="Requirements:\n1. Generate a reflection on the failed test cases and code solution, followed by a better code solution without any additional text or test cases.",
                )[0]
                solution = response["code"]
            else:
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=task,
                    solution=solution,
                    exec_pass="executed successfully",
                    test_fail=result,
                )
                response = agent.call_json_format_llm(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    return_dict_keys=["reflection", "code"],
                    agent_role="Expert Python Programmer",
                    instructions="Requirements:\n1. Generate a reflection on the failed test cases and code solution, followed by a better code solution without any additional text or test cases.",
                )[0]
                solution = response["code"]
            count += 1
        
        result = agent.exec_test(solution, entry_point)
        if result == "no error":
            return {"result": True, "solution": solution, 'feedback': result, "count": count}
        else:
            return {"result": False, "solution": solution, "feedback": result, "count": count}
    
    def extract_answer_str(agent, answer_str):
        # Extract the LaTeX answer enclosed in \boxed{}
        pattern = r"\\boxed{((?:[^{}]|{(?:[^{}]|{[^{}]*})*})*)}"
        boxed_matches = re.findall(pattern, answer_str, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        # Extract the last number in the answer string
        matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(answer_str))
        if matches:
            last_number = matches[-1].replace(",", "")
            return last_number

        # Extract the last sentence in the answer string
        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, answer_str)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    def extract_code_block(agent,response, entry_point="solution"):
        import re
        code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if not code_blocks:
            code_blocks = re.findall(r'```\n(.*?)\n```', response, re.DOTALL)
        
        solution = ""
        for block in code_blocks:
            if f"def {entry_point}" in block:
                solution = block
                break
        return solution


REFLECTION_ON_PUBLIC_TEST_PROMPT = """
Given a code problem and a python code solution which failed to pass test or execute, you need to analyze the reason for the failure and propose a better code solution.: 
### problem
{problem}

### Code Solution
{solution}

### Execution Result
{exec_pass}

#### Failed Test Case
{test_fail}

Please provide a reflection on the failed test cases and code solution, followed by a better code solution without any additional text or test cases.
"""