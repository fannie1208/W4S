import json

TASK_DESCRIPTION = """
Grade School Math (GSM) Hard Benchmark.
**Type**: Mathematical problem solving
**Answer Type**: The answer must be a float.
**Description**: GSM Hard is a harder version of the GSM8k benchmark dataset that contains more challenging mathematical problems. 
The dataset focuses on complex multi-step reasoning problems that require deeper mathematical understanding.

Example Question:
The girls are trying to raise money for a carnival. Kim raises $320 more than Alexandra, who raises $430, and Maryam raises $400 more than Sarah, who raises $300. How much money, in dollars, did they all raise in total?

Example solution code:"def solution():\n   \"\"\"The girls are trying to raise money for a carnival. Kim raises $320 more than Alexandra, who raises $430, and Maryam raises $400 more than Sarah, who raises $300. How much money, in dollars, did they all raise in total?\"\"\"\n   alexandra_money = 430\n   kim_money = alexandra_money + 320\n   sarah_money = 300\n   maryam_money = sarah_money + 400\n   total_money = alexandra_money + kim_money + sarah_money + maryam_money\n   result = total_money\n   return result"

Final answer: 2180
"""

SYSTEM = """You are an AI agent system improvement expert specializing in LLM prompting techniques and state-of-the-art LLM agent architectures. Your mission is to evolve and optimize agentic systems through innovative prompts, strategies, and architectural patterns. Your core focus is on continuously enhancing system performance through:

1. Careful analysis of historical agentic systems and their performance feedback
2. Creative exploration of novel architectures and techniques 
3. Systematic improvement by optimizing the agentic system code based on empirical results

You will carefully study evaluation feedback to extract actionable insights and identify promising directions for improvement. Think critically about what worked, what didn't, and why. Use this understanding to design targeted enhancements while maintaining system stability.

Your improvements should push boundaries through principled innovation - each iteration building upon proven successes while thoughtfully exploring new approaches. Draw inspiration broadly from LLM agent research and other relevant fields.
"""

USER = """### **Agentic System Interface**:
Function you should optimize: `solver(agent, task: str) -> dict`
- Description: Solve the target task using current agent.
- Input: task (str) - The question to be solved
- Output: The output of the function MUST be a dictionary, and the final answer MUST be placed under the key `"answer"`.
- Available API: 
  + `agent.call_json_format_llm(messages, temperature, num_of_response, agent_role, return_dict_keys, instructions)`: Call OpenAI APIs and return a list of dictionary format responses containing the keys specified in `return_dict_keys`.
  + `agent.execute_code(code)`: Execute the code and return the output. The code MUST contain a `solution` function. The output of `execute_code(code)` will be the return value of the `solution` function if the code is executed successfully or raise an exception.

### **Guiding Principles**:

+ **Analysis before improving**: 
    + **Always analysis first before modifying the code of `solver`.** 
    + Analysis may include following things: reasonable plan about improving performance, **CASE STUDIES of LOW SCORE valid examples of EVLUATION FEEDBACK**, other possible solving ideas. 
    + Your system will be tested on a holdout test set finally, so you MUST NOT overfit to the validation samples provided in case studies.
+ **Improve `solver`**:
    + Do not change the input and output of the function.
    + Do not do unnecessary changes.
    + In `messages`, the value of `role` can only be `user` or `assistant`.
    + Can call `call_json_format_llm` multiple times and across multiple rounds in the solver to improve performance.
    + When multiple outputs are required, set `num_of_response`, a parameter of `call_json_format_llm`, to the required number of outputs in the function.
    + Additionally, can call different role-based LLMs by specifying and MUST specifying the `agent_role` and `instructions` to further assist task-solving.
    + For each key, if a specific format is required, such as int, float, enum or list, the instructions must specify the conditions.
    + Improve the prompts according to task descriptions, task inputs/outpus (shown in valid samples), LLM roles or techniques.
    + Do not include any system prompt in 'messages' since the system prompt will be automated created using `agent_role` and `instructions` when calling `call_json_format_llm()`.
    + If performance doesn't improve, explore alternative methods.
    + When calling `call_json_format_llm`, you can use change the `return_dict_keys` to get different outputs.
        1. e.g., `return_dict_keys=["reasoning", "answer"]` to get the reasoning and answer.
        2. e.g., `return_dict_keys=["reasoning", "code"]` to get the reasoning and code.
    + Can call `agent.execute_code(code)` to safely execute code generated by LLMs. If the code is not executed successfully, you can implement a try-except mechanism to handle failures gracefully. The recommended approach is to:
        1. Can increase number of code generation and execution
        2. Can combine with self-consistency
        3. If code execution fails, fall back to direct LLM reasoning using `call_json_format_llm`
        4. Optionally, implement multiple fallback strategies with different prompting techniques
    + Explore techniques like:
        + **Chain-of-Thought**: Solving problems by breaking them down into a sequence of steps, reasoning through each step. You can add `reasoning` to the `return_dict_keys` parameter of `call_json_format_llm()` to get the reasoning of the LLM.
        + **Self-consistency**: Ensuring coherence by comparing multiple outputs and selecting the most consistent one. (Can try to increase `num_of_response` and use majority voting to get high score)
        + **LLM Debate**: Multiple models engage in a discussion to critique and refine responses, improving solution quality. You can try to increase `num_of_response` or number of different models/roles.
        + **Dynamic Assignment of Roles**: Assigning and adjusting roles among AI components dynamically to enhance task performance.
        + **Task Decomposition**: Dividing complex tasks into smaller subtasks, solving them individually, and reintegrating the solutions for overall task success.
        + **Reflective Evaluation**: Reviewing performance after task completion to identify successes and failures, enabling continuous self-improvement.
        + **Iterative Refinement**: Revising the response iteratively based on its own feedback on its previous answer.
        + **Code Generation**: Call LLM to write executable Python code to solve the task programmatically. The LLM can generate code that implements mathematical calculations, algorithms. This technique is useful for tasks requiring precise calculations or complex algorithmic solutions.
    + Can combine above techniques.

### Task Description
The task your designed agentic system should solve is:

[TASK]

### History Agentic Systems
Here is the archive of the history agentic systems and their evaluation feedback. 
'system code' is the code of the solver function
'eval_feedback' includes performance metrics and randomly selected validation samples:

[ARCHIVE]

### Output Format
You MUST respond with:
1. Your analysis

2. A complete implementation of the solver function in a Python code block, formatted EXACTLY as follows:
```python
def solver(agent, task: str):
    \"""
    Solve the target task using current agent. Use `agent.call_json_format_llm` to call OpenAI APIs.
    Fill in your code here. Any helper functions or import should be included in this function.
    \"""
    return return_dict
```
"""

COT = {"name":"Chain-of-Thought",
       "code":"""
def solver(agent, task: str):
    math_expert_instructions = "Requirements:\\n1. Please explain step by step.\\n2. The answer MUST be a float.\\n"
    messages = [{"role": "user", "content": f"# Your Task:\\n{task}"}]
    response = agent.call_json_format_llm(
        messages=messages, 
        temperature=0.8, 
        num_of_response=1,
        agent_role="math expert", 
        return_dict_keys=["reasoning", "answer"], 
        instructions=math_expert_instructions.strip(),
    )
    
    return_dict = response[0]
    return_dict["answer"] = str(return_dict.get("answer", ""))
    return return_dict
"""
}

COT_SC = {"name":"Chain-of-Thought with Self Consistency",
       "code":"""def solver(agent, task: str):
    from collections import Counter
    math_expert_instructions = "Requirements:\\n1. Please explain step by step.\\n2. The answer MUST be a float.\\n"
    messages = [{"role": "user", "content": f"# Your Task:\\n{task}"}]
    
    # Generate multiple solutions with different temperatures
    responses = agent.call_json_format_llm(
        messages=messages,
        temperature=0.8,
        num_of_response=5, # Generate 5 different solutions
        agent_role="math expert",
        return_dict_keys=["reasoning", "answer"],
        instructions=math_expert_instructions.strip(),
    )

    answers = []
    for response in responses:
        
        try:
            answer = str(response.get("answer", ""))
            answers.append(answer)
        except:
            continue
        
    answer_counts = Counter(answers)
    most_common_answer = answer_counts.most_common(1)[0][0]
    
    return_dict = {
        "answer": most_common_answer,
    }
    
    return return_dict
"""
}

Code_Generation = {
    "name": "Code Generation",
    "code": """def solver(agent, task: str):
    programmer_instructions = "You should generate valid Python code to solve the math problem. Requirements:\\n1. The code must define a solution() function and return only the final numerical answer.\\n2. Use only basic arithmetic operation.\\n3. Do not introduce a dead loop."
    messages = [{"role": "user", "content": f"Write Python code to solve this math problem. The code should follow the requirements.\\nProblem: {task}"}]
    code_response = agent.call_json_format_llm(
        messages=messages, 
        temperature=0.3, 
        num_of_response=1,
        agent_role="Python programmer", 
        return_dict_keys=["reasoning", "code"], 
        instructions=programmer_instructions.strip(),
    )[0]
    
    solution_code = code_response.get("code", "")
    try:
        result = agent.execute_code(solution_code)
        if isinstance(result, (int, float)):
            return_dict = {"answer": str(result), "code": solution_code}
        else:
            raise ValueError("Invalid solution code - result must be numeric")
    except Exception as e:
        return_dict = {
                "answer": str(0),
                "error": str(e)
            }
    return return_dict
"""
}

def get_prompt(current_archive, task_desp):
    archive_str = []
    for sol in current_archive[:-1]:
        sol_copy = sol.copy()
        if 'eval_feedback' in sol_copy:
            del sol_copy['eval_feedback']
        archive_str.append(json.dumps(sol_copy))
    archive_str.append(json.dumps(current_archive[-1]))
    archive_str = ",\n".join(archive_str)
    archive_str = f"[\n{archive_str}\n]"
    prompt = USER.replace("[ARCHIVE]", archive_str)
    prompt = prompt.replace("[TASK]", task_desp)
    return prompt