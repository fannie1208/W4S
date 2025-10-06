import os
import argparse
import json
from typing import Optional, Dict, List
from agent.meta_agent import MetaAgent
from tasks import MGSM_Task, GSMHARD_Task, MBPP_Task, HumanEval_Task
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SystemOptimizer:
    def __init__(self, 
                 model: str = "gpt-4o",
                 task = MGSM_Task,
                 max_iter: int = 30,
                 context_window: int = 5,
                 num_of_samples: int = 1,
                 debug_max: int = 3,
                 api_key: Optional[str] = None,
                 key_path: str = '../key.env',
                 messages_path = "test.json",
                 log_path = "data.jsonl",
                 system_path = 'system.jsonl',
                 reward_type: str = "discrete",
                 temperature: float = 0.5,
                 port: int = 8000,
                 use_direct_api: bool = True):
        """
        Args:
            config_path: 配置文件路径
            model_type: 'api' 或 'local'
            use_direct_api: Whether to use direct vLLM API instead of HTTP server
        """
        self.reward_type = reward_type
        self.task = task
        self.max_iter = max_iter
        self.window = context_window
        self.num_of_samples = num_of_samples
        self.messages_path = messages_path
        self.system_path = system_path
        self.log_path = log_path
        self.log_data = {}
        self.meta_agent = MetaAgent(
            model=model,
            task=self.task,
            api_key=api_key, 
            key_path=key_path,
            debug_max=debug_max,
            num_of_samples=num_of_samples,
            temperature=temperature,
            port=port,
            use_direct_api=use_direct_api
        )

        self.messages = []
        self.all_messages = []
    
    def reset_messages(self, eval_history):
        # history = eval_history[-self.window:]
        example_history = [item for item in eval_history if isinstance(item['iteration'], str) and 'Example' in item['iteration']]
        generated_history = [item for item in eval_history if isinstance(item['iteration'], int)]
        if not generated_history:
            history = example_history
        else:
            history = example_history + generated_history[-1:]
        # best_valid_acc = max([item['valid_acc'] for item in history])
        best_valid_acc = max(item['valid_acc'] for item in history)
        system_prompt, initial_prompt = self.task.get_prompt(history)
        self.messages = [{"role": "system", "content": system_prompt},{"role": "user", "content": initial_prompt}]
        return best_valid_acc, initial_prompt
    
    def _construct_feedback(self, eval_results: Dict, prev_valid_acc: Optional[float] = None) -> str:
        """Construct feedback message from evaluation results,
           including information on whether the validation accuracy has increased,
           decreased, or remained unchanged compared to the previous iteration."""
        current_valid_acc = eval_results.get('valid_acc')
        change_message = ""
        if current_valid_acc is not None and prev_valid_acc is not None:
            if current_valid_acc > prev_valid_acc:
                change_message = f"Validation accuracy increased from {prev_valid_acc:.2f} to {current_valid_acc:.2f}."
            elif current_valid_acc < prev_valid_acc:
                change_message = f"Validation accuracy decreased from {prev_valid_acc:.2f} to {current_valid_acc:.2f}."
            else:
                change_message = f"Validation accuracy remained unchanged at {current_valid_acc:.2f}."
        
        return f"""The evaluation feedback of your proposed agentic system is:

{eval_results['eval_feedback']}

{change_message}

Please improve the agentic system to increase validation performance.
"""

    def optimize_system(self) -> str:
        """
        # Optimize the system through iterative refinement.
        # Returns:
        #     The optimized system code.
        """
        eval_history: List[Dict] = []
        best_acc = 0
        current_system = ""
        best_system = ""
        example_systems = self.task.initial_system
        # Save system history to jsonl file
        # First check if system history file exists
        if os.path.exists(self.system_path):
            # Load existing history entries
            with open(self.system_path, 'r') as f:
                for line in f:
                    history_entry = json.loads(line)
                    eval_history.append(history_entry)
            
            # Check if we need to evaluate additional example systems
            existing_examples = set(entry['iteration'] for entry in eval_history 
                                 if isinstance(entry['iteration'], str) and 'Example' in entry['iteration'])
            
            # Remove already evaluated examples from example_systems
            example_systems = [
                system for system in example_systems 
                if f'Example {system["name"]} system' not in existing_examples
            ]
        # Evaluate initial example systems
        for i, system in enumerate(example_systems):
            feedback, valid_acc, test_acc = self.meta_agent.evaluate_on_task(system['code'])
            history_entry = {
                'iteration': f'Example {system["name"]} system',
                'valid_acc': valid_acc,
                'system_code': system['code'],
                'eval_feedback': feedback,
            }
            eval_history.append(history_entry)
            # Create and append to new jsonl file
            os.makedirs(os.path.dirname(self.system_path), exist_ok=True)
            with open(self.system_path, 'a') as f:
                f.write(json.dumps(history_entry) + '\n')
        # Find system with highest validation accuracy from history
        best_entry = max(eval_history, key=lambda x: x['valid_acc'])
        best_system = best_entry['system_code']
        current_system = best_entry['system_code']
        best_acc = best_entry['valid_acc']
        
        best_valid_acc, initial_prompt = self.reset_messages(eval_history)

        self.all_messages = self.messages
        resume_iter = eval_history[-1]['iteration'] if isinstance(eval_history[-1]['iteration'],int) else 0
        for i in range(resume_iter, self.max_iter, self.window):
            best_valid_acc, initial_prompt = self.reset_messages(eval_history)
            # query = self.get_truncated_prompt(eval_history)
            log_episode = {
                "task_idx": i // self.window,
                "query": initial_prompt,
                'actions': [],
                'observations': [],
                'rewards': [],
                'best_action': [],
                'best_observation': [],
                'best_reward': []
            }
            for j in range(i, i + self.window):
                if j != 0:
                    self.meta_agent.num_of_samples = self.num_of_samples
                print(f"Iteration {j+1}/{self.max_iter}")

                retry_count = 0
                while retry_count < 3:
                    model_outputs, new_systems, eval_feedbacks, valid_accs, test_accs, best_result = self.meta_agent.improve_system(
                        messages=self.messages
                    )
                    if len(model_outputs) > 0 and len(new_systems) > 0 and len(eval_feedbacks) > 0:
                        break
                    print(f"Empty results, retrying iteration... (attempt {retry_count + 1}/3)")
                    retry_count += 1
                
                if retry_count == 3:
                    print("Failed to get valid results after 3 attempts. Terminating program.")
                    return best_system

                rewards = []
                for valid_acc_candidate in valid_accs:
                    if self.reward_type == "continuous":
                        reward = (valid_acc_candidate - best_valid_acc) / best_valid_acc if best_valid_acc > 0 else 1
                    else:  # discrete
                        if valid_acc_candidate > best_valid_acc:
                            reward = 1
                        elif (all(isinstance(item['iteration'], str) 
                                for item in eval_history) 
                              and valid_acc_candidate > eval_history[0]['valid_acc']) \
                             or valid_acc_candidate > eval_history[-1]['valid_acc']:
                            reward = 0.5
                        else:
                            reward = 0
                        if valid_acc_candidate < 0.3:
                            reward -= 0.1
                    rewards.append(reward)
                
                # Update evaluation history with best result
                eval_results = {
                    'iteration': j + 1,
                    'valid_acc': best_result['valid_acc'],
                    'system_code': best_result['system_code'],
                    'eval_feedback': best_result['eval_feedback'],
                }

                observations = [
                    self._construct_feedback({'eval_feedback': ef, 'valid_acc': valid_accs[idx]}, best_valid_acc)
                    for idx, ef in enumerate(eval_feedbacks)
                ]
                best_observation = self._construct_feedback(eval_results, best_valid_acc)

                log_episode['rewards'].append(rewards)
                log_episode['actions'].append(model_outputs)
                log_episode['observations'].append(observations)
                log_episode['best_action'].append(best_result['model_output'])
                log_episode['best_observation'].append(best_observation)
                log_episode['best_reward'].append(max(rewards))

                self.messages.extend([
                    {"role": "assistant", "content": best_result['model_output']},
                    {"role": "user", "content": best_observation}
                ])
                self.all_messages.extend([
                    {"role": "assistant", "content": best_result['model_output']},
                    {"role": "user", "content": best_observation}
                ])
            
                with open(self.messages_path, 'w', encoding='utf-8') as f:
                    json.dump(self.all_messages, f, ensure_ascii=False, indent=2)
                logger.info(f"Messages saved to {self.messages_path}")

                # Save best system entry to system history file
                with open(self.system_path, 'a') as f:
                    f.write(json.dumps(eval_results) + '\n')

                # if best_result['valid_acc'] > best_valid_acc:
                #     best_valid_acc = best_result['valid_acc']
                
                if best_result['test_acc'] > best_acc:
                    best_acc = best_result['test_acc']
                    best_system = best_result['system_code']
                    logger.info(f"New best system found! Test accuracy: {best_result['test_acc']}")
                
                eval_history.append(eval_results)
                current_system = best_result['system_code']

            self.log_data[i // self.window] = log_episode
            with open(self.log_path, "w") as fp:
                json.dump(self.log_data, fp, indent=2)
        
        logger.info(f"Optimization completed. Best test accuracy: {best_acc}")
        return best_system

def main(args: argparse.Namespace) -> None:
    if not os.path.exists('../results'):
        os.makedirs('../results')
    file_name = f"{args.task}_" + args.output
    messages_path = os.path.join('../results', file_name)
    data_name = f"{args.task}_{args.context_window}_turns_{args.data_output}.json"
    system_name = os.path.join('../systems', f'{args.task}_{args.context_window}_turns_{args.data_output}_archive.jsonl')
    log_path = os.path.join(args.log_dir, data_name)
    task_map = {
        'mgsm': MGSM_Task,
        'gsmhard': GSMHARD_Task,
        'mbpp': MBPP_Task,
        'humaneval': HumanEval_Task
    }
    
    task = task_map.get(args.task)
    if not task:
        raise ValueError(f"Unknown task: {args.task}")
    
    optimizer = SystemOptimizer(
        model=args.model, 
        task=task(),
        max_iter=args.iterations,
        context_window=args.context_window,
        num_of_samples=args.num_samples,
        messages_path=messages_path,
        system_path=system_name,
        log_path=log_path,
        reward_type=args.reward_type,
        temperature=args.temperature,
        port=args.port,
        use_direct_api=args.use_direct_api
    )
    
    optimized_system = optimizer.optimize_system()
    logger.info("Optimization completed. Final system:")
    print(optimized_system)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run system optimization for agent tuning')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='Name or path of the model to use (e.g. gpt-4o, gpt-3.5-turbo)')
    parser.add_argument('--iterations', type=int, default=30,
                        help='Number of optimization iterations')
    parser.add_argument('--task', type=str, default='mgsm', choices=['mgsm', 'drop', 'mmlu', 'gpqa', 'math','gsm8k','gsmhard','gsmplus','mmlupro','svamp','aime'],
                        help='Task type to optimize for')
    parser.add_argument('--context_window', type=int, default=2,
                        help='Number of history iterations')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to generate per iteration')
    parser.add_argument('--output', type=str, default='test_9.json',
                        help='Output json filename for messages')
    parser.add_argument('--data_output', type=str, default='2_1',
                        help='Output json filename for data')
    parser.add_argument('--log_dir', type=str, default="../datasets/generated", 
                        help='folder to save experiment run log file to')
    parser.add_argument('--reward_type', type=str, default='discrete', choices=['discrete', 'continuous'],
                        help='Type of reward calculation (discrete or continuous)')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature parameter for model sampling')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port number for VLLM server')
    parser.add_argument('--use_direct_api', action='store_true', default=False,
                        help='Whether to use direct vLLM API instead of HTTP server')
    args = parser.parse_args()
    main(args)