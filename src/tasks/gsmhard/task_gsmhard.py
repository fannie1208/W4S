import random
import time
from ..task_base import Task
from utils.helper import bootstrap_confidence_interval, wrap_solver
from tqdm import tqdm
from collections import namedtuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import re
import json
from .prompt import *

FILE_PATH = "../datasets/gsmhardv2.jsonl"

def get_all_examples(filepath=FILE_PATH):
    with open(filepath, mode="r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    for example in examples:
        example['inputs'] = "Solve this math problem:\n" + example['input']
        example['targets'] = example['target']
    return examples

def score_gsmhard(target: str, prediction: str) -> bool:
    try:
        target = float(target)
        prediction = float(prediction)
    except:
        return False

    if abs(target - prediction) < 1e-1:
        return True
    else:
        return False

def extract_answer_str(answer_str):
    # Define the regular expression pattern
    pattern = r'#### (-?\d+)'

    # Use re.search to find the match
    match = re.search(pattern, answer_str)

    if match:
        answer = match.group(1)
    else:
        raise AssertionError("No match found")

    return answer


class GSMHARD_Task(Task):
    def __init__(self, mode="test", threshold=0.7):
        super().__init__(threshold=threshold)
        self.mode = mode
        
    @property
    def task_description(self) -> str:
        return TASK_DESCRIPTION
        
    @property
    def initial_system(self) -> str:
        return [COT, COT_SC, Code_Generation]

    def evaluate(self, solver):
        examples = get_all_examples()
        random.seed(0)
        random.shuffle(examples)
        valid_set = examples[:128]
        valid_sample_set = examples[628: 828]
        train_set = examples[828:]
        private_valid_set = train_set[:60]
        public_valid_set = train_set[60:]
        random.seed(time.time())
        random.shuffle(public_valid_set)
        random.shuffle(valid_sample_set)
        if self.mode == 'test':
            examples = valid_set+valid_sample_set[:10]
        else:
            examples = private_valid_set+public_valid_set[:10]
        questions = [example['inputs'] for example in examples]
        answers = [example['targets'] for example in examples]
        max_workers = min(len(examples), 48)
        task_queue = []
        for q in questions:
            task_queue.append(q)
    
        acc_list = []
        info_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(solver, task_queue), total=len(task_queue)))

        for q_idx, res in enumerate(results):
            try:
                extracted_answer = str(res["answer"])
                correct_answer = str(answers[q_idx])
                correct = score_gsmhard(correct_answer, extracted_answer)
            except Exception as e:
                info_list.append(f"Valid Sample {q_idx}:\n{repr(e)}\nModel Output: {res}\n")
                acc_list.append(0)
                continue
            acc_list.append(correct)
            info_list.append(f"Valid Sample {q_idx}:\n{task_queue[q_idx]}\nModel Answer: {extracted_answer}\nCorrect Answer: {correct_answer}\nIs Correct: {acc_list[-1]}\n")

        valid_acc = sum(acc_list) / len(acc_list)
        median, interval = bootstrap_confidence_interval(acc_list)
        print("Acc:", valid_acc)
        if valid_acc >= self.threshold:
            test_acc, test_interval = self.test_evaluate(solver)
            feedback = f"Valid Accuracy: {valid_acc}\nTest Accuracy {test_acc}\n" + "Evaluation Info:\n" + "\n".join(random.sample(info_list, min(10, len(info_list))))
        else:
            test_acc = 0
            feedback = f"`Valid Accuracy: `{valid_acc}\n" + "Evaluation Info:\n" + "\n".join(random.sample(info_list, min(10, len(info_list))))
        return feedback, valid_acc, test_acc
    
    def test_evaluate(self, solver):
        examples = get_all_examples()
        random.seed(0)
        random.shuffle(examples)
        examples = examples[128:628]
        questions = [example['inputs'] for example in examples]
        answers = [example['targets'] for example in examples]
        max_workers = min(len(examples), 4)
        task_queue = []
        for q in questions:
            task_queue.append(q)

        acc_list = []
        info_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(wrap_solver(solver), task_queue), total=len(task_queue)))

        for q_idx, res in enumerate(results):
            try:
                extracted_answer = str(res["answer"])
                correct_answer = str(answers[q_idx])
                correct = score_gsmhard(correct_answer, extracted_answer)
            except Exception as e:
                info_list.append(f"Sample {q_idx}:\n{repr(e)}\nModel Output: {res}\n")
                acc_list.append(0)
                continue
            acc_list.append(correct)
            info_list.append(f"Sample {q_idx}:\n{task_queue[q_idx]}\nModel Output: {res}\nModel Answer: {extracted_answer}\nCorrect Answer: {correct_answer}\nIs Correct: {acc_list[-1]}\n\n")

        acc = sum(acc_list) / len(acc_list)
        median, interval = bootstrap_confidence_interval(acc_list)
        import os
        os.makedirs("../results", exist_ok=True)
        open(f"../results/gsmhard_{round(acc, 4)}.txt", "w").writelines([interval+'\n'] + info_list)
        return acc, interval
    
    def get_prompt(self, history):
        return SYSTEM, get_prompt(history, self.task_description)