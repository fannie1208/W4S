import random
import csv
import argparse
import os
import pandas as pd
import numpy as np
import sys
import json
import copy
import matplotlib.pyplot as plt
import goal_prompt
from transformers import AutoTokenizer

tokenizer = None

def count_tokens(dialogue):
    global tokenizer
    prompt_token = 0
    response_token = 0
    for i in range(len(dialogue) - 1):
        message = dialogue[i]["value"]
        if not isinstance(message, str):
            print(message)
        tokens = tokenizer.tokenize(message)
        prompt_token += len(tokens)
    
    message = dialogue[-1]["value"]
    tokens = tokenizer.tokenize(message)
    response_token += len(tokens)

    return prompt_token, response_token


def init_conversation(query):
    conversation = [
        {
            "from": "human",
            "value": goal_prompt.system,
            "loss": 0.0,
        },
        {
            "from": "human",
            "value": query[query.find("### **Agentic System Interface**"):],
            "loss": 0.0,
        }
        # {
        #     "from": "human",
        #     "value": query,
        #     "loss": 0.0,
        # }
    ]
    return conversation

def create_user_message(observation):
    user_message = {
        "from": "human",
        "value": observation,
        "loss": 0.0
    }
    return user_message

def create_agent_message(action, reward):
    agent_message = {
        "from": "gpt",
        "value": action,
        "loss": reward
    }
    return agent_message


def skip_dialogue(only_success, log):
    max_reward = max(log["rewards"])
    if only_success:
        return max_reward == 0
    else:
        return False

def convert_to_data(log, args):
    query = log["query"]

    actions = log["actions"]
    rewards = log["rewards"]

    best_action = log['best_action']
    best_observation = log['best_observation']
    best_reward = log['best_reward']

    dataset = []
    conversations = init_conversation(query)

    if args.criteria == 'success':
        # Find last non-zero reward index
        last_success_idx = -1
        for i in range(len(best_reward)-1, -1, -1):
            if best_reward[i] != 0:
                last_success_idx = i
                best_action = best_action[:last_success_idx+1]
                best_observation = best_observation[:last_success_idx+1] 
                best_reward = best_reward[:last_success_idx+1]
                break
    
    for i in range(len(best_action)):
        for j in range(len(actions[i])):
            # Skip if args.filter is True and reward is -1
            if args.filter and rewards[i][j] == -1:
                continue
                
            conversation = copy.deepcopy(conversations)
            action = actions[i][j]
            # observation = observations[i][j]
            reward = rewards[i][j]
            conversation.append(create_agent_message(action, reward))
            
            current_turn = i + 1
            if current_turn in args.turns:
                dataset.append(conversation)
                
        conversations.append(create_agent_message(best_action[i], best_reward[i]))
        conversations.append(create_user_message(best_observation[i]))

    return dataset

def deep_copy_dataset(dataset):
    new_dataset = []
    for conversation in dataset:
        new_dataset.append(copy.deepcopy(conversation))
    return new_dataset

def exp_dataset_reward(dataset, temperature=1.0):
    for conversations in dataset:
        for i in range(len(conversations)):
            if conversations[i]['from'] == "gpt":
                conversations[i]["loss"] = np.exp(conversations[-1]["loss"]/temperature)
    return dataset

def plot_rewards_hist(rewards, save=None):
    plt.hist(rewards, bins=100)
    if save:
        plt.savefig(save, dpi=300)
    else:
        plt.show()

def get_rewards(dataset):
    rewards = []
    for conversations in dataset:
        rewards.append(conversations[-1]["loss"])
    return rewards

def conversations_to_dataset(dataset):
    new_dataset = []
    index = 0
    for conversation in dataset:
        interaction = {
            "id": f"identity_{index}",
            "conversations": conversation,
        }
        new_dataset.append(interaction)
        index += 1
    return new_dataset

def save_dataset(dataset, filename):
    rewards = get_rewards(dataset)
    plot_rewards_hist(rewards, save=f"{filename}_hist.png")
    fine_tune_dataset = conversations_to_dataset(dataset)
    with open(f"{filename}", "w") as f:
        json.dump(fine_tune_dataset, f, indent=4)

def data_analysis(dataset):
    prompt_tokens = []
    response_tokens = []
    for idx, dialogue in enumerate(dataset):
        prompt_token, response_token = count_tokens(dialogue)
        if prompt_token > 20000:
            print(f"Long prompt found at id: {idx}, token length: {prompt_token}")
        prompt_tokens.append(prompt_token)
        response_tokens.append(response_token)
    size = len(dataset)
    prompt_mean = np.mean(np.array(prompt_tokens))
    prompt_var = np.var(np.array(prompt_tokens))
    response_mean = np.mean(np.array(response_tokens))
    response_var = np.var(np.array(response_tokens))

    print(f"Size of dataset: {size}")
    print(f"Prompt Length: [{prompt_mean}, {prompt_var}, {np.max(np.array(prompt_tokens))}]")
    print(f"Response Length: [{response_mean}, {response_var}, {np.max(np.array(response_tokens))}]")
    print(f"====================================================")
    print(f"")

def main():
    global tokenizer
    parser = argparse.ArgumentParser(description='Convert dialogue into a datast for SFT')
    parser.add_argument('--file_dir', type=str, default='../datasets/generated', help='directory containing log files')
    parser.add_argument('--filename', type=str, nargs='+', help='names of log files')
    parser.add_argument('--output', type=str, help='path to save dataset')
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="path to load tokenizer")
    parser.add_argument('--temperature', type=float, default=0.4, help="temperature parameter used in reward mapping")
    parser.add_argument('--only_success', action='store_true', help="only keep success trajectory when activate")
    parser.add_argument('--criteria', choices=['all', 'success', 'best', 'better'], default="all", help="all: loss on all actions; success: loss on only the success actions; best: loss on the best actions among a trajectory; better: loss on actions that are better than history")
    parser.add_argument('--shuffle', action='store_true', help="shuffle the dataset")
    parser.add_argument('--turns', type=int, nargs='+', help="specify which turns of data to use (e.g. 1 2 for first and second turns)")
    parser.add_argument('--filter', action='store_true', help="whether to filter the dataset")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    dataset = []
    for filename in args.filename:
        with open(os.path.join(args.file_dir, filename)) as f:
            logs = json.load(f)
            for key in logs:
                log = logs[key]
                if skip_dialogue(args.only_success, log):
                    continue
                dataset += convert_to_data(log, args)
    
    data_analysis(dataset)
    if args.shuffle:
        random.shuffle(dataset)
    
    weighted_dataset = exp_dataset_reward(deep_copy_dataset(dataset), temperature=args.temperature)
    save_dataset(weighted_dataset, args.output)
    
if __name__ == "__main__":
    main()