'''
This code follows the work and methods presented in:
https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute
'''

from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/MATH-500")

# implement best of N with PRM/ORM
# PRM - Process Reward Model, evaluates steps in a process, scores often aggregated
# ORM - Outcome Reward Model, evaluates outcomes of a process, answers extracted from \boxed{}

# RLHFlow/Llama3.1-8B-PRM-Deepseek-Data
# the model will serve as both PRM and ORM
# the model can be used as ORM by scores aggregation


# pseudocode
def answer(question, model):
    steps = model(question)
    return steps

def ORM_score(steps, model):
    scores = model(steps)
    score = last(scores) or product(scores) or min(scores)
    return score

def best_of_N(question, model, N):
    scores = []
    for i in range(N):
        steps = answer(question, model)
        score = ORM_score(steps, model)
        scores.append(score)
    return steps[max_arg(scores)]

def weighted_best_of_N(question, model, N):
    scores = {}
    for i in range(N):
        steps = answer(question, model)
        score = ORM_score(steps, model)
        # steps["answer"] is the answer extracted from \boxed{}
        scores[steps["answer"]].append(score)
    return steps[max_arg(scores)]

