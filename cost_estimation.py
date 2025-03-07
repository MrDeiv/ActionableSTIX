import json
import math

if __name__ == '__main__':

    models = {
        "gpt-4-32k": {
            "input": 60,
            "output": 120
        },
        "gpt-4o": {
            "input": 2.5,
            "output": 10
        },
        "o1": {
            "input": 15,
            "output": 60
        },
        "o1-mini": {
            "input": 3,
            "output": 12
        },
        "gemini-1.5-flash-8B": {
            "input": 0.04,
            "output": 0.15
        },
    }

    output = json.load(open('out/output.json'))

    word_count = 0
    for milestone in output:
        for action in milestone['attack_steps']:
            word_count += len(action['description'].split())
            for indicator in action['indicators']:
                word_count += len(indicator.split())
            
        for pre in milestone['pre-conditions']:
            word_count += len(pre.split())
        
        for post in milestone['post-conditions']:
            word_count += len(post.split())

    tokens = math.ceil((word_count / 75))/10000

    print(f"Total word count: {word_count}")
    print(f"Total tokens: {tokens}")

    output_costs = []
    for model in models:
        output_costs.append({
            "model": model,
            "cost": models[model]['output']*tokens
        })

    print(output_costs)
