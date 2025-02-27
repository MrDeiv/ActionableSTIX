import json
import evaluate
import numpy as np

if __name__ == "__main__":
    output = json.loads(open("out/output.json").read())

    # extract all the techniques from the output
    generated_techniques = []
    for attack_step in output:
        for action in attack_step["actions"]:
            generated_techniques.append(action['mitre_technique']['name'])


    ground_truth = [
        "Create or modify system process: windows service",
        "Masquerading: Match Legitimate Name or Location",
        "Virtualization/Sandbox Evasion: Time Based Evasion",
        "Virtualization/Sandbox Evasion: System Checks",
        "Virtualization/Sandbox Evasion: User Activity Based Checks",
        "Obfuscated Files or Information: Software Packing",
        "Deobfuscate/Decode Files or Information",
        "Hide Artifacts: Hidden Window",
        "Indicator Removal on Host: File Deletion",
        "Hijack Execution Flow: DLL Side-Loading",
        "Process Injection: Process Hollowing",
        "Signed Binary Proxy Execution: Rundll32",
        "System Information Discovery",
        "Application Layer Protocol: Web Protocols",
        "Fallback Channels",
        "Non-Standard Port"
    ]

    # evaluate the generated techniques
    bert_score = evaluate.load("bertscore")
    res = bert_score.compute(
        predictions=generated_techniques, 
        references=ground_truth, 
        lang="en")
    
    print("Precision:", np.mean(res['precision']))
    print("Recall:", np.mean(res['recall']))
    print("F1:", np.mean(res['f1']))
    