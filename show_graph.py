import json
from pyvis.network import Network

if __name__ == '__main__':
    result = json.load(open('out/LOW_output.json'))

    # Create PyVis network
    net = Network(directed=True, notebook=True, height='100vh', width='100vw')

    # Add START node
    net.add_node('START', label='START', color='yellow', shape='square', size=20)

    # Add nodes and edges
    action_id = 0
    for attack_step_id, attack_step in enumerate(result):
        net.add_node(attack_step['id'], label=f"m{str(attack_step_id+1)}", title="STEP "+str(attack_step_id+1), color='blue', shape='circle', size=30)
        
        # Get previous attack step
        if attack_step_id > 0:
            prev_actions = result[attack_step_id - 1]['attack_steps']
            for action in prev_actions:
                net.add_node(action['id'], label=f"a{str(action_id+1)}", title=action['name'], color='red', shape='square', size=10)
                net.add_edge(action['id'], attack_step['id'], color='black')

                
        else:
            net.add_edge('START', result[0]['id'], color='black')
        
        for action in attack_step['attack_steps']:
            net.add_node(action['id'], label=f"a{str(action_id+1)}", title=action['name'], color='red', shape='square', size=10)
            net.add_edge(attack_step['id'], action['id'], color='black')

            action_id += 1


    # Add END node
    last_attack_step = result[-1]
    net.add_node('END', label='END', color='yellow', shape='square', size=20)
    for action_id, action in enumerate(last_attack_step['attack_steps']):
        net.add_edge(action['id'], 'END', color='black')
        
    # Save and display the network
    net.show('out/graph.html', notebook=False)
    net.save_graph