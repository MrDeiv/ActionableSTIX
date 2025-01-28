import json
from pyvis.network import Network

# load attack file
attack = json.loads(open("demo.json").read())

# create a network
g = Network(
    height="100vh",
    width="100vw",
    directed=True,
    notebook=False,
    bgcolor="white",
    heading="Malware",
    select_menu=True)

# add nodes
# each kill chain phase is a circular node
tactics = list(attack.keys())
for id, tactic in enumerate(tactics):
    g.add_node(tactic, 
               shape="circle", 
               label="State " + str(id),
               title=attack[tactics[id]]["title"],
               group=1,
               size=30)

    # if exists, connect to previous attack patterns
    if id > 0:
        for ap in attack[tactics[id-1]]["actions"]:
            g.add_edge(ap["name"], tactic, color="black")

    for i,ap in enumerate(attack[tactics[id]]["actions"]):
        # attack patterns are square nodes
        g.add_node(
            n_id=ap["name"],
            title=ap["name"],
            label=str(i+1),
            shape="square", group=2, size=10)
        g.add_edge(tactic, ap["name"], color="black")

        iocs = ap["indicators"].split(",")
        for j, ioc in enumerate(iocs):
            g.add_node(
                n_id=ioc,
                title=ioc,
                label=str(j+1),
                shape="triangle", group=3, size=5)
            g.add_edge(ioc, ap["name"], color="black")


# show graph
g.show("attack_graph.html", notebook=False)

