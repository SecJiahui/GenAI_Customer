import math
import networkx as nx

import mesa

from GenAI_Customer.model import OnlinePlatformModel


def customer_network_portrayal(G):
    def node_color(agent):
        # Assuming the CustomerAgent has a 'state' attribute (replace 'state' with the actual attribute name)
        return {"Infected": "#FF0000", "Susceptible": "#008000"}.get(
            agent.state, "#808080"
        )

    def edge_color(agent1, agent2):
        # Assuming the CustomerAgent has a 'state' attribute (replace 'state' with the actual attribute name)
        if "Resistant" in (agent1.state, agent2.state):
            return "#000000"
        return "#e8e8e8"

    def edge_width(agent1, agent2):
        # Assuming the CustomerAgent has a 'state' attribute (replace 'state' with the actual attribute name)
        if "Resistant" in (agent1.state, agent2.state):
            return 3
        return 2

    def get_agents(source, target):
        return G.nodes[source]["agent"][0], G.nodes[target]["agent"][0]

    portrayal = {}
    portrayal["nodes"] = [
        {
            "size": 6,
            "color": node_color(agents[0]),
            "tooltip": f"id: {agents[0].unique_id}<br>state: {agents[0].state}",
        }
        for (_, agents) in G.nodes.data("agent")
    ]

    portrayal["edges"] = [
        {
            "source": source,
            "target": target,
            "color": edge_color(*get_agents(source, target)),
            "width": edge_width(*get_agents(source, target)),
        }
        for (source, target) in G.edges
    ]

    return portrayal


# Visualization modules
network = mesa.visualization.NetworkModule(
    customer_network_portrayal,
    500, 500,
)

chart_sales = mesa.visualization.ChartModule(
    [{"Label": "Sales", "Color": "#0000FF"}],
    data_collector_name="datacollector"
)
# Add more chart modules as needed

model_params = {
    "num_customers": mesa.visualization.Slider("Number of Customers", 50, 10, 100, 10, 1),
    "num_products": mesa.visualization.Slider("Number of Products", 20, 10, 100, 10, 1),
    "num_retailers": mesa.visualization.Slider("Number of Retailers", 10, 5, 30, 2, 1),
    # Add more parameters as needed
}

# Create Mesa server
server = mesa.visualization.ModularServer(
    OnlinePlatformModel,
    [network, chart_sales],  # Add visualization modules
    "Online Platform Model",
    model_params,
)

server.port = 8521