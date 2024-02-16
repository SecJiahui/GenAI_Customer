import mesa

from GenAI_Customer.agent import State
from GenAI_Customer.model import OnlinePlatformModel


def customer_network_portrayal(G):
    def node_color(agent):
        return {State.LowSatisfaction: "#FF0000", State.MediumSatisfaction: "#808080"}.get(
            agent.state, "#008000"
        )

    def node_shape(agent):
        return "rect" if not agent.willing_to_share_info else "circle"

    def node_size(agent):
        return 3 if not agent.willing_to_share_info else 6

    def edge_color(source, target):
        """for target_agent_id, rating in source.review_history:
            if target_agent_id == target.unique_id:
                if rating >= 4:
                    return "#006400"  # green states for positive comments
                elif rating <= 1:
                    return "#8B0000"  # red states for negative comments"""
        return "#FFFFFF"

    def edge_width(agent1, agent2):
        if State.HighSatisfaction in (agent1.state, agent2.state):
            return 3
        return 0.2

    def get_agents(source, target):
        return G.nodes[source]["agent"][0], G.nodes[target]["agent"][0]

    portrayal = {}
    portrayal["nodes"] = [
        {
            "size": node_size(agents[0]),
            "shape": node_shape(agents[0]),
            "color": node_color(agents[0]),
            "tooltip": f"id: {agents[0].unique_id}<br>state: {agents[0].state.name}",
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

chart_satisfaction = mesa.visualization.ChartModule(
    [
        {"Label": "LowSatisfaction", "Color": "#FF0000"},
        {"Label": "MediumSatisfaction", "Color": "#808080"},
        {"Label": "HighSatisfaction", "Color": "#008000"},
    ]
)

chart_sales = mesa.visualization.ChartModule(
    [
        {"Label": "Sales", "Color": "#0000FF"},
        {"Label": "Sales (Willing)", "Color": "#008000"},
        {"Label": "Sales (Unwilling)", "Color": "#FF0000"},
    ]
)

chart_num_sold_products = mesa.visualization.ChartModule(
    [
        {"Label": "Sold Products", "Color": "#0000FF"},
        {"Label": "Sold Products (Willing)", "Color": "#008000"},
        {"Label": "Sold Products (Unwilling)", "Color": "#FF0000"},
    ]
)

# Add average customer satisfaction
chart_avg_satisfaction = mesa.visualization.ChartModule(
    [
        {"Label": "Average Satisfaction", "Color": "#808080"},
        {"Label": "Avg Satisfaction (Willing)", "Color": "#008000"},
        {"Label": "Avg Satisfaction (Unwilling)", "Color": "#FF0000"},
    ],
    data_collector_name='datacollector'
)

chart_avg_rating = mesa.visualization.ChartModule(
    [
        {"Label": "Average Seller Rating", "Color": "#808080"},
    ],
    data_collector_name='datacollector'
)

# Add average customer satisfaction
chart_mean_purchase_position = mesa.visualization.ChartModule(
    [
        {"Label": "mean_purchase_position", "Color": "#808080"},
        {"Label": "mean_purchase_position (Willing)", "Color": "#008000"},
        {"Label": "mean_purchase_position (Unwilling)", "Color": "#FF0000"},
    ],
    data_collector_name='datacollector'
)

chart_sharing_preferences = mesa.visualization.ChartModule(
    [
        {"Label": "Willing to Share Customers", "Color": "green"},
        {"Label": "Unwilling to Share Customers", "Color": "red"}
    ],
    data_collector_name='datacollector'
)

chart_number_products = mesa.visualization.ChartModule(
    [
        {"Label": "Number of Products", "Color": "blue"},
    ],
    data_collector_name='datacollector'
)

chart_AIC_willing = mesa.visualization.ChartModule(
    [
        {"Label": "AIC Linear (Willing)", "Color": "blue"},
        {"Label": "AIC Quadratic (Willing)", "Color": "red"},
        {"Label": "AIC Cubic (Willing)", "Color": "green"},
        {"Label": "AIC Quartic (Willing)", "Color": "yellow"},
        # {"Label": "AIC Hierarchical", "Color": "green"},
    ],
    data_collector_name='datacollector'
)

chart_AIC_unwilling = mesa.visualization.ChartModule(
    [
        {"Label": "AIC Linear (Unwilling)", "Color": "blue"},
        {"Label": "AIC Quadratic (Unwilling)", "Color": "red"},
        {"Label": "AIC Cubic (Unwilling)", "Color": "green"},
        {"Label": "AIC Quartic (Unwilling)", "Color": "yellow"},
        # {"Label": "AIC Hierarchical", "Color": "green"},
    ],
    data_collector_name='datacollector'
)

chart_AIC_sum = mesa.visualization.ChartModule(
    [
        {"Label": "AIC Linear (Sum)", "Color": "blue"},
        {"Label": "AIC Quadratic (Sum)", "Color": "red"},
        {"Label": "AIC Cubic (Sum)", "Color": "green"},
        {"Label": "AIC Quartic (Sum)", "Color": "yellow"},
        # {"Label": "AIC Hierarchical", "Color": "green"},
    ],
    data_collector_name='datacollector'
)

chart_AIC = mesa.visualization.ChartModule(
    [
        {"Label": "AIC Quartic (Willing)", "Color": "green"},
        {"Label": "AIC Quartic (Unwilling)", "Color": "yellow"},
    ],
    data_collector_name='datacollector'
)

model_params = {
    "num_customers": mesa.visualization.Slider("Number of Customers", 100, 20, 200, 5, 1),
    "percentage_willing_to_share_info": mesa.visualization.Slider("Percentage of Customer willing to share Info", 0.5, 0, 1.0, 0.1, 1),
    "num_products": mesa.visualization.Slider("Number of Products", 100, 30, 150, 10, 1),
    "num_retailers": mesa.visualization.Slider("Number of Retailers", 15, 5, 30, 2, 1),
    "learning_rate_gen_ai": mesa.visualization.Slider("Learning Rate of Gen AI", 0.3, 0.1, 0.9, 0.1, 1),
    "learning_rate_customer": mesa.visualization.Slider("Learning Rate of Customer", 0.3, 0.1, 0.9, 0.1, 1),
    "capacity_gen_ai": mesa.visualization.Slider("Capacity of Gen AI", 0.9, 0.1, 1, 0.1, 1),
    "creativity_gen_ai": mesa.visualization.Slider("Creativity of Gen AI", 0.8, 0.1, 1, 0.1, 1),
}

# Create Mesa server
server = mesa.visualization.ModularServer(
    OnlinePlatformModel,
    [network, chart_satisfaction, chart_avg_satisfaction, chart_avg_rating,
     chart_mean_purchase_position, chart_sales, chart_num_sold_products, chart_AIC_sum, chart_AIC],  # Add visualization modules
    # [network, chart_sales],  # Add visualization modules
    "Online Platform Model",
    model_params,
)

server.port = 8521
