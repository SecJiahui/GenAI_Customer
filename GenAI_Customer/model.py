import math
import random
import mesa
import networkx as nx
import numpy as np

import GenAI_Customer.agent


def update_customer_satisfaction(customer_agent, products_purchased, average_order_of_purchased_product):
    """Update customer satisfaction based on the number of products purchased."""
    if products_purchased > 1:
        customer_agent.satisfaction += 0.03
    elif products_purchased == 1:
        customer_agent.satisfaction += 0.01
    else:
        customer_agent.satisfaction -= 0.005

    # Additional logic based on average order of purchased products
    if average_order_of_purchased_product is not None:
        if average_order_of_purchased_product <= 5:
            # Positive feedback for lower average order values
            customer_agent.satisfaction += 0.1
        elif average_order_of_purchased_product <= 10:
            # Positive feedback for lower average order values
            customer_agent.satisfaction += 0.02
        else:
            # Negative feedback for higher average order values
            customer_agent.satisfaction -= 0.01

    customer_agent.satisfaction = max(0, min(1, customer_agent.satisfaction))


def update_ratings(product_agent, customer_agent):
    """Update ratings based on customer feedback."""
    if customer_agent.made_positive_comment():
        product_agent.retailer.rating += 0.05
    else:
        product_agent.retailer.rating += 0.015


def collect_keywords():
    # Implement logic to collect keywords (placeholder)
    return []


def get_product_by_id(product_id):
    # Implement logic to get product by ID (placeholder)
    return None


def collect_customer_feedback():
    # Implement logic to collect customer feedback (placeholder)
    return {}


class OnlinePlatformModel(mesa.Model):
    def __init__(self, num_customers, num_products, num_retailers):
        super().__init__()
        self.step_counter = 0
        self.used_ids = set()  # Track used IDs
        self.num_customers = num_customers
        self.num_products = num_products
        self.num_retailers = num_retailers
        self.total_sales = 0
        # probe test network
        prob = num_retailers / self.num_customers
        self.G = nx.erdos_renyi_graph(n=self.num_customers, p=prob)
        self.grid = mesa.space.NetworkGrid(self.G)

        self.schedule = mesa.time.RandomActivation(self)

        self.datacollector = mesa.DataCollector(
            {
                "LowSatisfaction": GenAI_Customer.agent.number_LowSatisfaction,
                "MediumSatisfaction": GenAI_Customer.agent.number_MediumSatisfaction,
                "HighSatisfaction": GenAI_Customer.agent.number_HighSatisfaction,
                "Sales": lambda model: model.total_sales,
                "Average Satisfaction": self.average_satisfaction,
            }
        )

        # Create customers
        for i in range(self.num_customers):
            unique_id = self.get_unique_id()
            customer = GenAI_Customer.agent.CustomerAgent(unique_id, self)
            self.schedule.add(customer)
            self.grid.place_agent(customer, i)

        # Create retailers
        for i in range(self.num_retailers):
            unique_id = self.get_unique_id()
            retailer = GenAI_Customer.agent.RetailerAgent(unique_id, self)
            self.schedule.add(retailer)

        # Create products
        for i in range(self.num_products):
            unique_id = self.get_unique_id()
            product = GenAI_Customer.agent.ProductAgent(unique_id, self)

            # Randomly select a retailer for the product
            retailer = random.choice(
                [agent for agent in self.schedule.agents if isinstance(agent, GenAI_Customer.agent.RetailerAgent)])
            product.retailer = retailer
            retailer.products.append(product)

            self.schedule.add(product)

        # Create Generative AI
        self.generative_ai = GenAI_Customer.agent.GenerativeAI()

        self.running = True
        self.datacollector.collect(self)

    def get_unique_id(self):
        # Generate a unique ID that has not been used before
        unique_id = None
        while unique_id is None or unique_id in self.used_ids:
            unique_id = self.random.randint(1, 1000000)  # Adjust the range as needed
        self.used_ids.add(unique_id)
        return unique_id

    def get_total_sales(self):
        return self.total_sales

    def get_product_agents(self):
        """Return a list of all product agents in the model."""
        return [agent for agent in self.schedule.agents if isinstance(agent, GenAI_Customer.agent.ProductAgent)]

    def get_customer_agents(self):
        """Return a list of all product agents in the model."""
        return [agent for agent in self.schedule.agents if isinstance(agent, GenAI_Customer.agent.CustomerAgent)]

    def average_satisfaction(self):
        """Calculate average customer satisfaction."""
        satisfaction_values = [customer_agent.satisfaction for customer_agent in self.schedule.agents
                               if isinstance(customer_agent, GenAI_Customer.agent.CustomerAgent)]
        return np.mean(satisfaction_values) if satisfaction_values else 0

    def process_customer_purchases(self, default_recommendations):
        """Process purchases for each customer."""
        # Get all product agents or use default recommendations if provided
        product_agents = default_recommendations if default_recommendations is not None else self.get_product_agents()
        customer_agents = self.get_customer_agents()

        for customer_agent in customer_agents:
            products_purchased = 0  # Track the amount of purchased products
            total_order_of_purchased_products = 0  # Sum of orders of all purchased products

            # Get personalized products for customers willing to share info
            # For others, use the default list of products
            products_to_consider = self.generative_ai.generate_personalized_recommendations(customer_agent,
                                                                                            product_agents) if customer_agent.willing_to_share_info else product_agents

            for index, product_agent in enumerate(products_to_consider):
                print(f"Customer {customer_agent.unique_id} is making a decision for product {product_agent}")
                decision = customer_agent.make_purchase_decision(product_agent)

                if decision == "Purchase":
                    products_purchased += 1
                    total_order_of_purchased_products += index
                    self.total_sales += product_agent.price
                    update_ratings(product_agent, customer_agent)

            # Calculate the average order of purchased products if any products were purchased
            if products_purchased > 0:
                average_order_of_purchased_product = total_order_of_purchased_products / products_purchased
            else:
                average_order_of_purchased_product = None

            # Update customer satisfaction
            update_customer_satisfaction(customer_agent, products_purchased, average_order_of_purchased_product)

    def step(self):
        # Increment step counter
        self.step_counter += 1
        # A: E-commerce platform updates sellers and product information

        # B: Generative AI generates basic recommendations

        product_agents = self.get_product_agents()
        recommendations = self.generative_ai.generate_basic_recommendations(product_agents)

        # C: Customers enter keywords to search for products
        # D: E-commerce platform pushes customer information and keywords to Gen AI
        # E: Generative AI provides hyper-personalized recommendations to customers
        # F: Customers purchase on e-commerce platforms and provide feedback
        # Simulate customers purchasing products and providing feedback
        if self.step_counter == 1:
            self.process_customer_purchases(default_recommendations=recommendations)
        else:
            self.process_customer_purchases(default_recommendations=None)

        # G: E-commerce platform updates platform information
        # Implement logic to update platform information

        # H: Generative AI learns from customer interactions, updating algorithms to improve future recommendations
        # self.generative_ai.learn_from_customer_interactions(customers_feedback)

        # Advance the model's time step
        self.schedule.step()

        # collect data
        self.datacollector.collect(self)
