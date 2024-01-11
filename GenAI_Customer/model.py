import random
import mesa
import networkx as nx
import numpy as np

import GenAI_Customer.agent


def update_customer_satisfaction(customer_agent, products_purchased, average_order_of_purchased_product):
    """Update customer satisfaction based on the number of products purchased."""
    if products_purchased > 1:
        customer_agent.satisfaction += 0.05
    elif products_purchased == 1:
        customer_agent.satisfaction += 0.03
    else:
        customer_agent.satisfaction -= 0.1

    # Additional logic based on average order of purchased products
    if average_order_of_purchased_product is not None:
        if average_order_of_purchased_product <= 10:
            # Positive feedback for lower average order values
            customer_agent.satisfaction += 0.1
        elif average_order_of_purchased_product <= 20:
            # Positive feedback for lower average order values
            customer_agent.satisfaction += 0.05
        else:
            # Negative feedback for higher average order values
            customer_agent.satisfaction -= 0.03

    customer_agent.satisfaction = max(0, min(1, customer_agent.satisfaction))


def update_ratings(product_agent, customer_agent):
    """Update ratings based on customer feedback."""
    if customer_agent.give_positive_comment:
        product_agent.seller.rating += 0.05
    else:
        product_agent.seller.rating -= 0.015


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
    def __init__(self, num_customers, num_products, num_retailers, num_customers_willing_to_share_info):
        super().__init__()
        self.step_counter = 0
        self.used_ids = set()  # Track used IDs
        self.num_customers = num_customers
        self.num_products = num_products
        self.num_retailers = num_retailers
        self.num_customers_willing_to_share_info = num_customers_willing_to_share_info
        self.total_sales = 0
        self.sales_willing = 0
        self.sales_unwilling = 0
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
                "Sales Willing": lambda model: model.sales_willing,
                "Sales Unwilling": lambda model: model.sales_unwilling,
                "Average Satisfaction": self.average_satisfaction,
                "Avg Satisfaction (Willing)": self.average_satisfaction_willing_to_share,
                "Avg Satisfaction (Unwilling)": self.average_satisfaction_unwilling_to_share,
                "Willing to Share Customers": self.count_willing_to_share_customers,
                "Unwilling to Share Customers": self.count_unwilling_to_share_customers,
                "Number of Products": self.count_products,
                "mean_purchase_position": self.mean_purchase_position,
                "mean_purchase_position (Willing)": self.mean_purchase_position_willing_to_share,
                "mean_purchase_position (Unwilling)": self.mean_purchase_position_unwilling_to_share,
            }
        )

        # Create customers
        for i in range(self.num_customers_willing_to_share_info):
            unique_id = self.get_unique_id()
            customer = GenAI_Customer.agent.CustomerAgent(unique_id, self, True)
            self.schedule.add(customer)
            self.grid.place_agent(customer, i)

        for i in range(self.num_customers_willing_to_share_info, self.num_customers):
            unique_id = self.get_unique_id()
            customer = GenAI_Customer.agent.CustomerAgent(unique_id, self, False)
            self.schedule.add(customer)
            self.grid.place_agent(customer, i)

        # Create retailers
        for i in range(self.num_retailers):
            unique_id = self.get_unique_id()
            retailer = GenAI_Customer.agent.SellerAgent(unique_id, self)
            self.schedule.add(retailer)

        # Create products
        for i in range(self.num_products):
            unique_id = self.get_unique_id()
            product = GenAI_Customer.agent.ProductAgent(unique_id, self)

            # Randomly select a retailer for the product
            retailer = random.choice(
                [agent for agent in self.schedule.agents if isinstance(agent, GenAI_Customer.agent.SellerAgent)])
            product.seller = retailer
            retailer.products.append(product)

            self.schedule.add(product)

        # Create Generative AI
        self.generative_ai = GenAI_Customer.agent.GenerativeAI(self)

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

    def average_satisfaction_willing_to_share(self):
        """Calculate average satisfaction for customers willing to share information."""
        satisfaction_values = [customer_agent.satisfaction for customer_agent in self.schedule.agents if
                               isinstance(customer_agent, GenAI_Customer.agent.CustomerAgent) and
                               customer_agent.willing_to_share_info]
        return np.mean(satisfaction_values) if satisfaction_values else 0

    def average_satisfaction_unwilling_to_share(self):
        """Calculate average satisfaction for customers unwilling to share information."""
        satisfaction_values = [customer_agent.satisfaction for customer_agent in self.schedule.agents
                               if isinstance(customer_agent, GenAI_Customer.agent.CustomerAgent) and not
                               customer_agent.willing_to_share_info]
        return np.mean(satisfaction_values) if satisfaction_values else 0

    def mean_purchase_position(self):
        """Calculate average purchase position for customers."""
        purchase_position = [agent.mean_purchase_position for agent in self.schedule.agents
                             if isinstance(agent,
                                           GenAI_Customer.agent.CustomerAgent) and agent.mean_purchase_position is not None]
        return np.mean(purchase_position) if purchase_position else 0

    def mean_purchase_position_willing_to_share(self):
        """Calculate average satisfaction for customers willing to share information."""
        purchase_position = [customer_agent.mean_purchase_position for customer_agent in self.schedule.agents if
                               isinstance(customer_agent, GenAI_Customer.agent.CustomerAgent) and
                               customer_agent.willing_to_share_info and customer_agent.mean_purchase_position is not None]
        return np.mean(purchase_position) if purchase_position else 0

    def mean_purchase_position_unwilling_to_share(self):
        """Calculate average satisfaction for customers willing to share information."""
        purchase_position = [customer_agent.mean_purchase_position for customer_agent in self.schedule.agents if
                               isinstance(customer_agent, GenAI_Customer.agent.CustomerAgent) and not
                               customer_agent.willing_to_share_info and customer_agent.mean_purchase_position is not None]
        return np.mean(purchase_position) if purchase_position else 0
    def count_willing_to_share_customers(self):
        """Count the number of customers willing to share their information."""
        return sum(1 for agent in self.schedule.agents
                   if isinstance(agent, GenAI_Customer.agent.CustomerAgent) and agent.willing_to_share_info)

    def count_unwilling_to_share_customers(self):
        """Count the number of customers willing to share their information."""
        return sum(1 for agent in self.schedule.agents
                   if isinstance(agent, GenAI_Customer.agent.CustomerAgent) and not agent.willing_to_share_info)

    def count_products(self):
        """Count the number of products."""
        return sum(1 for agent in self.schedule.agents
                   if isinstance(agent, GenAI_Customer.agent.ProductAgent))

    def process_customer_purchases(self, default_recommendations):
        """Process purchases for each customer."""
        # Get all product agents or use default recommendations if provided
        product_agents = default_recommendations if default_recommendations is not None else self.get_product_agents()
        random.shuffle(product_agents)

        for product in product_agents:
            # update products price and discount randomly in each simulation
            product.price = max(0, product.price + 5 * random.uniform(-1, 1))
            product.discount = max(0, product.discount + random.uniform(-1, 1))

        customer_agents = self.get_customer_agents()

        for customer_agent in customer_agents:
            products_purchased = 0  # Track the amount of purchased products
            total_purchase_position = 0  # Sum of orders of all purchased products

            # Get personalized products for customers willing to share info
            # For others, use the default list of products
            products_to_consider = self.generative_ai.generate_personalized_recommendations(self, customer_agent,
                                                                                            product_agents) if customer_agent.willing_to_share_info else product_agents

            for index, product_agent in enumerate(products_to_consider):
                print(f"Customer {customer_agent.unique_id} is making a decision for product {product_agent}")
                decision = customer_agent.make_purchase_decision(product_agent)

                if decision == "Purchase":
                    products_purchased += 1
                    total_purchase_position += index
                    self.total_sales += product_agent.price
                    if customer_agent.willing_to_share_info:
                        self.sales_willing += product_agent.price
                    else:
                        self.sales_unwilling += product_agent.price
                    update_ratings(product_agent, customer_agent)

            # Calculate the average order of purchased products if any products were purchased
            if products_purchased > 0:
                customer_agent.mean_purchase_position = total_purchase_position / products_purchased
            else:
                customer_agent.mean_purchase_position = None

            # Update customer satisfaction
            update_customer_satisfaction(customer_agent, products_purchased, customer_agent.mean_purchase_position)



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
