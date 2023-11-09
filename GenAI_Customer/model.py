import math
import random
import mesa
import networkx as nx

from GenAI_Customer.agent import CustomerAgent, ProductAgent, RetailerAgent, GenerativeAI


class OnlinePlatformModel(mesa.Model):
    def __init__(self, num_customers, num_products, num_retailers):
        super().__init__()
        self.used_ids = set()  # Track used IDs
        self.num_customers = num_customers
        self.num_products = num_products
        self.num_retailers = num_retailers

        self.schedule = mesa.time.RandomActivation(self)

        # Create customers
        for i in range(self.num_customers):
            unique_id = self.get_unique_id()
            customer = CustomerAgent(unique_id, self)
            self.schedule.add(customer)

        # Create retailers
        for i in range(self.num_retailers):
            unique_id = self.get_unique_id()
            retailer = RetailerAgent(unique_id, self)
            self.schedule.add(retailer)

        # Create products
        for i in range(self.num_products):
            unique_id = self.get_unique_id()
            product = ProductAgent(unique_id, self, price=random.uniform(1, 100),
                                   quality=random.uniform(0, 1),
                                   discount=random.uniform(0, 0.5),
                                   keywords=["keyword1", "keyword2"],
                                   brand=["brand1"])

            # Randomly select a retailer for the product
            retailer = random.choice([agent for agent in self.schedule.agents if isinstance(agent, RetailerAgent)])
            product.retailer = retailer
            retailer.products.append(product)

            self.schedule.add(product)

        # Create Generative AI
        self.generative_ai = GenerativeAI()

    def get_unique_id(self):
        # Generate a unique ID that has not been used before
        unique_id = None
        while unique_id is None or unique_id in self.used_ids:
            unique_id = self.random.randint(1, 1000000)  # Adjust the range as needed
        self.used_ids.add(unique_id)
        return unique_id

    def step(self):
        # A: E-commerce platform updates sellers and product information
        # Implement logic to update sellers and product information

        # B: Generative AI generates basic recommendations
        # recommendations = self.generative_ai.generate_recommendations()

        # C: Customers enter keywords to search for products
        # Simulate customers searching for products based on keywords

        # D: E-commerce platform pushes customer information and keywords to Gen AI
        # self.generative_ai.receive_customer_info(customers_info, keywords)

        # E: Generative AI provides hyper-personalized recommendations to customers
        # self.generative_ai.provide_personalized_recommendations(customers)

        # F: Customers purchase on e-commerce platforms and provide feedback
        # Simulate customers purchasing products and providing feedback
        for customer_agent in self.schedule.agents:
            if isinstance(customer_agent, CustomerAgent):
                # Iterate over all ProductAgent instances
                for product_agent in self.schedule.agents:
                    if isinstance(product_agent, ProductAgent):
                        # Make a purchase decision for each product
                        decision = customer_agent.make_purchase_decision(product_agent)

                        # Update satisfaction based on the purchase decision
                        if decision == "Purchase":
                            customer_agent.satisfaction += 0.1
                            # Check if the customer made a positive comment
                            if customer_agent.made_positive_comment():
                                # Seller rating update based on successful purchase and positive comment
                                product_agent.retailer.rating += 0.1
                            else:
                                # Seller rating update based on successful purchase but no comment
                                product_agent.retailer.rating += 0.05
                        else:
                            customer_agent.satisfaction -= 0.1

                        # Ensure satisfaction stays within the [0, 1] range
                        customer_agent.satisfaction = max(0, min(1, customer_agent.satisfaction))

        # G: E-commerce platform updates platform information
        # Implement logic to update platform information

        # H: Generative AI learns from customer interactions, updating algorithms to improve future recommendations
        # self.generative_ai.learn_from_customer_interactions(customers_feedback)

        # Advance the model's time step
        self.schedule.step()


# Example usage:
model = OnlinePlatformModel(num_customers=100, num_products=10, num_retailers=5)
for i in range(10):
    model.step()
