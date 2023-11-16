import math
import random
import mesa
import networkx as nx

import GenAI_Customer.agent


class OnlinePlatformModel(mesa.Model):
    def __init__(self, num_customers, num_products, num_retailers):
        super().__init__()
        self.used_ids = set()  # Track used IDs
        self.num_customers = num_customers
        self.num_products = num_products
        self.num_retailers = num_retailers
        self.total_sales = 0
        # probe test net
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
            if isinstance(customer_agent, GenAI_Customer.agent.CustomerAgent):
                products_purchased = 0  # track the amount of purchased products
                #  Make a purchase decision for each product
                for product_agent in self.schedule.agents:
                    if isinstance(product_agent, GenAI_Customer.agent.ProductAgent):
                        print(f"Customer {customer_agent.unique_id} with satisfaction {customer_agent.satisfaction} is making ")
                        decision = customer_agent.make_purchase_decision(product_agent)
                        print(f"Done")

                        if decision == "Purchase":
                            products_purchased += 1
                            self.total_sales += product_agent.price

                            if customer_agent.made_positive_comment():
                                product_agent.retailer.rating += 0.05
                            else:
                                product_agent.retailer.rating += 0.025

                # Update satisfaction based on the purchase decision
                if products_purchased > 1:
                    customer_agent.satisfaction += 0.05  # purchased more than one item
                elif products_purchased == 1:
                    customer_agent.satisfaction += 0.025  # purchased only one item
                else:
                    customer_agent.satisfaction -= 0.01  # none item was purchased

                # Ensure satisfaction stays within the [0, 1] range
                customer_agent.satisfaction = max(0, min(1, customer_agent.satisfaction))

        # G: E-commerce platform updates platform information
        # Implement logic to update platform information

        # H: Generative AI learns from customer interactions, updating algorithms to improve future recommendations
        # self.generative_ai.learn_from_customer_interactions(customers_feedback)

        # Advance the model's time step
        self.schedule.step()

        # collect data
        self.datacollector.collect(self)
