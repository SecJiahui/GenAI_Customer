from tqdm import tqdm
import itertools
import os
import random
import mesa
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import GenAI_Customer.agent

random.seed(42)
np.random.seed(42)


class OnlinePlatformModel(mesa.Model):
    def __init__(self, num_customers, percentage_willing_to_share_info, num_products, num_retailers,
                 learning_rate_gen_ai, learning_rate_customer, creativity_gen_ai, capacity_gen_ai, total_steps,
                 purchase_threshold):
        super().__init__()
        self.step_counter = 0
        self.total_steps = total_steps
        self.used_ids = set()  # Track used IDs
        self.num_customers = num_customers
        self.num_products = num_products
        self.num_retailers = num_retailers
        self.num_customers_willing_to_share_info = int(percentage_willing_to_share_info * num_customers)
        self.percentage_willing_to_share_info = percentage_willing_to_share_info
        self.total_sales = 0
        self.sales_willing = 0
        self.sales_unwilling = 0
        self.num_sold_products = 0
        self.num_sold_products_willing = 0
        self.num_sold_products_unwilling = 0
        self.purchase_threshold = purchase_threshold
        self.min_aic = 0

        self.learning_rate_gen_ai = learning_rate_gen_ai
        self.learning_rate_customer = learning_rate_customer
        self.capacity_gen_ai = capacity_gen_ai
        self.creativity_gen_ai = creativity_gen_ai

        self.G = nx.erdos_renyi_graph(n=self.num_customers, p=0.5)
        self.grid = mesa.space.NetworkGrid(self.G)

        self.schedule = mesa.time.RandomActivation(self)

        self.purchase_decisions = []

        """
        "AIC Linear (Willing)": lambda model: self.calculate_linear_aic(True),
        "AIC Quadratic (Willing)": lambda model: self.calculate_polynomial_aic(2, True),
        "AIC Cubic (Willing)": lambda model: self.calculate_polynomial_aic(3, True),
        "AIC Quartic (Willing)": lambda model: self.calculate_polynomial_aic(3, True),
        "AIC Linear (Unwilling)": lambda model: self.calculate_linear_aic(False),
        "AIC Quadratic (Unwilling)": lambda model: self.calculate_polynomial_aic(2, False),
        "AIC Cubic (Unwilling)": lambda model: self.calculate_polynomial_aic(3, False),
        "AIC Quartic (Unwilling)": lambda model: self.calculate_polynomial_aic(4, False),
        """

        self.datacollector = mesa.DataCollector(
            {
                "LowSatisfaction": GenAI_Customer.agent.number_LowSatisfaction,
                "MediumSatisfaction": GenAI_Customer.agent.number_MediumSatisfaction,
                "HighSatisfaction": GenAI_Customer.agent.number_HighSatisfaction,
                "Sales": lambda model: model.total_sales,
                "Sales (Willing)": lambda model: model.sales_willing,
                "Sales (Unwilling)": lambda model: model.sales_unwilling,
                "Sold Products": lambda model: model.num_sold_products,
                "Sold Products (Willing)": lambda model: model.num_sold_products_willing,
                "Sold Products (Unwilling)": lambda model: model.num_sold_products_unwilling,
                "Average Seller Rating": self.average_rating,
                "Average Satisfaction": self.average_satisfaction,
                "Avg Satisfaction (Willing)": self.average_satisfaction_willing_to_share,
                "Avg Satisfaction (Unwilling)": self.average_satisfaction_unwilling_to_share,
                "Willing to Share Customers": self.count_willing_to_share_customers,
                "Unwilling to Share Customers": self.count_unwilling_to_share_customers,
                "Number of Products": self.count_products,
                "mean_purchase_position": self.mean_purchase_position,
                "mean_purchase_position (Willing)": self.mean_purchase_position_willing_to_share,
                "mean_purchase_position (Unwilling)": self.mean_purchase_position_unwilling_to_share,
                "creativity_gen_ai": self.get_gen_ai_creativity,
                "AIC Linear (Sum)": lambda model: self.calculate_linear_aic(),
                "AIC Quadratic (Sum)": lambda model: self.calculate_polynomial_aic(2),
                "AIC Cubic (Sum)": lambda model: self.calculate_polynomial_aic(3),
                "AIC Quartic (Sum)": lambda model: self.calculate_polynomial_aic(4),
                "AIC Quartic (Minimum)": lambda model: model.min_aic,
                "AIC Quartic (Willing)": lambda model: self.calculate_polynomial_aic(4, True),
                "AIC Quartic (Unwilling)": lambda model: self.calculate_polynomial_aic(4, False),

                # "AIC Quadratic": self.calculate_quadratic_aic,
                # "AIC Cubic": self.calculate_cubic_aic,
                # "AIC Quartic": self.calculate_quartic_aic,
                # "AIC Hierarchical":self.calculate_hierarchical_aic,
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

        # Creat platform owner
        unique_id = self.get_unique_id()
        platform_owner = GenAI_Customer.agent.PlatformOwnerAgent(unique_id, self, self.learning_rate_gen_ai,
                                                                 self.capacity_gen_ai,
                                                                 self.creativity_gen_ai)
        self.schedule.add(platform_owner)

        # Create Generative AI
        self.generative_ai = GenAI_Customer.agent.GenerativeAI(self, platform_owner.learning_rate_gen_ai,
                                                               platform_owner.capacity_gen_ai,
                                                               platform_owner.creativity_gen_ai)

        self.running = True
        self.datacollector.collect(self)

    def get_unique_id(self):
        # Generate a unique ID that has not been used before
        unique_id = None
        while unique_id is None or unique_id in self.used_ids:
            unique_id = self.random.randint(1, 1000000)  # Adjust the range as needed
        self.used_ids.add(unique_id)
        return unique_id

    def get_gen_ai_creativity(self):
        return self.generative_ai.creativity

    def get_total_sales(self):
        return self.total_sales

    def get_product_agents(self):
        """Return a list of all product agents in the model."""
        return [agent for agent in self.schedule.agents if isinstance(agent, GenAI_Customer.agent.ProductAgent)]

    def get_customer_agents(self):
        """Return a list of all customer agents in the model."""
        return [agent for agent in self.schedule.agents if isinstance(agent, GenAI_Customer.agent.CustomerAgent)]

    def get_customer_agents_willing(self):
        """Return a list of all customer agents willing to share information in the model."""
        return [customer_agent for customer_agent in self.schedule.agents if
                isinstance(customer_agent, GenAI_Customer.agent.CustomerAgent) and
                customer_agent.willing_to_share_info]

    def get_customer_agents_unwilling(self):
        """Return a list of all customer agents willing to share information in the model."""
        return [customer_agent for customer_agent in self.schedule.agents if
                isinstance(customer_agent, GenAI_Customer.agent.CustomerAgent) and not
                customer_agent.willing_to_share_info]

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

    def average_rating(self):
        """Calculate average customer satisfaction."""
        rating_values = [seller_agent.rating for seller_agent in self.schedule.agents
                         if isinstance(seller_agent, GenAI_Customer.agent.SellerAgent)]
        return np.mean(rating_values) if rating_values else 0

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

    def print_all_product_parameters(self):
        """
        Print the parameters of all product agents in the model, each product in one line.
        """
        for agent in self.schedule.agents:
            if isinstance(agent, GenAI_Customer.agent.ProductAgent):
                product_info = (
                    f"Product ID: {agent.unique_id}, "
                    f"Price: {agent.price}, "
                    f"Quality: {agent.quality}, "
                    f"Keywords: {', '.join(agent.keywords)}, "
                    f"Content Score: {agent.content_score}, "
                    f"Brand: {', '.join(agent.brand)}, "
                    f"Sales Count: {agent.sales_count}"
                )
                print(product_info)

    def process_customer_purchases(self, default_content):
        """Process purchases for each customer."""
        # Get all product agents or use default recommendations if provided
        product_agents = default_content if default_content is not None else self.get_product_agents()
        # random.shuffle(product_agents)
        # Sort the product_agents list by the unique_id attribute of each ProductAgent
        sorted_product_agents = sorted(product_agents, key=lambda x: x.unique_id)

        # for product in product_agents:
        # update products price and discount randomly in each simulation
        # product.price = max(0, product.price + 0.5 * random.uniform(-1, 1))

        customer_agents = self.get_customer_agents()

        # Calculate the number of customers to receive AI-generated considerations based on capacity_gen_ai
        num_customers_gen_ai = int(len(customer_agents) * self.capacity_gen_ai * self.percentage_willing_to_share_info)

        # Randomly select a subset of customers for generative AI enhancements
        customers_gen_ai = random.sample(customer_agents, num_customers_gen_ai)

        self.total_sales = 0
        self.num_sold_products = 0
        self.sales_willing = 0
        self.sales_unwilling = 0
        self.num_sold_products_willing = 0
        self.num_sold_products_unwilling = 0

        for customer in customer_agents:
            products_purchased = 0  # Track the amount of purchased products
            total_purchase_position = 0  # Sum of orders of all purchased products
            number_content_matched = 0

            # Determine if this customer is selected for generative AI enhancements
            use_gen_ai = customer in customers_gen_ai

            products_to_consider = sorted_product_agents

            num_products_to_consider = int(len(products_to_consider) * 0.8)

            for index, product in enumerate(products_to_consider[:num_products_to_consider]):
                if products_purchased >= 6:
                    # If the consumer has purchased 6 items, stop purchasing
                    break

                # print(f"Customer {customer.unique_id} is making a decision for product {product.unique_id}:")
                decision = customer.make_purchase_decision(product, customer.willing_to_share_info and use_gen_ai,
                                                           self.creativity_gen_ai, self.learning_rate_gen_ai,
                                                           self.purchase_threshold)
                customer.make_comment(product)

                # Collect purchase decision data
                decision_info = {
                    "step": self.step_counter,
                    "customer_id": customer.unique_id,
                    "willing_to_share_info": int(customer.willing_to_share_info),
                    "satisfaction": customer.satisfaction,
                    'purchase_decision': decision['purchase_decision'],
                    'decision_factor': decision['decision_factor'],
                    'generative_ai_learning_rate': decision['generative_ai_learning_rate'],
                    'content_matched': decision['content_matched'],
                    "product_id": product.unique_id,
                    "product_price": product.price,
                    "product_quality": product.quality,
                    "product_content": product.content_score
                }
                self.purchase_decisions.append(decision_info)

                if decision['purchase_decision']:

                    if decision['content_matched']:
                        number_content_matched += 1

                    products_purchased += 1
                    total_purchase_position += index
                    self.total_sales += product.price
                    self.num_sold_products += 1
                    if customer.willing_to_share_info:
                        self.sales_willing += product.price
                        self.num_sold_products_willing += 1
                    else:
                        self.sales_unwilling += product.price
                        self.num_sold_products_unwilling += 1
                    update_seller_ratings_based_on_purchase(product, index, len(products_to_consider))

            # Calculate the average order of purchased products if any products were purchased
            if products_purchased > 0:
                customer.mean_purchase_position = total_purchase_position / products_purchased
            else:
                customer.mean_purchase_position = num_products_to_consider

            customer.num_content_matched = number_content_matched

            # Update customer satisfaction
            update_customer_satisfaction(customer, products_purchased, customer.mean_purchase_position,
                                         number_content_matched, len(products_to_consider))

        if random.random() < 0.5:
            self.generative_ai.creativity += (self.learning_rate_gen_ai * 0.8)
            self.generative_ai.creativity = max(0, min(self.generative_ai.creativity, 5))
        else:
            self.generative_ai.creativity -= (self.learning_rate_gen_ai * 0.1)
            self.generative_ai.creativity = max(0, min(self.generative_ai.creativity, 5))

    def export_purchase_decisions_to_csv(self, filename):
        """Exports purchase decision data to a CSV file."""
        df = pd.DataFrame(self.purchase_decisions)
        df.to_csv(filename, index=False)

    def export_combined_simulation_data_to_csv(self, filename):
        """
        Exports both model parameters and data collected by the DataCollector to a single CSV file.

        Parameters:
        filename - Name of the CSV file to which the combined data will be exported.
        """
        # Collect model parameters into a DataFrame
        model_parameters = {
            'num_customers': self.num_customers,
            'num_products': self.num_products,
            'num_retailers': self.num_retailers,
            'num_customers_willing_to_share_info': self.num_customers_willing_to_share_info,
            'learning_rate_gen_ai': self.learning_rate_gen_ai,
            'learning_rate_customer': self.learning_rate_customer,
            'capacity_gen_ai': self.capacity_gen_ai,
            'creativity_gen_ai': self.creativity_gen_ai,
            'total_steps': self.total_steps
        }
        model_parameters_df = pd.DataFrame([model_parameters])

        # Get the latest data collected by the DataCollector
        collected_data_df = self.datacollector.get_model_vars_dataframe().iloc[-1:]

        # Combine model parameters and the latest collected data into one DataFrame
        combined_df = pd.concat([model_parameters_df.reset_index(drop=True), collected_data_df.reset_index(drop=True)],
                                axis=1)
        # Export the combined DataFrame to a CSV file
        combined_df.to_csv(filename, index=False)

    def export_data_if_final_step(self):
        pass
        # if self.step_counter == self.total_steps:
        # self.export_purchase_decisions_to_csv("purchase_decisions.csv")
        # self.export_combined_simulation_data_to_csv("simulation.csv")

    def calculate_linear_aic(self, willing_to_share=None):
        """
        Calculate the AIC for a customer satisfaction model based on product attributes using a linear model.

        AIC = -2 ln(L) + 2k

        Parameters:
        willing_to_share: Boolean indicating whether to include only customers who are willing to share their info or not.

        Returns:
        AIC value
        """

        data = {
            'avg_price': [],
            'avg_quality': [],
            'avg_content': [],
            'willing_to_share': [],
            'satisfaction': []
        }
        if willing_to_share is None:
            customers = self.get_customer_agents()
        else:
            customers = self.get_customer_agents_willing() if willing_to_share else self.get_customer_agents_unwilling()

        for customer in customers:
            if customer.shopping_history:
                avg_price = np.mean([product.price for product in customer.shopping_history])
                avg_quality = np.mean([product.quality for product in customer.shopping_history])
                avg_content = np.mean([product.content_score for product in customer.shopping_history])
                satisfaction = customer.satisfaction
                willing_to_share_info = int(customer.willing_to_share_info)

                data['avg_price'].append(avg_price)
                data['avg_quality'].append(avg_quality)
                data['avg_content'].append(avg_content)
                data['willing_to_share'].append(willing_to_share_info)
                data['satisfaction'].append(satisfaction)

        df = pd.DataFrame(data)

        for col in ['avg_price', 'avg_quality', 'avg_content']:
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)

        if df.empty:
            return None

        # Checking and handling NaN values
        if df.isnull().values.any():
            df = df.dropna()  # or you can use df.fillna(value)

        X = sm.add_constant(df[['avg_price', 'avg_quality', 'avg_content', 'willing_to_share']])
        model = sm.OLS(df['satisfaction'], X).fit()

        """if willing_to_share is None:
            print(model.summary())"""

        return model.aic

    def calculate_polynomial_aic(self, max_degree, willing_to_share=None):
        """
        Calculate the AIC for a customer satisfaction model based on product attributes using a quadratic model.

        AIC = -2 ln(L) + 2k

        Returns:
        AIC value
        """
        data = {
            'avg_price': [],
            'avg_quality': [],
            'avg_content': [],
            'satisfaction': [],
            'willing_to_share': [],
            'customer_id': [],
            'mean_purchase_position': []
        }

        if willing_to_share is None:
            customers = self.get_customer_agents()
        else:
            customers = self.get_customer_agents_willing() if willing_to_share else self.get_customer_agents_unwilling()

        for customer in customers:
            if customer.shopping_history:
                avg_price = np.mean([product.price for product in customer.shopping_history])
                avg_quality = np.mean([product.quality for product in customer.shopping_history])
                avg_content = np.mean([product.content_score for product in customer.shopping_history])
                willing_to_share_info = int(customer.willing_to_share_info)
                satisfaction = customer.satisfaction

                data['avg_price'].append(avg_price)
                data['avg_quality'].append(avg_quality)
                data['avg_content'].append(avg_content)
                data['willing_to_share'].append(willing_to_share_info)
                data['satisfaction'].append(max(0, min(1, satisfaction)))
                data['customer_id'].append(customer.unique_id)
                # data['num_content_matched'].append(customer.num_content_matched)
                data['mean_purchase_position'].append(customer.mean_purchase_position)

        df = pd.DataFrame(data)

        if df.empty:
            # print("No data available for AIC calculation.")
            return None

        X = df[['avg_price', 'avg_quality', 'avg_content', 'mean_purchase_position']]
        for degree in range(2, max_degree + 1):
            X = np.column_stack((X, df[
                ['avg_price', 'avg_quality', 'avg_content', 'willing_to_share', 'mean_purchase_position']] ** degree))

        X = sm.add_constant(X)
        model = sm.OLS(df['satisfaction'], X).fit()

        aic = model.aic

        if aic < self.min_aic:
            self.min_aic = aic

        return aic

    def step(self):
        # Increment step counter
        self.step_counter += 1
        # A: E-commerce platform updates sellers and product information
        # self.print_all_product_parameters()

        # B: Generative AI generates basic content

        product_agents = self.get_product_agents()
        content = self.generative_ai.generate_basic_content(product_agents)

        # C: Customers enter keywords to search for products
        # D: E-commerce platform pushes customer information and keywords to Gen AI
        # E: Generative AI provides hyper-personalized content to customers
        # F: Customers purchase on e-commerce platforms and provide feedback
        # Simulate customers purchasing products and providing feedback
        if self.step_counter == 1:
            self.process_customer_purchases(default_content=content)
        else:
            self.process_customer_purchases(default_content=None)

        # G: E-commerce platform updates platform information
        product_agents = self.get_product_agents()
        for product in product_agents:
            update_seller_ratings_based_on_product(product)

        # H: Generative AI learns from customer interactions, updating algorithms to improve future content (done)

        # Advance the model's time step
        self.schedule.step()

        # collect data
        self.datacollector.collect(self)

        self.export_data_if_final_step()


def update_customer_satisfaction(customer_agent, products_purchased, average_order_of_purchased_product,
                                 num_content_matched, num_products):
    """
    Update the customer's satisfaction based on the number of products purchased and the average order of purchased products.

    The satisfaction level is adjusted upwards if the customer purchases more than one product or if the average order
    of the purchased products is low, indicating quick and efficient service. Conversely, the satisfaction decreases
    if no products are purchased or if the average order is high, indicating slow service.

    Parameters:
    customer_agent - The CustomerAgent instance.
    products_purchased - The number of products purchased by the customer.
    average_order_of_purchased_product - The average order position of the purchased products.

    The customer's satisfaction is bounded between 0 and 1.
    """
    # Update the customer's satisfaction based on the number of products purchased
    if products_purchased >= 5:
        customer_agent.satisfaction += 0.1
    elif products_purchased >= 3:
        customer_agent.satisfaction += 0.05
    elif products_purchased == 1:
        customer_agent.satisfaction += 0.03
    else:
        customer_agent.satisfaction -= 0.05

    # Update the customer's satisfaction based on the number of interest matched products
    if num_content_matched >= 5:
        customer_agent.satisfaction += 0.1
    elif num_content_matched >= 3:
        customer_agent.satisfaction += 0.05
    elif num_content_matched == 1:
        customer_agent.satisfaction += 0.03

    # Additional logic based on average order of purchased products
    if average_order_of_purchased_product is not None:
        if average_order_of_purchased_product <= num_products * 0.15:
            # Positive feedback for lower average order values
            customer_agent.satisfaction += 0.1
        elif average_order_of_purchased_product <= num_products * 0.3:
            # Positive feedback for lower average order values
            customer_agent.satisfaction += 0.05
        else:
            # Negative feedback for higher average order values
            customer_agent.satisfaction -= 0.03

    customer_agent.satisfaction = max(0, min(5, customer_agent.satisfaction))


def update_seller_ratings_based_on_purchase(product_agent, order_of_purchased_product, num_products):
    """
    Update the seller's rating based on customer feedback.

    This function adjusts the seller's rating based on the order position of the purchased product,
    indicating the timeliness and efficiency of the seller's service.

    Parameters:
    product_agent - The ProductAgent instance for the purchased product.
    customer_agent - The CustomerAgent instance making the purchase.
    order_of_purchased_product - The order position of the purchased product.

    The seller's rating is bounded between 0 and 1.
    """
    # Update seller's rating based on order_of_purchased_product
    if order_of_purchased_product is not None:
        if order_of_purchased_product <= num_products * 0.15:
            # Positive feedback for lower average order values
            product_agent.seller.rating += 0.05
        elif order_of_purchased_product <= num_products * 0.3:
            # Positive feedback for lower average order values
            product_agent.seller.rating += 0.03
        else:
            # Negative feedback for higher average order values
            product_agent.seller.rating -= 0.01

    # Make sure the seller's rating is between 0 and 5
    product_agent.seller.rating = max(0, min(product_agent.seller.rating, 5))


def update_seller_ratings_based_on_product(product_agent):
    """
    Update the seller's rating based on product's average rating.

    This function calculates the average customer rating for the product and further adjusts the seller's rating
    based on this average rating, reflecting the overall customer satisfaction with the product.

    The seller's rating is bounded between 0 and 1.
    """

    # Calculate the average rating of the product
    if product_agent.customers_comment:
        average_product_rating = sum(product_agent.customers_comment.values()) / len(product_agent.customers_comment)
    else:
        average_product_rating = 0

    # Update seller rating based on product's average rating
    if average_product_rating >= 4:
        product_agent.seller.rating += 0.03
    elif average_product_rating >= 3:
        product_agent.seller.rating += 0.02
    elif average_product_rating < 3:
        product_agent.seller.rating -= 0.01
    else:
        product_agent.seller.rating -= 0.02

    # Make sure the seller's rating is between 0 and 5
    product_agent.seller.rating = max(0, min(product_agent.seller.rating, 5))


def run_and_export_combined_data(model_class, params_ranges, export_filename):
    # Generate all combinations of parameter values
    param_names = sorted(params_ranges)
    combinations = list(itertools.product(*(params_ranges[name] for name in param_names)))

    # Create an empty DataFrame to store data from all simulations
    all_data_df = pd.DataFrame()

    # Iterate over each combination using tqdm for the progress bar
    for params_tuple in tqdm(combinations, desc="Running simulations"):
        params = dict(zip(param_names, params_tuple))

        # Initialize and run the model
        model = model_class(**params)
        for _ in range(model.total_steps):
            model.step()

        # Collect model parameters and the latest DataCollector data
        model_parameters_df = pd.DataFrame([params])
        collected_data_df = model.datacollector.get_model_vars_dataframe().iloc[-1:]

        # Merge the data
        combined_df = pd.concat([model_parameters_df.reset_index(drop=True), collected_data_df.reset_index(drop=True)], axis=1)

        # Append the combined data to the overall DataFrame
        all_data_df = pd.concat([all_data_df, combined_df], ignore_index=True)

    # Export the combined data to a CSV file
    all_data_df.to_csv(export_filename, index=False)


def run_simulation_with_params(model_class, params):
    """根据给定参数运行模型并返回结果DataFrame"""
    model = model_class(**params)
    for _ in range(model.total_steps):
        model.step()
    collected_data_df = model.datacollector.get_model_vars_dataframe().iloc[-1:]
    return collected_data_df


def compare_and_rerun_if_needed(model_class, params, all_data_df):
    """
    Compares results of two simulation runs with different 'percentage_willing_to_share_info' settings,
    and decides whether to rerun the simulations based on certain conditions.

    Parameters:
    - model_class: The simulation model class.
    - params: Dictionary of parameters used for the simulation.
    - all_data_df: DataFrame containing results from all simulations.

    Returns:
    - Updated DataFrame with new simulation results if conditions for rerunning are not met.
    """
    # Copy parameters and set 'percentage_willing_to_share_info' to 0 for the first run
    params_zero = params.copy()
    params_zero['capacity_gen_ai'] = 0
    result_zero = run_simulation_with_params(model_class, params_zero)

    # Copy parameters and set 'percentage_willing_to_share_info' to 1 for the second run
    params_one = params.copy()
    params_one['capacity_gen_ai'] = 1
    result_one = run_simulation_with_params(model_class, params_one)

    # Extract AIC Quartic (Sum) and Average Satisfaction values for both runs
    aic_quartic_sum_zero = result_zero['AIC Quartic (Minimum)'].values[0]
    aic_quartic_sum_one = result_one['AIC Quartic (Minimum)'].values[0]
    avg_satisfaction_zero = result_zero['Average Satisfaction'].values[0]
    avg_satisfaction_one = result_one['Average Satisfaction'].values[0]

    # List to store reasons for rerunning the simulations
    rerun_reasons = []
    # Check conditions and add reasons for rerunning if conditions are met
    if aic_quartic_sum_one >= aic_quartic_sum_zero:
        rerun_reasons.append("AIC Quartic for willing is not less than for unwilling.")
    if avg_satisfaction_one <= avg_satisfaction_zero:
        rerun_reasons.append("Average Satisfaction for willing is not greater than for unwilling.")
    if aic_quartic_sum_one < -3000:
        rerun_reasons.append("AIC Quartic for willing is less than -3000.")

    # Rerun simulations if any conditions are met
    if rerun_reasons:
        print("Rerunning simulation due to conditions not met:", "; ".join(rerun_reasons))
        return compare_and_rerun_if_needed(model_class, params, all_data_df)
    else:
        # Add results to all_data_df if no conditions for rerunning are met
        all_data_df = pd.concat([all_data_df, result_zero.assign(**params_zero)], ignore_index=True)
        all_data_df = pd.concat([all_data_df, result_one.assign(**params_one)], ignore_index=True)
    return all_data_df


def run_and_export_combined_data_test(model_class, params_ranges, export_filename):
    param_names = sorted([name for name in params_ranges if name != 'capacity_gen_ai'])
    combinations = list(itertools.product(*(params_ranges[name] for name in param_names)))
    all_data_df = pd.DataFrame()

    for params_tuple in tqdm(combinations, desc="Running simulations"):
        params = dict(zip(param_names, params_tuple))
        all_data_df = compare_and_rerun_if_needed(model_class, params, all_data_df)

    # 导出数据到CSV
    all_data_df.to_csv(export_filename, index=False)


# Specified parameter ranges

"""params_ranges = {
    'num_customers': [80, 100],
    'num_products': [60, 100],
    'num_retailers': [20],
    'learning_rate_gen_ai': [0.1, 0.5, 0.9],
    'learning_rate_customer': [0.3],
    'capacity_gen_ai': [0.1, 0.5, 0.9],
    'creativity_gen_ai': [0.1, 0.5, 0.9],
    'total_steps': [50],
    'percentage_willing_to_share_info': [0, 1],
    'purchase_threshold': [1.5]
    'learning_rate_gen_ai': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],

}"""

params_ranges = {
    'num_customers': [100],
    'num_products': [100],
    'num_retailers': [20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    'learning_rate_gen_ai': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'learning_rate_customer': [0.3],
    'capacity_gen_ai': [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
    'creativity_gen_ai': [0.1, 0.25, 0.4, 0.55, 0.7, 0.9],
    'total_steps': [40],
    'percentage_willing_to_share_info': [1],
    'purchase_threshold': [1.5]
}

"""params_ranges = {
    'num_customers': [100],
    'num_products': [100],
    'num_retailers': [20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    'learning_rate_gen_ai': [0, 0.4, 0.8],
    'learning_rate_customer': [0.3],
    'capacity_gen_ai': [0, 0.1, 0.5, 0.9],
    'creativity_gen_ai': [0.1, 0.5, 0.9],
    'total_steps': [40],
    'percentage_willing_to_share_info': [1],
    'purchase_threshold': [1.5]
}"""

"""params_ranges = {
    'num_customers': [50, 100, 150],
    'num_products': [60, 100, 150],
    'num_retailers': [10, 20, 30],
    'learning_rate_gen_ai': [0.4],
    'learning_rate_customer': [0.3],
    'capacity_gen_ai': [0, 1],
    'creativity_gen_ai': [0.4],
    'total_steps': [40, 40, 40, 40, 40],
    'percentage_willing_to_share_info': [1],
    'purchase_threshold': [1.5]
}"""

run_and_export_combined_data(OnlinePlatformModel, params_ranges, 'combined_simulation_data.csv')