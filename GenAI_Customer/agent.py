import mesa
import random
from enum import Enum

import numpy as np


class State(Enum):
    LowSatisfaction = 0
    MediumSatisfaction = 1
    HighSatisfaction = 2


def number_state(model, state):
    return sum(1 for a in model.grid.get_all_cell_contents() if a.state is state)


def number_LowSatisfaction(model):
    return number_state(model, State.LowSatisfaction)


def number_MediumSatisfaction(model):
    return number_state(model, State.MediumSatisfaction)


def number_HighSatisfaction(model):
    return number_state(model, State.HighSatisfaction)


def initialize_interests():
    potential_interests = [
        "Sports", "Technology", "Fashion", "Travel", "Music",
        "Cooking", "Art", "Literature", "Cinema", "Gaming",
        "Gardening", "Photography", "Health and Fitness", "History", "Science",
        "Nature", "DIY Crafts", "Astronomy", "Politics", "Dance"
    ]
    return random.sample(potential_interests, k=2)


def initialize_keyword():
    potential_keyword = [
        "Sports", "Technology", "Fashion", "Travel", "Music",
        "Cooking", "Art", "Literature", "Cinema", "Gaming",
        "Gardening", "Photography", "Health and Fitness", "History", "Science",
        "Nature", "DIY Crafts", "Astronomy", "Politics", "Dance"
    ]
    return random.sample(potential_keyword, k=random.randint(1, len(potential_keyword)))


def initialize_brand():
    potential_brand = ["A", "B", "C", "D", "E"]
    return random.sample(potential_brand, 1)


class CustomerAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.shopping_history = []
        self.shopping_amount = 0
        self.interests = initialize_interests()  # initialize interest
        self.willing_to_share_info = random.random() < 0.3  # 30% customer agree to share their information
        self.satisfaction = random.uniform(0.3, 0.7)
        self.purchase_threshold = np.random.normal(0.5, 0.1)  # Normal distribution
        self.price_sensitivity = np.random.beta(2, 5)  # Beta distribution, tends to higher price sensitivity
        self.quality_sensitivity = np.random.beta(5, 2)  # Beta distribution, tends to lower quality sensitivity
        self.discount_sensitivity = np.random.beta(2, 5)  # Beta distribution, tends to higher price sensitivity
        self.brand_loyalty = np.random.beta(2, 2)  # Beta distribution, balanced brand loyalty
        self.state = self.get_satisfaction_level()

    def get_satisfaction_level(self):
        if self.satisfaction >= 0.8:
            return State.HighSatisfaction
        elif 0.4 <= self.satisfaction < 0.8:
            return State.MediumSatisfaction
        else:
            return State.LowSatisfaction

    def update_satisfaction_level(self):
        self.state = self.get_satisfaction_level()

    def make_purchase_decision(self, product):
        # Extract product attributes
        product_price = product.price
        product_quality = product.quality
        product_discount = product.discount
        product_keywords = product.keywords
        brand = product.brand

        purchase_decision = False

        # Calculate weighted factors
        # negative influence for higher price_sensitivity
        price_factor = (1 - self.price_sensitivity) * (1 - product_price / 10)
        # positive influence for higher quality_sensitivity
        quality_factor = self.quality_sensitivity * product_quality
        # positive influence for higher discount_sensitivity
        discount_factor = self.discount_sensitivity * product_discount

        decision_factor = price_factor + quality_factor + discount_factor

        # print Decision Factors information
        print(
            f'Decision Factors for product: {product.unique_id}, '
            f'with Price Factor: {price_factor}, Quality Factor: {quality_factor}，Discount Factor: {discount_factor}'
        )

        # Increase decision factor if product keywords match customer interests
        for interest in self.interests:
            if interest in product_keywords:
                print(f"Customer is interested in this product")
                decision_factor += 0.1

        # Increase decision factor if product brand matches customer shopping list
        if brand in self.shopping_history:
            decision_factor += (0.1 * self.brand_loyalty)
            print(f"Brand: {brand} has been purchased before.")

        print(f"Total: {decision_factor}")

        # Make a purchase decision based on decision factor and threshold
        purchase_threshold = 1.4
        if decision_factor > purchase_threshold:
            purchase_decision = True

        if purchase_decision:
            product.sales_count += 1

            print(f"Purchase Decision: True")
            return "Purchase"
        else:
            print(f"Purchase Decision: False")
            return "Do Not Purchase"


    def made_positive_comment(self):
        pass

    def step(self):
        # Implement any customer behavior or interactions with the platform
        self.update_satisfaction_level()


class ProductAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # Product attributes
        self.retailer = None
        self.price = 10 * np.random.beta(1, 1)  # Beta distribution, tends to higher price sensitivity
        self.quality = np.random.beta(1, 1)
        self.discount = np.random.beta(1, 1)
        self.keywords = initialize_keyword()
        self.brand = initialize_brand()
        self.sales_count = 0


class RetailerAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # Retailer attributes
        self.products = []
        self.rating = random.uniform(0, 1)

    def step(self):
        # Implement retailer behavior, e.g., updating product availability, offering discounts, etc.
        pass




class GenerativeAI:
    def __init__(self):
        # Initialize any necessary attributes
        self.customers_info = {}

    def receive_customer_info(self, customers_info, keywords):
        # Process received customer information and keywords
        for customer_id, info in customers_info.items():
            # Process and store customer info for future use
            # This could include preferences, past purchases, etc.
            # For simplicity, this example doesn't implement detailed logic
            pass

    def generate_basic_recommendations(self, product_agents):
        """
        Implement basic recommendation logic by randomly shuffling product agents.
        """
        # Shuffle the list of product agents to simulate recommendation
        shuffled_agents = list(product_agents)  # Create a copy to avoid modifying the original list
        random.shuffle(shuffled_agents)

        recommendations = shuffled_agents
        return recommendations

    def generate_personalized_recommendations(self, customer, products):
        """Provide personalized recommendations based on customer sensitivities."""
        # Initialize a dictionary to store product scores
        product_scores = {}

        # Iterate through each product and calculate scores based on customer sensitivities
        for product in products:
            # Calculate scores based on customer sensitivities
            price_score = (1 - customer.price_sensitivity) * (1 - product.price / 10)
            quality_score = customer.quality_sensitivity * product.quality
            discount_score = customer.discount_sensitivity * product.discount

            total_score = price_score + quality_score + discount_score
            product_scores[product] = total_score

        # Sort products based on total score
        sorted_products = sorted(products, key=lambda p: product_scores[p], reverse=True)

        # Create recommendations list from sorted products
        recommendations = sorted_products

        return recommendations

    def learn_from_customer_interactions(self, customers_feedback):
        # TODO：define GenAI self learning
        # Learn from customer feedback and update algorithms
        for customer_id, feedback in customers_feedback.items():
            # Process and learn from feedback
            # For example, update product popularity based on feedback
            for product_id in feedback.get('purchased', []):
                self.product_popularity[product_id] = self.product_popularity.get(product_id, 0) + 1
