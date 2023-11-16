import mesa
import random
from enum import Enum


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


class CustomerAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # Customer attributes
        self.shopping_history = []
        self.shopping_amount = 0
        self.areas_of_interest = []
        self.satisfaction = random.uniform(0, 1)
        self.purchase_threshold = random.uniform(0, 1)
        self.price_sensitivity = random.uniform(0, 1)
        self.quality_sensitivity = random.uniform(0, 1)
        self.discount_sensitivity = random.uniform(0, 1)
        self.brand_loyalty = random.uniform(0, 1)
        self.state = self.get_satisfaction_level()

    def get_satisfaction_level(self):
        if self.satisfaction >= 0.8:
            return State.HighSatisfaction
        elif 0.6 <= self.satisfaction < 0.8:
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

        # Calculate weighted factors
        price_factor = self.price_sensitivity * product_price
        quality_factor = self.quality_sensitivity * product_quality
        discount_factor = self.discount_sensitivity * product_discount

        decision_factor = price_factor + quality_factor + discount_factor

        # Increase decision factor if product keywords match customer interests
        for interest in self.areas_of_interest:
            if interest in product_keywords:
                decision_factor += 0.1

        # Increase decision factor if product brand matches customer shopping list
        if brand in self.shopping_history:
            decision_factor += 0.1 * self.brand_loyalty

        # print Decision Factors infomation
        print(f"Decision Factors - Price: {price_factor}, Quality: {quality_factor}, "
              f"Discount: {discount_factor}, Total: {decision_factor}")

        # Make a purchase decision based on decision factor and threshold
        purchase_threshold = 0.6
        if decision_factor > purchase_threshold:
            product.sales_count += 1
            self.areas_of_interest.extend(product.keywords)
            print("Decision: Purchase")
            return "Purchase"
        else:
            print("Decision: Do Not Purchase")
            return "Do Not Purchase"

    def made_positive_comment(self):
        pass

    def step(self):
        # Implement any customer behavior or interactions with the platform
        pass


class ProductAgent(mesa.Agent):
    def __init__(self, unique_id, model, price, quality, discount, keywords, brand):
        super().__init__(unique_id, model)
        # Product attributes
        self.retailer = None
        self.price = price
        self.quality = quality
        self.discount = discount
        # TODO：define keywords
        self.keywords = keywords
        self.brand = brand
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
    # TODO：define Gen AI
    def generate_recommendations(self):
        # Implement basic recommendation logic
        recommendations = ["Product1", "Product2", "Product3"]
        return recommendations

    def receive_customer_info(self, customers_info, keywords):
        # Process received customer information and keywords
        pass

    def provide_personalized_recommendations(self, customers):
        # Provide personalized recommendations
        personalized_recommendations = {}
        for customer in customers:
            # Example: Provide recommendations based on Gen AI's logic
            recommendations = self.generate_recommendations()
            personalized_recommendations[customer.unique_id] = recommendations
        return personalized_recommendations

    def learn_from_customer_interactions(self, customers_feedback):
        # Learn from customer feedback and update algorithms
        pass
