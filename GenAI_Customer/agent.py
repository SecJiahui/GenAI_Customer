import mesa
import random


class CustomerAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # Customer attributes
        self.shopping_history = []
        self.areas_of_interest = []
        self.satisfaction = random.uniform(0, 1)
        self.purchase_threshold = random.uniform(0, 1)
        self.price_sensitivity = random.uniform(0, 1)
        self.quality_sensitivity = random.uniform(0, 1)
        self.discount_sensitivity = random.uniform(0, 1)
        self.brand_loyalty = random.uniform(0, 1)

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

        # Make a purchase decision based on decision factor and threshold
        purchase_threshold = 0.6
        if decision_factor > purchase_threshold:
            return "Purchase"
        else:
            return "Do Not Purchase"

    def step(self):
        # Implement any customer behavior or interactions with the platform
        pass

    def made_positive_comment(self):
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
        # TODO：define brand
        self.brand = brand


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
    def generate_recommendations(self):
        # Implement basic recommendation logic
        pass

    def receive_customer_info(self, customers_info, keywords):
        # Process received customer information and keywords
        pass

    def provide_personalized_recommendations(self, customers):
        # Provide personalized recommendations
        pass

    def learn_from_customer_interactions(self, customers_feedback):
        # Learn from customer feedback and update algorithms
        pass
