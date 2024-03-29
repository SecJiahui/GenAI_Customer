import mesa
import random
from enum import Enum

import numpy as np

import GenAI_Customer


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
        "Cooking", "Art", "Literature", "Cinema", "Gaming"
    ]
    return random.sample(potential_interests, k=2)


def initialize_keyword():
    potential_keyword = [
        "Sports", "Technology", "Fashion", "Travel", "Music",
        "Cooking", "Art", "Literature", "Cinema", "Gaming"
    ]
    return random.sample(potential_keyword, k=5)


def initialize_brand():
    potential_brand = ["A", "B", "C", "D", "E"]
    return random.sample(potential_brand, 1)


class CustomerAgent(mesa.Agent):
    def __init__(self, unique_id, model, willing_to_share):
        super().__init__(unique_id, model)
        self.shopping_history = []
        self.review_history = []
        self.shopping_amount = 0
        self.interests = initialize_interests()  # initialize interest
        self.willing_to_share_info = willing_to_share  # customer agree to share their information
        self.satisfaction = random.uniform(0.3, 0.7)
        self.price_sensitivity = np.random.beta(2, 5)  # Beta distribution, tends to higher price sensitivity
        self.quality_sensitivity = np.random.beta(5, 2)  # Beta distribution, tends to lower quality sensitivity
        self.content_sensitivity = np.random.beta(2, 5)  # Beta distribution, tends to higher price sensitivity
        self.brand_loyalty = np.random.beta(2, 2)  # Beta distribution, balanced brand loyalty
        self.mean_purchase_position = None
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

    def make_purchase_decision(self, product, use_gen_ai, content_improvement, generative_ai_creativity):
        """
        Make a purchase decision for a given product based on various factors.

        This method evaluates a product based on its price, quality, discount, and relevance to the customer's interests.
        A decision factor is calculated by considering the customer's sensitivities to price, quality, and contents.
        Higher price sensitivity decreases the decision factor with higher product prices, while higher quality and content
        sensitivities increase the decision factor respectively.

        Additionally, the method checks if the product's keywords match the customer's interests and if the product's brand
        is already in the customer's shopping history, which further influences the decision factor.

        The method also takes into account the existing customer comments for the product. It randomly samples three comments
        (or uses all if less than three are available) and adjusts the decision factor based on these comments. Ratings of 4
        and 5 are considered positive and increase the decision factor, while ratings of 1 and 2 are considered negative and
        decrease it.

        A purchase decision is made if the final decision factor exceeds a specified threshold. If a purchase is made, the
        product's sales count is incremented, and the product is added to the customer's shopping history.

        Returns:
        A string indicating the purchase decision: 'Purchase' or 'Do Not Purchase'.
        """

        # Extract product attributes
        product_price = product.price
        product_quality = product.quality
        product_content = product.content_score
        product_keywords = product.keywords
        brand = product.brand

        purchase_decision = False

        # Calculate weighted factors
        # negative influence for higher price_sensitivity
        price_factor = (1 - self.price_sensitivity) * (1 - product_price)
        # positive influence for higher quality_sensitivity
        quality_factor = self.quality_sensitivity * product_quality
        # Adjust content factor based on whether generative AI is used

        # Initially set the content factor based on the product's content score.
        # The use of generative AI and customer interests can further influence this factor.
        content_factor = product_content

        # Check each customer interest against the product's keywords.
        # Increase the content factor for each matching interest, indicating higher relevance.
        for interest in self.interests:
            if interest in product_keywords:
                content_factor += 2

        content_match_count = sum(interest in product_keywords for interest in self.interests)

        content_matched = False
        if content_match_count > 0:
            content_matched = True

        # Adjust the content factor based on the use of generative AI.
        # If generative AI is used, incorporate the learning rate into the content score and apply the customer's content sensitivity.
        # Otherwise, simply apply the content sensitivity to the base product content score.
        if use_gen_ai:
            content_factor = self.content_sensitivity * (content_factor + generative_ai_creativity)
        else:
            content_factor = self.content_sensitivity * product_content

        decision_factor = price_factor + quality_factor + content_factor

        # print Decision Factors information
        """print(
            f'Decision Factors for product: {product.unique_id}, '
            f'with Price Factor: {price_factor}, Quality Factor: {quality_factor}，Content Factor: {content_factor}'
        )"""

        # Increase decision factor if product keywords match customer interests


        # Increase decision factor if product brand matches customer shopping list
        if brand in self.shopping_history:
            decision_factor += (0.1 * self.brand_loyalty)
            # print(f"Brand: {brand} has been purchased before.")

        # Retrieve and process comments for the product
        product_comments = list(product.customers_comment.items())

        # Check if there are enough comments for sampling
        if len(product_comments) >= 3:
            # Randomly sample three comments
            sampled_comments = random.sample(product_comments, 3)
        elif product_comments:
            # Use all available comments if less than three
            sampled_comments = product_comments
        else:
            # No comments available for the product
            sampled_comments = []

        # Adjust the decision factor based on the comments
        for customer_id, comment in sampled_comments:
            if comment >= 4:  # Positive ratings
                decision_factor += 0.1
            elif comment <= 2:  # Negative ratings
                decision_factor -= 0.1
            # Record the interaction for visualization
            self.review_history.append((customer_id, comment))

        # print(f"Total: {decision_factor}")

        # Make a purchase decision based on decision factor and threshold
        purchase_threshold = 1.9
        if decision_factor > purchase_threshold:
            purchase_decision = True

        if purchase_decision:
            product.sales_count += 1
            self.shopping_history.append(product)
            # print(f"Purchase Decision: True")
        """else:
            print(f"Purchase Decision: False")"""

        decision_info = {
            'purchase_decision': purchase_decision,
            'decision_factor': decision_factor,
            'generative_ai_creativity': generative_ai_creativity,
            'content_matched': content_matched
        }

        return decision_info

    def make_comment(self, product):
        """
        Make a comment on the product, returning a rating between 1 and 5.
        The rating is influenced by customer satisfaction.
        """
        if self.satisfaction > 0.7:
            rating = random.choice([4, 5])  # More likely to give 4 or 5
        elif self.satisfaction > 0.5:
            rating = random.choice([3, 4])  # More likely to give 3, 4, or 5
        elif self.satisfaction > 0.4:
            rating = random.choice([2, 3, 4])  # More likely to give 1, 2, or 3
        else:
            rating = random.choice([1, 2, 3])  # Can give any rating, but less likely to give 5

        product.customers_comment[self.unique_id] = rating

        # print(f"Customer {self.unique_id} rated product {product.unique_id} with a rating of {rating}")

    def step(self):
        # Implement any customer behavior or interactions with the platform
        self.update_satisfaction_level()


class ProductAgent(mesa.Agent):
    def __init__(self, unique_id, model, price=None, quality=None, content=None, keywords=None, brand=None):
        super().__init__(unique_id, model)
        # Product attributes with options for custom initialization
        self.seller = None
        self.price = np.random.beta(1, 1) if price is None else price
        self.quality = np.random.beta(1, 1) if quality is None else quality
        self.content_score = np.random.beta(4, 7) if content is None else content
        self.keywords = initialize_keyword() if keywords is None else keywords
        self.brand = initialize_brand() if brand is None else brand
        self.customers_comment = {}
        self.sales_count = 0


class PlatformOwnerAgent(mesa.Agent):
    def __init__(self, unique_id, model, capacity_gen_ai=None, creativity_gen_ai=None):
        super().__init__(unique_id, model)
        # Product attributes with options for custom initialization
        self.capacity_gen_ai = capacity_gen_ai
        self.creativity_gen_ai = creativity_gen_ai


class SellerAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # Retailer attributes
        self.products = []
        self.rating = random.uniform(0, 1)

    def step(self):
        # Implement retailer behavior, e.g., updating product availability, offering discounts, etc.
        pass


class GenerativeAI:
    def __init__(self, model, learning_rate=None, capacity=0.5, creativity=0.5):
        # Initialize any necessary attributes
        self.customers_info = {}
        self.learning_rate = learning_rate
        self.product_popularity = {}
        self.model = model
        self.capacity = capacity
        self.creativity = creativity
        self.content_improvement = 0.3

    def generate_basic_content(self, product_agents):
        """
        Implement basic production content logic by randomly shuffling product agents.
        """
        # Shuffle the list of product agents to simulate recommendation
        shuffled_agents = list(product_agents)  # Create a copy to avoid modifying the original list
        random.shuffle(shuffled_agents)

        basic_content = shuffled_agents
        return basic_content

    def generate_personalized_content(self, customer, products):
        """Provide personalized production content on customer sensitivities."""
        # Initialize a dictionary to store product scores
        product_scores = {}

        # Iterate through each product and calculate scores based on customer sensitivities
        for product in products:
            product_keywords = product.keywords
            brand = product.brand
            price_score = (1 - customer.price_sensitivity) * (1 - product.price)
            quality_score = customer.quality_sensitivity * product.quality

            content_factor = product.content_score + self.content_improvement
            # Increase content factor if product keywords match customer interests
            for interest in customer.interests:
                if interest in product_keywords:
                    # print(f"Customer is interested in this product")
                    content_factor += 0.1

            content_score = customer.content_sensitivity * content_factor

            total_score = price_score + quality_score + content_score

            product_scores[product] = total_score

        # Find the product that gives the highest total_score for this customer
        best_product = max(product_scores, key=product_scores.get)

        # Extract the best attributes and modify slightly to create a new product content
        new_product_attributes = {
            'price': best_product.price,
            'quality': best_product.quality,
            'content': best_product.content_score,
            'keywords': best_product.keywords,
            'brand': initialize_brand()
        }

        new_product = self.generate_new_product(new_product_attributes)

        # Randomly select a retailer for the new product
        retailer = random.choice(
            [agent for agent in self.model.schedule.agents if isinstance(agent, GenAI_Customer.agent.SellerAgent)])
        new_product.seller = retailer
        retailer.products.append(new_product)

        # Add new product to products list
        products.insert(0, new_product)

        """# Update scores with the new product
        new_product_score = (1 - customer.price_sensitivity) * (1 - new_product.price) \
                            + customer.quality_sensitivity * new_product.quality \
                            + customer.content_sensitivity * new_product.content_score
        product_scores[new_product] = new_product_score

        # Sort products based on creativity
        num_products_to_sort = int(len(products) * self.creativity)
        top_products = sorted(products[:num_products_to_sort], key=lambda p: product_scores[p], reverse=True)
        sorted_products = top_products + products[num_products_to_sort:]"""

        content = products

        return content

    def learn_from_customer_interactions(self, customers_feedback):
        # Learn from customer feedback and update algorithms
        for customer_id, feedback in customers_feedback.items():
            # Process and learn from feedback
            # For example, update product popularity based on feedback
            for product_id in feedback.get('purchased', []):
                self.product_popularity[product_id] = self.product_popularity.get(product_id, 0) + self.learning_rate

    def generate_new_product(self, best_attributes):
        """
        Generate a new product based on the best attributes for a customer.
        """
        new_product = ProductAgent(
            unique_id=1008610086,
            model=self.model,
            price=max(best_attributes['price'] - 10 * self.learning_rate, 0),
            quality=min(best_attributes['quality'] + self.learning_rate, 1),
            content=min(best_attributes['content'] + self.learning_rate, 1),
            keywords=best_attributes['keywords']
        )

        # print(new_product.unique_id)

        # Set attributes for the new product based on customer data and trends
        return new_product
