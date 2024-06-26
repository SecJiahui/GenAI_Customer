import mesa
import random
from enum import Enum
import numpy as np
import code
# random.seed(42)
# np.random.seed(42)


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


def initialize_brand():
    potential_brand = ["A", "B", "C", "D", "E"]
    return random.sample(potential_brand, 1)


class CustomerAgent(mesa.Agent):
    def __init__(self, unique_id, model, willing_to_share):
        super().__init__(unique_id, model)
        self.shopping_history = []
        self.review_history = []
        self.shopping_amount = 0
        self.willing_to_share_info = willing_to_share  # customer agree to share their information
        self.last_satisfaction = 0
        self.satisfaction = random.uniform(0.3, 0.7)
        self.price_sensitivity = np.random.beta(2, 5)  # Beta distribution, tends to higher price sensitivity
        self.quality_sensitivity = np.random.beta(5, 2)  # Beta distribution, tends to lower quality sensitivity
        self.content_sensitivity = np.random.beta(2, 5)  # Beta distribution, tends to higher price sensitivity
        self.brand_loyalty = np.random.beta(2, 2)  # Beta distribution, balanced brand loyalty
        self.mean_purchase_position = None
        self.mean_viewed_comments = 0
        self.state = self.get_satisfaction_level()
        self.num_content_matched = 0

    def get_satisfaction_level(self):
        if self.satisfaction >= 4:
            return State.HighSatisfaction
        elif 2 <= self.satisfaction < 4:
            return State.MediumSatisfaction
        else:
            return State.LowSatisfaction

    def update_satisfaction_level(self):
        self.state = self.get_satisfaction_level()

    def make_purchase_decision(self, product, customer_use_gen_ai, generative_ai_creativity, generative_ai_learning_rate, purchase_threshold):
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
        brand = product.brand

        purchase_decision = False
        # product_use_gen_ai = random.random() < generative_ai_creativity

        # Calculate weighted factors
        # negative influence for higher price_sensitivity
        price_factor = (1 - self.price_sensitivity) * (1 - product_price / 10)
        # positive influence for higher quality_sensitivity
        quality_factor = self.quality_sensitivity * product_quality

        # Initially set the content factor based on the product's content score.
        # The use of generative AI and customer interests can further influence this factor.
        content_factor = product_content

        content_matched = False

        # Adjust the content factor based on the use of generative AI.
        # If generative AI is used, incorporate the learning rate into the content score and apply the customer's content sensitivity.
        # Otherwise, simply apply the content sensitivity to the base product content score.
        if customer_use_gen_ai:
            content_factor = self.content_sensitivity * (content_factor + generative_ai_creativity * 0.5)
            if product_content + generative_ai_creativity > 2.5:
                content_matched = True
        else:
            content_factor = self.content_sensitivity * product_content

        decision_factor = price_factor + quality_factor + content_factor

        # print Decision Factors information
        """print(
            f'Decision Factors for product: {product.unique_id}, '
            f'with Price Factor: {price_factor}, Quality Factor: {quality_factor}，Content Factor: {content_factor}'
        )"""

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

        num_viewed_comments = 0

        # Adjust the decision factor based on the comments
        for customer_id, comment in sampled_comments:
            num_viewed_comments += 1
            if comment >= 4:  # Positive ratings
                decision_factor += 0.1
            elif comment <= 2:  # Negative ratings
                decision_factor -= 0.1
            # Record the interaction for visualization
            self.review_history.append((customer_id, comment))

        # print(f"Total: {decision_factor}")
        # Make a purchase decision based on decision factor and threshold
        if decision_factor > purchase_threshold:
            purchase_decision = True

        if purchase_decision:
            product.sales_count += 1
            self.shopping_history.append(product)
            # print(f"Purchase Decision: True")
        """else:
            print(f"Purchase Decision: False")"""

        decision_info = {
            'len_sampled_comments': num_viewed_comments,
            'purchase_decision': purchase_decision,
            'decision_factor': decision_factor,
            'generative_ai_learning_rate': generative_ai_learning_rate,
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
        self.price = 10 * np.random.beta(1, 1) if price is None else price
        self.quality = np.random.beta(1, 1) if quality is None else quality
        self.content_score = np.random.beta(4, 7) if content is None else content
        self.brand = initialize_brand() if brand is None else brand
        self.customers_comment = {}
        self.sales_count = 0


class PlatformOwnerAgent(mesa.Agent):
    def __init__(self, unique_id, model, learning_rate_gen_ai, capacity_gen_ai, creativity_gen_ai):
        super().__init__(unique_id, model)
        # Product attributes with options for custom initialization
        self.learning_rate_gen_ai = learning_rate_gen_ai
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


def generate_basic_content(product_agents):
    """
    Implement basic production content logic by randomly shuffling product agents.
    """
    # Shuffle the list of product agents to simulate recommendation
    shuffled_agents = list(product_agents)  # Create a copy to avoid modifying the original list
    random.shuffle(shuffled_agents)

    basic_content = shuffled_agents
    return basic_content


class GenerativeAI:
    def __init__(self, model, learning_rate, capacity):
        # Initialize necessary attributes
        self.learning_rate = learning_rate
        self.model = model
        self.capacity = capacity
        # Use a dictionary for creativity with customer ID as key
        self.creativity = {}  # Initialize an empty dict

    def initialize_gen_ai_creativity(self, creativity, customers):
        for customer in customers:
            if customer.unique_id not in self.creativity:
                # If customer is not in the creativity dict, initialize their creativity
                self.creativity[customer.unique_id] = creativity

    def update_gen_ai_creativity(self, customers):
        for customer in customers:
            # Calculate the change in creativity based on satisfaction change
            creativity_change = self.learning_rate * (
                    customer.satisfaction - customer.last_satisfaction) / max(0.01, customer.satisfaction)

            # Update the creativity for this specific customer
            new_creativity = self.creativity[customer.unique_id] + creativity_change

            # Ensure the new creativity value is between 0 and 3
            self.creativity[customer.unique_id] = max(0, min(new_creativity, 3))

            # Update the customer's last satisfaction
            customer.last_satisfaction = customer.satisfaction

