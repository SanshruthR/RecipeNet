import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import time
from typing import Tuple, Dict
import json
import gradio as gr

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap',
    'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake',
    'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame',
    'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel',
    'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari',
    'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon',
    'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream',
    'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels',
    'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake',
    'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
    'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu',
    'tuna_tartare', 'waffles'
]

# Function to load JSON files and concatenate them
def load_and_concatenate_json(json_paths: Tuple[str, str]) -> list:
    combined_data = []
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data = json.load(f)
            combined_data.extend(data)
    return combined_data

json_paths = ('./demos/recipenet/recipesData_v1.json', './demos/recipenet/recipesData_v2.json')  # Replace with your JSON file paths
combined_data = load_and_concatenate_json(json_paths)

# Load the model
weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)

# Update the classifier and heads
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=101)
)
model.heads = nn.Linear(in_features=768, out_features=101)

# Load the state dictionary
state_dict = torch.load('./demos/recipenet/best_model.pth', map_location=device)
model.load_state_dict(state_dict)
model.to(device)
auto_transforms = weights.transforms()

def get_recipe_details(label: str, combined_data: list) -> dict:
    label = label.replace("_", " ")
    for item in combined_data:
        if item["FoodName"].lower() == label.lower():
            return {
                "Ingredients": item["RawIngredients"],
                "RecipeURL": item["RecipeURL"],
                "RecipeName": item["RecipeName"]
            }
    return {}

def predict(img: Image.Image, model, class_names, device, combined_data: list) -> Tuple[Dict[str, float], float, dict]:
    # Start a timer
    start_time = time.time()

    # Transform the input image for use with EfficientNet_B0
    img = auto_transforms(img).unsqueeze(0).to(device)

    # Put the model in evaluation mode
    model.eval()

    with torch.inference_mode():
        # Make predictions
        pred_probs = torch.softmax(model(img), dim=1)

    # Create a dictionary of prediction labels and probabilities
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # End the timer
    end_time = time.time()
    pred_time = end_time - start_time

    # Get the maximum prediction label and probability
    max_label = max(pred_labels_and_probs, key=pred_labels_and_probs.get)
    max_prob = pred_labels_and_probs[max_label]

    # Fetch recipe details for the predicted label
    recipe_details = get_recipe_details(max_label, combined_data)

    return pred_labels_and_probs, pred_time, recipe_details

def predict_wrapper(img: Image.Image):
    # Provide the additional arguments required by the predict function
    pred_labels_and_probs, pred_time, recipe_details = predict(img, model, class_names, device, combined_data)

    # Extract the recipe, website, and ingredients from recipe_details
    recipe = recipe_details.get('RecipeName', '')
    website = recipe_details.get('RecipeURL', '')
    ingredients = recipe_details.get('Ingredients', [])
    time = len(ingredients) * 10  # Assuming each ingredient takes 10 seconds to prepare
    ingredients = ', '.join(ingredients)  # Convert the list of ingredients to a string

    return pred_labels_and_probs, pred_time, recipe, time, website, ingredients

# Define the CSS for transparent background
css = """
.gradio-container,.svelte-vt1mxs.gap.panel {
    background: repeating-linear-gradient(
        to top,
        rgba(255, 255, 255, 0.03) 0px 2px,
        transparent 2px 4px
    ),
    linear-gradient(to bottom, #200933 75%, #3d0b43);
    color: #d5d7de;
    font-family: sans-serif;
}
gradio-app[control_page_title="true"][embed="false"][eager="true"] {
    background-color: #271139 !important;
    background-image: linear-gradient(to bottom, transparent 95%, #000 5%);
}

.gradio-root {
    overflow: hidden;
    box-shadow: 0px 0px 10px 0px rgba(0, 0, 0, 0.1);
}

.gradio-control {
    border: none;
}

.gradio-input {
    background-color: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
}
"""

demo = gr.Interface(
    fn=predict_wrapper,
    inputs=gr.Image(type='pil'),
    outputs=[
        gr.Label(num_top_classes=3, label='Predictions'),
        gr.Number(label='Prediction Time (s)'),
        gr.Textbox(label='Dish Name', elem_id="textbox_id"),
        gr.Textbox(label='Preparation time(in Minutes)'),
        gr.Textbox(label='Recipe available at'),
        gr.Textbox(label='Ingredients')
    ],
    description="recipenet",
    article="Made using EfficientNet-b0",
    css=css,  # Add the custom CSS
    examples=[["./demos/recipenet/examples/pic1.jpg"], ["./demos/recipenet/examples/pic2.jpg"], ["./demos/recipenet/examples/pic3.jpg"]]
)

demo.launch(debug=False, share=False)
