import os
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torch

# Import utils
from utils.model_utils import load_model, load_classes, predict_and_gradcam
from utils.image_utils import apply_transformations, preprocess_image
from utils.gradcam_utils import get_gradcam
from utils.recipe_utils import get_recipe_and_nutrition

# os.environ["TORCH_HOME"] = os.path.abspath(os.getcwd())
os.environ["STREAMLIT_WATCH_DIRECTORIES"] = "false"
os.environ["STREAMLIT_WATCH_MODULES"] = "false"

# Config
MODEL_PATHS = {
    'Baseline': './models/baseline/food101_checkpoint_best.pth',
    'Pretrained Freeze': './models/pretrained_freeze/food101_checkpoint_best.pth',
    'Pretrained Unfreeze': './models/pretrained_unfreeze/food101_checkpoint_best.pth',
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = load_classes()

# Load models
@st.cache_resource
def load_models():
    return {name: load_model(path) for name, path in MODEL_PATHS.items()}

models = load_models()

def show():
    st.markdown("## 🤖 Model Predictions Comparison")
    models = load_models()
    # CSS Styling
    css_file = os.path.join(os.path.dirname(__file__), '..', 'style.css')
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Track previous slider values to control model re-processing
    previous_brightness = st.session_state.get('brightness_factor', 1.0)
    previous_contrast = st.session_state.get('contrast_factor', 1.0)
    previous_rotation = st.session_state.get('rotation_angle', 0)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:
        # Initialize session state
        if 'initialized' not in st.session_state or st.session_state.get('last_uploaded_image') != uploaded_file.name:
            st.session_state.update({
                'rotation_angle': 0,
                'brightness_factor': 1.0,
                'contrast_factor': 1.0,
                'last_uploaded_image': uploaded_file.name,
                'cached_outputs': {},
                'predictions': {},
                'selected_detail': None,
                'recipe_info': None,
                'initialized': True,
            })

        # Load original image
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown("### 🖼️ Original vs Transformed Image")
        colA, colB = st.columns(2)

        with colA:
            st.markdown("**Original Image**")
            st.image(image.resize((300, 300)), use_container_width=True)

        # === Sliders in one row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state['brightness_factor'] = st.slider("Brightness", 0.1, 2.0, st.session_state['brightness_factor'])
        with col2:
            st.session_state['contrast_factor'] = st.slider("Contrast", 0.1, 2.0, st.session_state['contrast_factor'])
        with col3:
            st.session_state['rotation_angle'] = st.slider("Rotation (degrees)", -180, 180, st.session_state['rotation_angle'])

        # Apply transformations
        transformed_image = apply_transformations(
            image,
            image.width,  # No resize
            image.height,
            st.session_state['rotation_angle'],
            st.session_state['brightness_factor'],
            st.session_state['contrast_factor']
        )

        # Display transformed image
        with colB:
            st.markdown("**Transformed Image**")
            st.image(transformed_image.resize((300, 300)), use_container_width=True)

        # Preprocess for model
        input_tensor = preprocess_image(transformed_image, DEVICE)

        # Check if sliders changed to force reprocessing
        sliders_changed = (
            st.session_state['brightness_factor'] != previous_brightness
            or st.session_state['contrast_factor'] != previous_contrast
            or st.session_state['rotation_angle'] != previous_rotation
        )

        # Process models only if sliders changed or predictions are empty
        if sliders_changed or not st.session_state['cached_outputs'] or not st.session_state['predictions']:
            st.session_state['cached_outputs'] = {}
            st.session_state['predictions'] = {}
            for model_name, model in models.items():
                activation_map_resized, predictions = predict_and_gradcam(
                    model,
                    input_tensor,
                    image_size=(image.size[1], image.size[0]),
                    classes=classes
                )

                st.session_state['cached_outputs'][model_name] = activation_map_resized
                st.session_state['predictions'][model_name] = predictions

            # for model_name, model in models.items():
            #     outputs = model(input_tensor)
            #     probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            #     predicted_class_idx = probabilities.argmax().item()

            #     activation_map_resized, _ = get_gradcam(
            #         model,
            #         input_tensor,
            #         predicted_class_idx,
            #         image_size=(image.size[1], image.size[0])
            #     )

            #     top_probs, top_idxs = torch.topk(probabilities, 3)
            #     predictions = []
            #     for prob, class_idx in zip(top_probs, top_idxs):
            #         if 0 <= class_idx < len(classes):
            #             predictions.append((classes[class_idx], prob.item()))

            #     st.session_state['cached_outputs'][model_name] = activation_map_resized
            #     st.session_state['predictions'][model_name] = predictions

        # Display GradCAM heatmaps
        col1, col2, col3 = st.columns(3)
        for model_name, col in zip(models.keys(), [col1, col2, col3]):
            with col:
                st.markdown(f"**{model_name}**")
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(transformed_image)
                ax.imshow(st.session_state['cached_outputs'][model_name], cmap='jet', alpha=0.5)
                ax.axis('off')
                st.pyplot(fig)


        # Display prediction buttons
        col1, col2, col3 = st.columns(3)

        for model_name, col in zip(models.keys(), [col1, col2, col3]):
            with col:
                st.markdown(f"<div class='model-column'>", unsafe_allow_html=True)
                for class_name, prob in st.session_state['predictions'][model_name]:
                    button_label = f"{class_name.capitalize()}\n({prob:.2%})"
                    if st.button(button_label, key=f"{model_name}_{class_name}_{uploaded_file.name}"):
                        st.session_state['selected_detail'] = class_name
                        with st.spinner("Fetching recipe..."):
                            st.session_state['recipe_info'] = get_recipe_and_nutrition(class_name)
                st.markdown("</div>", unsafe_allow_html=True)

        # Display recipe details
        if st.session_state['recipe_info']:
            st.markdown("---")
            recipe_html = st.session_state['recipe_info'].replace(
                '<img',
                '<img style="border: 4px solid #ccc; border-radius: 8px; margin-bottom: 16px; width: 100%; height: auto;"'
            )
            st.markdown(recipe_html, unsafe_allow_html=True)
