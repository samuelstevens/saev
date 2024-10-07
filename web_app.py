import json
import os
import pickle
import random

import pandas as pd
import plotly.express as px
import streamlit as st
import torch


def ordinal(n):
    if n == 1:
        return ""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def get_neuron_indices():
    directory = "web_app/neurons"
    indices = []
    for name in os.listdir(directory):
        full_path = os.path.join(directory, name)
        # Check if it's a directory
        if os.path.isdir(full_path) and name.isdigit():
            indices.append(int(name))
    random.shuffle(indices)
    return indices


def get_image_indices():
    directory = "web_app/images"
    indices = []
    for name in os.listdir(directory):
        full_path = os.path.join(directory, name)
        # Check if it's a directory
        if os.path.isdir(full_path) and name.isdigit():
            indices.append(int(name))
    random.shuffle(indices)
    return indices


def game_get_image(refresh_all=True):
    if st.session_state.game_blurr:
        image_index = st.session_state.game_image_indices[st.session_state.game_index]
        image = f"https://saeexplorer.s3.eu-west-2.amazonaws.com/saeexplorer/images/{image_index}/blurred_image.png"
    else:
        image_index = st.session_state.game_image_indices[st.session_state.game_index]
        image = f"https://saeexplorer.s3.eu-west-2.amazonaws.com/saeexplorer/images/{image_index}/image.png"
    st.session_state.game_image = image
    if refresh_all:
        df = pd.read_feather(f"web_app/images/{image_index}/activations.feather")
        fig = px.line(
            df,
            x="X",
            y="Y",
            labels={
                "X": "SAE index",  # Custom x-axis label
                "Y": "Activation value",  # Custom y-axis label
            },
        )
        st.session_state.game_activations = fig
        with open(f"web_app/images/{image_index}/top_five_indices.json", "r") as file:
            top_five_indices = json.load(file)
        st.session_state.top_five_features = [
            f"https://saeexplorer.s3.eu-west-2.amazonaws.com/saeexplorer/neurons/{neuron_index}/highest_activating_images.png"
            for neuron_index in top_five_indices
        ]


def game_next_image():
    st.session_state.game_index = (st.session_state.game_index + 1) % len(
        st.session_state.game_image_indices
    )
    st.session_state.game_blurr = True
    game_get_image()


def game_previous_image():
    st.session_state.game_index = (st.session_state.game_index - 1) % len(
        st.session_state.game_image_indices
    )
    st.session_state.game_blurr = True
    game_get_image()


def game_unblurr():
    if st.session_state.game_blurr:
        st.session_state.game_blurr = False
        game_get_image(refresh_all=False)


def set_selected_neuron():
    set_navigator_meta_data()
    set_navigator_image_grid()
    set_navigator_mlp()


def set_navigator_meta_data():
    with open(
        f"web_app/neurons/{st.session_state.navigator_selected_neuron_index}/meta_data.pkl",
        "rb",
    ) as file:
        # Load the data from the file
        st.session_state.navigator_meta_data = pd.DataFrame([pickle.load(file)])


def set_navigator_image_grid():
    st.session_state.navigator_image_grid = f"https://saeexplorer.s3.eu-west-2.amazonaws.com/saeexplorer/neurons/{st.session_state.navigator_selected_neuron_index}/highest_activating_images.png"


def set_navigator_mlp():
    df = pd.read_feather(
        f"web_app/neurons/{st.session_state.navigator_selected_neuron_index}/MLP.feather"
    )
    fig = px.line(
        df,
        x="X",
        y="Y",
        labels={
            "X": "MLP index",  # Custom x-axis label
            "Y": "Cosine similarity",  # Custom y-axis label
        },
    )
    fig.update_layout(
        yaxis=dict(range=[-0.3, 0.6])  # Set the y-axis range
    )
    st.session_state.navigator_mlp = fig


def navigator_previous_neuron():
    st.session_state.navigator_current_index = (
        st.session_state.navigator_current_index - 1
    ) % len(st.session_state.navigator_current_neuron_indices)
    st.session_state.navigator_selected_neuron_index = (
        st.session_state.navigator_current_neuron_indices[
            st.session_state.navigator_current_index
        ]
    )
    set_selected_neuron()


def navigator_next_neuron():
    st.session_state.navigator_current_index = (
        st.session_state.navigator_current_index + 1
    ) % len(st.session_state.navigator_current_neuron_indices)
    st.session_state.navigator_selected_neuron_index = (
        st.session_state.navigator_current_neuron_indices[
            st.session_state.navigator_current_index
        ]
    )
    set_selected_neuron()


def navigator_positive_entropy():
    st.session_state.navigator_current_neuron_indices = (
        st.session_state.positive_entropy_list
    )
    st.session_state.navigator_current_index = (
        0  # This is the index of the list of neuron indices
    )
    st.session_state.navigator_selected_neuron_index = (
        st.session_state.navigator_current_neuron_indices[
            st.session_state.navigator_current_index
        ]
    )
    set_selected_neuron()


def navigator_reset_entropy():
    st.session_state.navigator_current_neuron_indices = (
        st.session_state.navigator_all_neuron_indices
    )
    st.session_state.navigator_current_index = (
        0  # This is the index of the list of neuron indices
    )
    st.session_state.navigator_selected_neuron_index = (
        st.session_state.navigator_current_neuron_indices[
            st.session_state.navigator_current_index
        ]
    )
    set_selected_neuron()


def set_dropdown_index():
    st.session_state.navigator_selected_neuron_index = (
        st.session_state.navigator_dropdown_selected_neuron
    )
    st.session_state.navigator_current_index = (
        st.session_state.navigator_current_neuron_indices.index(
            st.session_state.navigator_selected_neuron_index
        )
    )
    set_selected_neuron()


# Define a function to render the home page
def home_page():
    st.markdown(
        "<h1 style='text-align: center;'>Home Page</h1>", unsafe_allow_html=True
    )
    st.header("Welcome to the App!")


# Define a function to render Subpage 1
def navigator():
    st.markdown(
        "<h1 style='text-align: center;'>Neuron navigator</h1>", unsafe_allow_html=True
    )

    if "navigator_selected_neuron_index" not in st.session_state:
        st.session_state.navigator_current_index = (
            0  # This is the index of the list of neuron indices
        )
        st.session_state.navigator_selected_neuron_index = (
            st.session_state.navigator_current_neuron_indices[
                st.session_state.navigator_current_index
            ]
        )

    if "navigator_meta_data" not in st.session_state:
        set_navigator_meta_data()

    if "navigator_image_grid" not in st.session_state:
        set_navigator_image_grid()

    if "navigator_mlp" not in st.session_state:
        set_navigator_mlp()

    col1, col2, col3, col4 = st.columns(4, gap="small")
    with col1:
        st.button(
            "Previous neuron",
            use_container_width=True,
            on_click=navigator_previous_neuron,
        )

    with col2:
        st.button(
            "Next neuron", use_container_width=True, on_click=navigator_next_neuron
        )

    with col3:
        st.button(
            "Filter entropy > 0",
            use_container_width=True,
            on_click=navigator_positive_entropy,
        )

    with col4:
        st.button(
            "Reset filter", use_container_width=True, on_click=navigator_reset_entropy
        )

    st.session_state.navigator_dropdown_selected_neuron = st.selectbox(
        "Select a neuron:",
        st.session_state.navigator_current_neuron_indices,
        index=st.session_state.navigator_current_index,
    )

    if (
        st.session_state.navigator_dropdown_selected_neuron
        != st.session_state.navigator_selected_neuron_index
    ):
        set_dropdown_index()

    # Simulated data for display
    st.header("Meta data")
    st.dataframe(
        st.session_state.navigator_meta_data, hide_index=True, use_container_width=True
    )
    st.header("Top 16 highest activating images")
    st.image(st.session_state.navigator_image_grid, use_column_width=True)
    st.header("Neuron alignment")
    st.plotly_chart(st.session_state.navigator_mlp)


# Define a function to render Subpage 2
def game():
    if "game_image" not in st.session_state:
        st.session_state.game_blurr = True
        st.session_state.game_index = 0
        game_get_image()

    st.markdown(
        "<h1 style='text-align: center;'>Guess the input image!</h1>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.button(
            "Previous image", use_container_width=True, on_click=game_previous_image
        )  # Chnages index value, sets blurr to true, then loads new image into game_image. see above code
    with col2:
        st.button(
            "Next image", use_container_width=True, on_click=game_next_image
        )  # Chnages index value, sets blurr to true, then loads new image into game_image. see above code
    st.button(
        "Unblurr", use_container_width=True, on_click=game_unblurr
    )  # If blurr is currently True, loads same image unblurred into game_image
    cola, colb = st.columns(2, gap="small")
    with cola:
        st.markdown("<p style='text-align: center;'>Input:</p>", unsafe_allow_html=True)
        st.image(st.session_state.game_image, use_column_width=True)
    with colb:
        st.markdown(
            "<p style='text-align: center;'>SAE activations:</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(st.session_state.game_activations, use_container_width=True)

    st.header("Top SAE features")
    for i, image in enumerate(st.session_state.top_five_features):
        st.markdown(
            f"<p style='text-align: center;'>{ordinal(i+1)} highest SAE feature:</p>",
            unsafe_allow_html=True,
        )
        st.image(image, use_column_width=True)


# A simple function to change the page state
def set_page(page_name):
    st.session_state.page = page_name


# Sidebar for navigation
with st.sidebar:
    if st.button("🏠 Home"):
        set_page("home")
    st.text("  ")  # Adding some space before subpage buttons
    st.text("  ")  # Adding some space before subpage buttons
    st.text("  ")  # Adding some space before subpage buttons
    if st.button(" 🔎 Neuron navigator"):
        set_page("navigator")
    if st.button(" 🎮 Guess the input image"):
        set_page("game")

# Define a dictionary linking page names to function renderers
pages = {"home": home_page, "navigator": navigator, "game": game}

# Initialize the session state for page if it's not already set
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "navigator" and (
    "navigator_all_neuron_indices" not in st.session_state
    or "positive_entropy_list" not in st.session_state
):  # Included as a santiy check. Should be set when the navigator button is pressed.
    st.session_state.navigator_all_neuron_indices = get_neuron_indices()
    st.session_state.navigator_current_neuron_indices = (
        st.session_state.navigator_all_neuron_indices
    )
    entropy = torch.load("web_app/neurons/entropy.pt")
    st.session_state.positive_entropy_list = [
        index
        for index in st.session_state.navigator_all_neuron_indices
        if entropy[index].item() > 0
    ]

if st.session_state.page == "game" and "game_image_indices" not in st.session_state:
    st.session_state.game_image_indices = get_image_indices()

# Render the current page
pages[st.session_state.page]()
