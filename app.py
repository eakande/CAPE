import streamlit as st
import streamlit.components.v1 as components
from PIL import  Image
import numpy as np
from lime_explainer import explainer, tokenizer, METHODS


def format_dropdown_labels(val):
    return METHODS[val]['name']

# Define page settings
st.set_page_config(
    page_title='LIME explainer app for classification models',
    # layout="wide"
)

display = Image.open('Logo.jpeg')
display = np.array(display)
col1, col2 = st.columns(2)
col1.image(display, width = 800)

# Build app
title_text = 'Explainer Dashboard for Sentiment Analysis'
subheader_text = '''1: Strongly Negative &nbsp 2: Weakly Negative &nbsp  3: Neutral &nbsp  4: Weakly Positive &nbsp  5: Strongly Positive'''

st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)
st.markdown(f"<h5 style='text-align: center;'>{subheader_text}</h5>", unsafe_allow_html=True)
st.text("")
input_text = st.text_input('Enter your text:', "")
n_samples = st.text_input('Number of samples to generate for LIME explainer: (For really long input text, go up to 5000)', value=1000)
method_list = tuple(label for label, val in METHODS.items())
method = st.selectbox(
    'Choose classifier:',
    method_list,
    index=3,
    format_func=format_dropdown_labels,
)

if st.button("Explain Results"):
    with st.spinner('Calculating...'):
        text = tokenizer(input_text)
        exp = explainer(method,
                        path_to_file=METHODS[method]['file'],
                        text=text,
                        lowercase=METHODS[method]['lowercase'],
                        num_samples=int(n_samples))
        # Display explainer HTML object
        components.html(exp.as_html(), height=800)

