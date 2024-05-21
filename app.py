import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Hate Detect", layout="centered")

# Function to load the selected model
@st.cache_resource()
def load_model(model_name):
    try:
        if model_name == "DistilBERT":
            model_path = "vipulkumar49/hate_detect_distilbert"
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif model_name == "RoBERTa":
            model_path = "vipulkumar49/hate_detect_roberta"
            model = RobertaForSequenceClassification.from_pretrained(model_path)
            tokenizer = RobertaTokenizer.from_pretrained(model_path)
        elif model_name == "BERT":
            model_path = "vipulkumar49/hate_detect_bert"
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif model_name == "GPT2":
            model_path = "DrZombiee/GPTHateSpeech"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            raise ValueError("Invalid model selected.")
        return model, tokenizer
    except Exception as e:
        st.error("Error loading model. Please check if the model files exist.")
        st.stop()

# Function to classify text using the loaded model
def classify_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    return probabilities

# Predefined example tweets
example_tweets = [
    "I love the diversity in our community!",
    "You are an idiot and you don't deserve to be here.",
    "All people should be treated equally regardless of their background.",
    "Get out of my country, you don't belong here.",
    "I can't stand those people, they're so annoying.",
    "Our strength lies in our differences and our unity.",
]

# Streamlit app layout
st.title("Hate Detect")
st.write("Use this app to classify whether a given text contains hate speech.")

st.sidebar.title("Model Selection")
model_name = st.sidebar.selectbox("Select Model", ["DistilBERT", "RoBERTa", "BERT", "GPT2"])

# Load the selected model
with st.spinner("Loading model..."):
    model, tokenizer = load_model(model_name)
st.success("Model loaded successfully!")

st.subheader("Text Input")
st.write("Enter the text you want to classify or choose from the examples:")

# Dropdown for example tweets
example_choice = st.selectbox("Choose an example tweet or write your own:", [""] + example_tweets)

# Input box for user to enter text or use the selected example
user_input = st.text_area("Enter text here", example_choice if example_choice else "", height=150)

# Submit button to trigger classification
if st.button("Classify Text"):
    if user_input.strip():
        with st.spinner("Classifying..."):
            probabilities = classify_text(user_input, model, tokenizer)
        
        st.subheader("Results")
        st.write(f"Probabilities: {probabilities}")
        
        hate_speech_prob = probabilities[1]
        st.progress(hate_speech_prob)

        if hate_speech_prob > 0.5:  # Threshold for considering hate speech
            st.error("This text contains hate speech.")
        else:
            st.success("This text does not contain hate speech.")
    else:
        st.warning("Please enter some text to classify.")

# Footer
st.markdown("""
    <style>
        footer {visibility: hidden;}
        .stApp { bottom: 0px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <hr>
    <div style='text-align: center'>
        <small>© 2024 Hate Detect. All rights reserved.</small>
    </div>
    """, unsafe_allow_html=True)