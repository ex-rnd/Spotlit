# Core Packages
import streamlit as st
import altair as alt

# EDA Packages 
import pandas as pd
import numpy as np

# Utilities 
import joblib

# Importing Pipeline
pipeline = joblib.load(open("models/emotion_classifier_pipeline.pkl","rb"))

# Other Functions 

# Function - Predicting Emotions
def predict_emotions(docx):
    results = pipeline.predict([docx])
    return results[0]

# Function - Predicting Probability
def get_prediction_proba(docx):
    results = pipeline.predict_proba([docx])
    return results 


# Emoji - Dictionary
emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗",
                       "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔",
                       "shame":"😳", "surprise":"😮"}

# Main Function 

def main():
    st.title("Spotlit")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Home - Spotlit::Emotion Aware Chat Assist")
        
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')
            
        if submit_text:
            col1, col2 = st.columns(2)
            
            # Apply Functions - Other Functions
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                
                st.success("Prediction")
                #st.write(prediction)
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction,emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))
                
            with col2:
                st.success("Prediction Probability")
                # st.write(probability)
                proba_dataset = pd.DataFrame(probability, columns=pipeline.classes_)
                # st.write(proba_dataset.T)
                proba_dataset_clean = proba_dataset.T.reset_index()
                proba_dataset_clean.columns = ["emotions", "probability"]
                
                figure = alt.Chart(proba_dataset_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(figure, use_container_width=True)
        
    elif choice == "Monitor":
        st.subheader("Monitor App")
        
        
    else:
        st.subheader("About")
        
        
        
if __name__ == '__main__':
    main()