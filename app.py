import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

import joblib
model=joblib.load(open("emotion_classifier.pkl","r",encoding='utf-8'))

def predict_emotion(msg):
        result=model.predict([msg])
        return result[0]

def predict_probability(msg):
        result=model.predict_proba([msg])
        return result

                             

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"😁", "joy":"😂", "neutral":"😃", "sadness":"😔", "shame":"😳", "surprise":"😮"}


def main():
        st.title("Emotion-Classifier App")
        st.subheader("Emotion Detection by Text                      - Vishnu Pandey")
        with st.form(key='emotion_clf_form'):
                input_text = st.text_area("Type Here")
                output_text = st.form_submit_button(label='Submit')

        if output_text:
                col1,col2=st.columns(2)

                prediction=predict_emotion(input_text)
                probability=predict_probability(input_text)
                

                with col1:
                        st.success("Text")
                        st.write(input_text)
                        st.success("Prediction")
                        emoji_icon = emotions_emoji_dict[prediction]
                        st.write("{}:{}".format(prediction,emoji_icon))
                       
                with col2:
                        st.success("Prediction Probability")
                        st.write(probability)
                        proba_df = pd.DataFrame(probability,columns=model.classes_)
                        proba_df_clean = proba_df.T.reset_index()
                        proba_df_clean.columns = ["emotions","probability"]

                        fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
                        st.altair_chart(fig,use_container_width=True)
                                
                

if __name__ == '__main__':
	main()
