
import streamlit as st
import pickle

# Load the pre-trained model and CountVectorizer
model_filename = 'nlp_model.pkl'
cv_filename = 'transform.pkl'

clf = pickle.load(open(model_filename, 'rb'))
cv = pickle.load(open(cv_filename, 'rb'))

# Title of the web app
st.title('Spam Detector')

# Text input for the message
message = st.text_area('Enter a message:')

# Predict button
if st.button('Predict'):
    if message:
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        
        # Display the prediction result
        if my_prediction[0] == 1:
            st.write('The message is *spam*.')
        else:
            st.write('The message is *not spam*.')
    else:
        st.write('Please enter a message to predict.')
