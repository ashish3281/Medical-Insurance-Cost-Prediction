import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


loaded_model = pickle.load(open('insur_model.sav', 'rb'))

def insurance_predicton(input_data):


# changing the input_data to numpy array
   input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
   input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)



   prediction = loaded_model.predict(input_data_reshaped)
   
   return 'The insurance cost in USD ',prediction[0] 
   
  
  
def main():

    st.title("Medical Insurance Prediction Model")
    st.text("Please fill the following details")

    age = st.text_input("Age")
    sex = st.text_input("Sex")
    bmi = st.text_input("BMI Value")
    children = st.text_input("Children")
    smoker = st.text_input("Smoker")
    region = st.text_input("Region")
    
    

    diagnosis = ""
    if st.button("Medical Insurance Test Result"):
        diagnosis = insurance_predicton([age, sex, bmi, children, smoker, region])
        # process input_data and make a prediction
       

    st.success(diagnosis)
    
    
    
    def load_data(nrows):
        data=pd.read_csv('insurance.csv',nrows=nrows)
        return data
    data_list=load_data(1000)
    
    st.subheader('Data set Use')
    st.write(data_list)
    # st.bar_chart(data_list['lables'])
    df=pd.DataFrame(data_list[:200],columns=['Density', 'age'])
    st.line_chart(df)
    df=pd.DataFrame(data_list[:200],columns=['count', 'sex'])
    st.line_chart(df)
    df=pd.DataFrame(data_list[:200],columns=['Density', 'bmi'])
    st.line_chart(df)
    df=pd.DataFrame(data_list[:200],columns=['Density', 'charges'])
    st.line_chart(df)
    df=pd.DataFrame(data_list[:200],columns=['count', 'children'])
    st.line_chart(df)
    df=pd.DataFrame(data_list[:200],columns=['count', 'smoker'])
    st.line_chart(df)
    df=pd.DataFrame(data_list[:200],columns=['count', 'region'])
    st.line_chart(df)
    df=pd.DataFrame(data_list[:200],columns=['Density', 'bmi'])
    st.line_chart(df)
    # Density vs Charges histogram
    # charges = insurance_df["charges"]
    # plt.hist(charges, bins=50)
    # plt.xlabel("Charges")
    # plt.ylabel("Density")
    # st.pyplot()
    
  
    
  
   



    
   

if __name__ == "__main__":
    main()
