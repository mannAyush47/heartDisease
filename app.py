import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from PIL import Image

# Load the dataset
df = pd.read_csv('dataset.csv')
st.markdown(
    """
    <style>
    table.dataframe {
        border-collapse: collapse;
        border: 2px solid red;
    }
     .stApp {
        background-color: white;
    }
    .stSidebar {
        background-color: red;
    }
    table.dataframe th {
        border: 2px solid red;
    }
    table.dataframe td {
        border: 2px solid red;
    }
    .css-10trblm{
    color:white
    }
    
    .css-1544g2n{
    padding: 6rem 1rem 1.5rem;
    background-color: aqua;
    }
    .css-1ec096l {
    font-size: 1.4rem;
    font-family: "Source Sans Pro", sans-serif;
    padding: 0.25rem 0.375rem;
    line-height: 1.5;
    overflow: overlay;
    # border: 8px solid firebrick;
    border-radius: 12px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.4);

    }
    .css-1vbkxwb {
    font-family: "Source Sans Pro", sans-serif;
    width: 15rem;
}
.css-1n543e5 {
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 600;
    padding: 0.25rem 0.75rem;
    border-radius: 0.25rem;
    /* margin: 0px; */
    line-height: 1.6;
    width: auto;
    user-select: none;
    background-color: #FF52A2;
    # border: 5px solid rgba(49, 51, 63, 0.2);
    margin-left: 30%;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.4);

    margin-top: 15%;
    border-radius: 33px;
    color: white;
}
    
element.style {
}
<style>
.css-5rimss p {
    word-break: break-word;
}
<style>
p,ol,ul,dl {
    margin: 0px 0px 1rem;
    padding: 0px;
    font-size: 3rem;
    font-weight: 400;
}
    .css-1544g2n {
    padding: 6rem 1rem 1.5rem;
     background-color: #FF52A2; 
    # background-image: linear-gradient(120deg, #f857a6 10%, #ef3f6e);
}

.css-11vi4b2 {
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 0.25rem;
    /* margin: 0px; */
    line-height: 1.6;
    color: inherit;
    width: 32rem;
    margin-top: 53px;
    margin-bottom: 30px;
    background-color: red;
    #: ;
    border: 6px solid white;
    height: 5rem;
    border-radius: 20px;
    margin-left: 13%;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.4);
}
.css-18ni7ap {
    position: fixed;
    top: 0px;
    left: 0px;
    right: 0px;
    height: 2.875rem;
    background: #FF52A2;
    outline: none;
    z-index: 999990;
    display: block;
}
.st-dy {
    background: white;
}
.css-5rimss {
    font-family: "Source Sans Pro", sans-serif;
    margin-bottom: -1rem;
    color: darkcyan;
    font-size: 2rem;
    margin-top: 3%;
    text-underline-offset: auto;
}
p, ol, ul, dl {
    margin: 0px 0px 1rem;
    padding: 0px;
     font-size: 2rem; 
    font-weight: 400;
}
    </style>
    """,
    unsafe_allow_html=True
)


# Data preprocessing
df['cp'] = df['cp'].astype(str)  # Convert 'cp' column to string
df['cp'] = df['cp'].apply(lambda x: x.split()[0])  # Keep only the first value of 'cp' column
df['ca'] = df['ca'].replace({'0.0': '0'})  # Replace '0.0' with '0' in 'ca' column
df['ca'] = df['ca'].astype(float).astype(int).astype(str)  # Convert 'ca' column to int and then to string
df['slope'] = df['slope'].astype(str)  # Convert 'slope' column to string
df['sex'] = df['sex'].astype(str)  # Convert 'sex' column to string
df['thal'] = df['thal'].apply(lambda x: x.split()[0] if isinstance(x, str) and ' ' in x else x)  # Keep only the first value of 'thal' column if a space is present and the value is of type string
df['fbs'] = df['fbs'].astype(str)  # Convert 'fbs' column to string

dataset = pd.get_dummies(df, columns=['restecg', 'exang'])
scaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])

# Split the dataset into features (X) and target variable (y)
y = dataset['target']
X = dataset.drop(['target'], axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = np.array(X_test)

# Create the web app using Streamlit
import streamlit as st

# Create the web app using Streamlit
# Create the web app using Streamlit
def main():
    st.title("Heart Disease Classification")
    st.sidebar.title("Options")
    st.write("Please fill the form below to the status of your heart based on your report")  # Adding the heading above the form

    classifier = st.sidebar.selectbox("Select Classifier", ("Decision Tree","Random Forest"))

    if classifier == "K-nearest Neighbors":
        k = st.sidebar.slider("Select the number of neighbors (K)", 1, 20, 12)
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X_train, y_train)

        # Get custom inputs for each column
        custom_inputs = {}
        for column in X.columns:
            if column == 'sex':
                sex_options = ['Female', 'Male']
                sex_index = st.sidebar.selectbox("Select Sex", sex_options, index=0)
                custom_inputs[column] = 1 if sex_index == 'Male' else 0
            else:
                custom_inputs[column] = st.sidebar.number_input(f"Enter {column}", value=0)

        # Create a DataFrame with the custom inputs
        custom_data = pd.DataFrame([custom_inputs])

        # Initialize all rows with 0
        custom_data = custom_data.fillna(0)

        # Preprocess the custom data
        custom_data[columns_to_scale] = scaler.transform(custom_data[columns_to_scale])


        # Display the input values
        # st.subheader("Input Values")
        # st.table(custom_data.T.style.set_properties(**{'height': '20px'}))

        col3,col4 = st.columns(2)
        with col3:
          st.subheader("Input Values")
          st.table(custom_data.T.style.set_properties(**{'height': '20px'}))
        with col4:
           # Display the image
           image = Image.open("doc.gif")
           # st.image(image, caption='Your Image Caption', use_column_width=True)
           st.image("doc.gif", caption='Your Image Caption', use_column_width=True)


           # Submit button
        if st.button("Submit"):


            y_pred = knn_classifier.predict(custom_data)
            y_pred = knn_classifier.predict(custom_data)
            if y_pred == 1:
                st.write(
                    "")
                # Display the image
                image = Image.open("emergency.png")
                # Reduce the size of the image to 50x50 pixels
                resized_image = image.resize((350, 350))

                # Display the resized image and the prediction text side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(resized_image, caption='Your Image Caption' )
                with col2:
                    st.write(
                        "Prediction: There are chances of heart disease. Please consider consulting your doctor for further help.")
            else:
                st.write("")
                # Display the image
                image = Image.open("thums.png")
                # Reduce the size of the image to 50x50 pixels
                resized_image = image.resize((350, 350))

                # Display the resized image and the prediction text side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(resized_image, caption='Your Image Caption')
                with col2:
                    st.write("Prediction: There are no chances of heart disease.")

            # Calculate and display the accuracy
            accuracy = accuracy_score(y_test, knn_classifier.predict(X_test))
            st.write("With Accuracy of:", accuracy*100 , "%")

    elif classifier == "Decision Tree":
        dt_classifier = DecisionTreeClassifier()
        dt_classifier.fit(X_train, y_train)

        # Get custom inputs for each column
        custom_inputs = {}
        for column in X.columns:
            if column == 'sex':
                sex_options = ['Female', 'Male']
                sex_index = st.sidebar.selectbox("Select Sex", sex_options, index=0)
                custom_inputs[column] = 1 if sex_index == 'Male' else 0
            else:
                custom_inputs[column] = st.sidebar.number_input(f"Enter {column}", value=0)

        # Create a DataFrame with the custom inputs
        custom_data = pd.DataFrame([custom_inputs])

        # Initialize all rows with 0
        custom_data = custom_data.fillna(0)

        # Preprocess the custom data
        custom_data[columns_to_scale] = scaler.transform(custom_data[columns_to_scale])

        # Display the image
        # image = Image.open("emergency.png")
        # resized_image = image.resize((100, 100))

        # st.image(image, caption='Your Image Caption', use_column_width=True)

        # Display the input values
        # st.subheader("Input Values")
        # st.table(custom_data.T.style.set_properties(**{'height': '20px'}))

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Input Values")
            st.table(custom_data.T.style.set_properties(**{'height': '20px'}))
        with col4:
            # Display the image
            image = Image.open("doc.gif")
            # st.image(image, caption='Your Image Caption', use_column_width=True)
            st.image("doc.gif", caption='Your Image Caption', use_column_width=True)
        # Submit button
        if st.button("Submit"):
            # Predict using the Decision Tree classifier
            y_pred = dt_classifier.predict(custom_data)
            if y_pred == 1:
                st.write(
                    "")
                # Display the image
                image = Image.open("emergency.png")
                # Reduce the size of the image to 50x50 pixels
                resized_image = image.resize((350, 350))

                # Display the resized image and the prediction text side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(resized_image, caption='Your Image Caption')
                with col2:
                    st.write(
                        "Prediction: There are chances of heart disease. Please consider consulting your doctor for further help.")
            else:
                st.write("")
                # Display the image
                image = Image.open("thums.png")
                # Reduce the size of the image to 50x50 pixels
                resized_image = image.resize((350, 350))

                # Display the resized image and the prediction text side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(resized_image, caption='Your Image Caption')
                with col2:
                    st.write("Prediction: There are no chances of heart disease.")

            # Calculate and display the accuracy
            accuracy = accuracy_score(y_test, dt_classifier.predict(X_test))
            st.write("With Accuracy of:", accuracy*100 , "%")

    elif classifier == "Random Forest":
        randomforest_classifier = RandomForestClassifier(n_estimators=10)
        randomforest_classifier.fit(X_train, y_train)

        # Get custom inputs for each column
        custom_inputs = {}
        for column in X.columns:
            if column == 'sex':
                sex_options = ['Female', 'Male']
                sex_index = st.sidebar.selectbox("Select Sex", sex_options, index=0)
                custom_inputs[column] = 1 if sex_index == 'Male' else 0
            else:
                custom_inputs[column] = st.sidebar.number_input(f"Enter {column}", value=0)

        # Create a DataFrame with the custom inputs
        custom_data = pd.DataFrame([custom_inputs])

        # Initialize all rows with 0
        custom_data = custom_data.fillna(0)

        # Preprocess the custom data
        custom_data[columns_to_scale] = scaler.transform(custom_data[columns_to_scale])

        # Display the image
        # image = Image.open("mreport.jpg")
        # st.image(image, caption='Your Image Caption', use_column_width=True)

        # Display the input values
        # st.subheader("Input Values")
        # st.table(custom_data.T.style.set_properties(**{'height': '20px'}))

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Input Values")
            st.table(custom_data.T.style.set_properties(**{'height': '20px'}))
        with col4:
            # Display the image
            image = Image.open("doc.gif")
            # st.image(image, caption='Your Image Caption', use_column_width=True)
            st.image("doc.gif", caption='Your Image Caption', use_column_width=True)
        # Submit button
        if st.button("Submit"):
            # Predict using the Random Forest classifier
            y_pred = randomforest_classifier.predict(custom_data)
            if y_pred == 1:
                st.write(
                    "")
                # Display the image
                image = Image.open("emergency.png")
                # Reduce the size of the image to 50x50 pixels
                resized_image = image.resize((350, 350))

                # Display the resized image and the prediction text side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(resized_image, caption='Your Image Caption')
                with col2:
                    st.write(
                        "Prediction: There are chances of heart disease. Please consider consulting your doctor for further help.")
            else:
                st.write("")
                # Display the image
                image = Image.open("thums.png")
                # Reduce the size of the image to 50x50 pixels
                resized_image = image.resize((350, 350))

                # Display the resized image and the prediction text side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(resized_image, caption='Your Image Caption')
                with col2:
                    st.write("Prediction: There are no chances of heart disease.")

            # Calculate and display the accuracy
            accuracy = accuracy_score(y_test, randomforest_classifier.predict(X_test))
            st.write("With Accuracy of:", accuracy*100 , "%")
            # Define the footer content
            footer_text = "This is the footer of the application."


if __name__ == "__main__":
    main()
