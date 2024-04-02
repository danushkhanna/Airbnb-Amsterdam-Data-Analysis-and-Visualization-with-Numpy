


#

%%capture
!pip install numpy streamlit pandas==1.5.2 scikit-learn==1.2.0


#



!wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1KTF77Sj0kWyft9gNT3_6k84gauPA95rG' -O listings.pkl



#



import pandas as pd

# Show all columns (instead of cascading columns in the middle)
pd.set_option("display.max_columns", None)
# Don't show numbers in scientific notation
pd.set_option("display.float_format", "{:.2f}".format)



#


df_list = pd.read_pickle("listings.pkl")

#

df_list.head(2)

#

df_list.info()

#

df_list = df_list.drop(columns=["id", "host_acceptance_rate", "host_is_superhost", "has_availability", "number_of_reviews_l30d",
                                "discount_per_5_days_booked", "discount_per_10_days_booked", "discount_per_30_and_more_days_booked", "service_cost"],
                       axis=1)


#


df_list.info()



#


X, y = (
    df_list[["neighbourhood", "room_type", "host_reported_average_tip", "amenities",
             "accommodates", "review_scores_rating", "instant_bookable"]],
    df_list[["price_in_dollar"]],
)



#


SEED = 42


#


import sklearn
from sklearn.model_selection import train_test_split

def train_validation_test_split(
    X, y, train_ratio: float, validation_ratio: float, test_ratio: float
):
    # Split up dataset into train and test, of which we split up the test
    X_train, X_test, y_train, y_test = train_test_split(
        ... # YOUR CODE HERE, SURELY ADD A SEED
    )

    # Split up test into two (validation and test)
    X_val, X_test, y_val, y_test = train_test_split(
        ... # YOUR CODE HERE, SURELY ADD A SEED
    )

    # Return the splits
    return X_train, X_val, X_test, y_train, y_val, y_test



#




X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(
    X, y, 0.75, 0.15, 0.1
)


#



X.shape, X_train.shape, X_val.shape, X_test.shape


#


X.info()


#


X_train["instant_bookable"] = X_train["instant_bookable"].astype(int)
X_val["instant_bookable"] = X_val["instant_bookable"].astype(int)
X_test["instant_bookable"] = X_test["instant_bookable"].astype(int)


#


X_train.head(2)


#



# One-hot encoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

# Define how the encoding should work
oh_encoder = OneHotEncoder(  # Define one-hot encoding
    sparse_output=False,  # Sparse matrix doesn't work well with Pandas DataFrame
    dtype="int",  # Set type to integer
)

# Define which columns to transform
oh_enc_transformer = make_column_transformer(  # Define how to output columns
    (oh_encoder, ... ), # YOUR CODE HERE
    verbose_feature_names_out=False,  # Column names are "more concise"
    remainder="passthrough",  # All other columns should be left untouched
)

# Train (fit) the transformation on the training set
oh_encoded = oh_enc_transformer.fit(X_train)  # Change from category to number


#



# Transform the train columns into one-hot encoding
X_train_oh_enc = oh_encoded.transform(X_train)

# Turn the encoded columns into a df
X_train = pd.DataFrame(
    X_train_oh_enc,  # Input the transformed dataset
    columns=oh_encoded.get_feature_names_out(),  # Set column names
    index=X_train.index,  # Keep index numbering of original df
)



#



X_val_oh_enc = oh_encoded.transform(X_val)

# Turn the encoded columns into a dataframe.
X_val = pd.DataFrame(
    X_val_oh_enc,  # Input the transformed dataset
    columns=oh_encoded.get_feature_names_out(),  # Set column names
    index=X_val.index,
)


#


# Transform the columns into one-hot-encoding.
X_test_oh_enc = oh_encoded.transform(X_test)

# Turn the encoded columns into a dataframe.
X_test = pd.DataFrame(
    X_test_oh_enc,  # Input the transformed dataset
    columns=oh_encoded.get_feature_names_out(),  # Set column names
    index=X_test.index,  # Keep index numbering of original df
)



#



X_train = X_train.convert_dtypes()
X_val = X_val.convert_dtypes()
X_test = X_test.convert_dtypes()



#



# Show how the dataframe looks like
X_train.head(2)



#



X_train.info()



#



# Correlation - what are the best features?
import numpy as np
import plotly.express as px

# Exclude "neighbourhood" colums for better visualization.
X_train_filtered = X_train.filter(regex="^((?!neighbourhood).)*$")

# combine X_train with Y_train
ndf_list = pd.concat([X_train_filtered, y_train], axis=1)

# Create a dataframe that can be used as a heatmap
fig = px.imshow(
    ndf_list.corr().round(2),
    text_auto=True,
    aspect="auto",
    color_continuous_scale="rdylgn",
)
fig.show()



#




# Splom
import plotly.express as px

fig = px.scatter_matrix(ndf_list, height=1200)
fig.show()



#




# Get the algorithm
from sklearn.linear_model import LinearRegression

# Create a regression algorithm
model = LinearRegression()



#



# Fit the model - Pick one feature "amenities"
model.fit(  # Train it ("Learn the material")
    X_train[["amenities"]],
    np.squeeze(y_train),
)



#



from sklearn.metrics import r2_score



#



y_predict = model.predict(X_val[["amenities"]])



#



r2_score(y_predict, y_val).round(4)



#



import plotly.express as px
import plotly.graph_objects as go
updatedX = np.array(X_val['amenities'])

reference_line = go.Scatter(x=updatedX,
                            y=y_predict,
                            mode="lines",
                            line=go.scatter.Line(color="gray"),
                            showlegend=False)

fig = px.scatter(x=X_val['amenities'], y=y_val['price_in_dollar'])

fig.add_trace(reference_line)

fig.show()



#



model.fit(  # Train it ("Learn the material")
    X_train["accommodates", "room_type_Shared room"]],
    np.squeeze(y_train),
)

# Predict
y_predict = model.predict(X_val[["accommodates", "room_type_Shared room"]])  # Do a "final exam"

# Score
# Compare algorithms' "final exam" vs. expected.
r2_score(y_predict, y_val).round(4)



#



# Fit, Predict, Score
model.fit(  # Train it ("Learn the material")
    X_train[["amenities", "accommodates", "instant_bookable"]],
    np.squeeze(y_train),
)

# Predict
y_predict = model.predict(X_test[["amenities", "accommodates", "instant_bookable"]])  # Do a "final exam"

# Score
# Compare algorithms' "final exam" vs. expected.
r2_score(y_predict, y_test).round(4)




#



example = [[20, 4, 1]]

model.predict(example).round(2)


#



import pickle

pickle.dump(model, open("model.pkl", "wb"))


#



from google.colab import files

# Download the file locally
files.download('model.pkl')



#


%%writefile streamlit_app.py
import streamlit as st
import sklearn
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.title("Week 3: The Airbnb dataset of Amsterdam")
st.markdown(
    "The dataset contains modifications with regards to the original for illustrative & learning purposes"
)

amenities = st.slider('How many amenities does the listing have?', 0, 50, 20)
accommodates = st.slider('How many people does the listing accommodate?', 1, 16, 2)
instant_bookable = st.radio(
    "Is the listing instantly bookable?",
    ("True", "False"))
instant_bookable = 1 if instant_bookable == "True" else 0

user_input = [[amenities, accommodates, instant_bookable]]

if st.button('Predict?'):
    st.write("The model predicts that the average tip for this listing is:", model.predict(user_input).round(2))



#
    


from google.colab import files

# Download the file locally
files.download('streamlit_app.py')



#



%%writefile requirements.txt
streamlit
pandas
scikit-learn



#


from google.colab import files

# Download the file locally
files.download('requirements.txt')

#



 


