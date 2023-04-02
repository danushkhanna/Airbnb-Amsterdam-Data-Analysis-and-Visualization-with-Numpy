import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

# Load the image
image = Image.open('airbnblogo.png')

# Create a 2-column layout
col1, col2 = st.beta_columns([2, 1])

# Display the image in the first column
col1.image(image, use_column_width=True)

# Calculate the position of the caption
image_width, _ = image.size
caption_position = int(image_width / 2)

# Display the caption in the second column, below the middle of the image
with col2:
    st.write("<div align='center'><span style='color:#FF5A5F; font-size: 24px;'>Amsterdam</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center'><span style='font-size: 30px;'>&#x25BC;</span></div>", unsafe_allow_html=True)

# Display title and text
st.markdown(
    f"<h1 style='color:#FF5A5F;'>Visualization with NumPy 🗺️</h1>", 
    unsafe_allow_html=True
)
st.markdown("Here we can see the dataframe created during this weeks project.")

st.sidebar.success("Select a demo above.")

# Read dataframe
dataframe = pd.read_csv(
    "WK1_Airbnb_Amsterdam_listings_proj_solution.csv",
    names=[
        "Airbnb Listing ID",
        "Price",
        "Latitude",
        "Longitude",
        "Meters from chosen location",
        "Location",
    ],
)

# We have a limited budget, therefore we would like to exclude
# listings with a price above 100 pounds per night
dataframe = dataframe[dataframe["Price"] <= 100]

# Display as integer
dataframe["Airbnb Listing ID"] = dataframe["Airbnb Listing ID"].astype(int)
# Round of values
dataframe["Price"] = "₹ " + dataframe["Price"].round(2).astype(str) # <--- CHANGE THIS POUND SYMBOL IF YOU CHOSE CURRENCY OTHER THAN POUND
# Rename the number to a string
dataframe["Location"] = dataframe["Location"].replace(
    {1.0: "To visit", 0.0: "Airbnb listing"}
)

# Display dataframe and text
st.dataframe(dataframe)
st.markdown("Below is a map showing all the Airbnb listings with a red dot and the location we've chosen with a blue dot.")

# Create the plotly express figure
fig = px.scatter_mapbox(
    dataframe,
    lat="Latitude",
    lon="Longitude",
    color="Location",
    color_discrete_sequence=["blue", "red"],
    zoom=11,
    height=500,
    width=800,
    hover_name="Price",
    hover_data=["Meters from chosen location", "Location"],
    labels={"color": "Locations"},
)
fig.update_geos(center=dict(lat=dataframe.iloc[0][2], lon=dataframe.iloc[0][3]))
fig.update_layout(mapbox_style="stamen-terrain")

# Show the figure
st.plotly_chart(fig, use_container_width=True)
