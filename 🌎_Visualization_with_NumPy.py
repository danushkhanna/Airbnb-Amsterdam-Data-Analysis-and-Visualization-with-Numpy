import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

# Load the image
image = Image.open('airbnblogo.svg.png')
st.image(image, width=150)

st.write("<div align='left'><span style='color:#FF5A5F; font-size: 15px;'>Amsterdam</span></div>", unsafe_allow_html=True)

st.markdown(
    "<div style='display: flex; align-items: left;'>"
    "<h1 style='color:#484848; margin-right: -40px'>Visualization with</h1>"
    "<h1 style='color:#FF5A5F;'>NumPy 🗺️</h1>"
    "</div>", 
    unsafe_allow_html=True
)

image=Image.open('amsterdam.png')
width=750
height=500
image_new=image.resize((width,height))
st.image(image_new)

st.markdown("Welcome to our charming Airbnb! We're thrilled to introduce you to our space that is the perfect getaway for those seeking a comfortable and peaceful stay. \n\nOur home is designed to cater to all your needs, and we take pride in providing our guests with an exceptional experience. As you step inside, you'll be welcomed by a cozy atmosphere that exudes warmth and relaxation. \n\nOne of the standout features of our Airbnb is the stunning visualization with NumPy. This unique feature allows you to explore different areas and visualize them on a map, giving you an immersive experience that you won't find anywhere else. \n\nWhether you're a solo traveler or a group of friends, our space is ideal for those looking for a comfortable and memorable stay. So why wait? Book your stay with us now and get ready to experience the best that Airbnb has to offer! \n\nHere we can see the dataframe created during this week's project.")

# AN EDIT BY DANUSH KHANNA
if st.sidebar.button("An edit by Danush Khanna"):
    js = "window.open('https://www.linkedin.com/in/danush-khanna-ba4240239/')"
    html = '<img src onerror="{}">'.format(js)
    st.sidebar.markdown(html, unsafe_allow_html=True)


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
