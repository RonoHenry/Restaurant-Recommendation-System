import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
#from deployment import home, app_about
from deployment.home import render_home_page
from deployment.app_about import about_us

# Company Logo
st.sidebar.image('Images/image5.png', use_column_width=True)



# Home Page Option Menu Initialization
selected = option_menu(
    menu_title=None,
    options=["Restaurant Name", "Cuisines", "About Us"],
    icons=["house", "award", "three-dots"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)


if selected == "Restaurant Name":
    render_home_page()


elif selected == "Cuisines":
    st.write("coming soon...")

elif selected == "About Us":
    about_us()


