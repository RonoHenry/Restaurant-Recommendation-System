import streamlit as st
import pandas as pd
from deployment.app_about import about_us
from streamlit_option_menu import option_menu
from deployment.home import render_home_page


st.set_page_config(
    page_title="Gourmet Gurus App",
    page_icon="	:gear:",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/AtomHarris/Waste-Image-Classification',
        'Report a bug': 'https://github.com/AtomHarris/Waste-Image-Classification',
        'About': "# Here to help with all your waste issues!"
    }
)
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


