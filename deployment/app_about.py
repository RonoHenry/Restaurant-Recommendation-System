import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from app_feedback import display_contact_info,contact_form



def about_us():
    # Custom CSS to reduce the size of the tabs
    st.markdown(
        """
        <style>
        .nav-pills .nav-link {
            font-size: 12px;
            padding: 5px 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    selected1 = option_menu(
        menu_title=None,
        options=["Company History", "Our Team", "Feedback"],
        icons=["bookmark", "pencil-square", "tools"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    if selected1 == "Company History":
        st.header('Company History', divider=True)
        st.write("")

    elif selected1 == "Our Team":
        st.header('Our Team', divider=True)
        display_contact_info()

    elif selected1 == "Feedback":
        st.header('Feedback', divider=True)
        contact_form()


