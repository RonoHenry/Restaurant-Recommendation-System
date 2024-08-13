import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deployment.app_classes import recommendation, pagenation, get_business_image_urls


# Load your restaurant data
# @st.cache_data
df = pd.read_pickle('pickled_files/restaurant_data.pkl')

# Assuming your DataFrame is named 'df' and has a column named 'state'
states_to_drop = ['Monatana', 'North Carolina']

# Drop rows where the 'state' column contains 'Montana' or 'Delaware'
df = df[~df['state'].isin(states_to_drop)]



def render_home_page():
    # Initialize session state variables
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = pd.DataFrame()
    if 'selected_restaurant' not in st.session_state:
        st.session_state.selected_restaurant = None

    st.markdown(
        """
        <style>
        .center-text {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<h2 class="center-text" style="color: royal blue;">GOURMET GURUS</h2>', unsafe_allow_html=True)
    st.markdown('<h3 class="center-text" style="color: royal blue;">A Revolution In Restaurant Recommendation</h3>', unsafe_allow_html=True)
    st.write(" ")

    # Sidebar for user inputs
    st.sidebar.header('Search for a Restaurant')

    # Get unique states, cities, and restaurant names and sort them
    states = sorted(df['state'].unique())

    # State selection
    state = st.sidebar.selectbox('Choose Your State:', ['All'] + states)

    # Filter cities based on selected state
    if state == 'All':
        filtered_cities = sorted(df['city'].unique())
    else:
        filtered_cities = sorted(df[df['state'] == state]['city'].unique())

    # City selection
    city = st.sidebar.selectbox('Choose Your City', ['All'] + filtered_cities)

    if city != 'All':
        # Option to either choose a restaurant name or enter a category
        search_option = st.sidebar.radio(
            "Search by",
            ('Restaurant Name', 'Cuisine')
        )

        if search_option == 'Restaurant Name':
            # Filter restaurant names based on selected state and city
            restaurants = sorted(df[(df['state'] == state) & (df['city'] == city)]['name'].unique())
            restaurant_name = st.sidebar.selectbox('Restaurant Name', ['All'] + restaurants)
        elif search_option =='Cuisine':
            # Extract unique categories and create a selectbox
            categories = sorted(df[(df['state'] == state) & (df['city'] == city)]['categories'].unique())
            selected_category = st.sidebar.selectbox("Choose a Cuisine", ['All'] + categories)
            # restaurant_name = selected_category

        # Add a "Recommend" button
        recommend_button = st.sidebar.button('Recommend')

        if recommend_button:
            if search_option == 'Restaurant Name' and restaurant_name != 'All':
                # Use content-based filtering to find similar restaurants by name
                st.session_state.recommendations = recommendation(df, state=state, name= restaurant_name)
            elif search_option == 'Cuisine' and selected_category != 'All':
                # Use content-based filtering to find restaurants by category
                st.session_state.recommendations = recommendation(df, state=state, category=selected_category)
            else:
                st.write("Please select a valid restaurant name or cuisine to get recommendations.")

    # Display the filtered DataFrame with selected columns
    if not st.session_state.recommendations.empty:
        st.spinner("COOKING SOMETHING UP FOR YOU!!!!!")
        st.write('### Recommended Restaurants')
        
        # Paginate the filtered recommendations
        filtered_df = pagenation(st.session_state.recommendations, 'city')

        # Add a dropdown to select a restaurant from the filtered DataFrame on the current page
        if not filtered_df.empty:
            selected_restaurant = st.selectbox(
                'Select a Restaurant to View Details',
                options=['Select a Restaurant'] + list(filtered_df['name']),
                index=0
            )
        # Update the session state for the selected restaurant
        st.session_state.selected_restaurant = selected_restaurant

        if st.session_state.selected_restaurant != 'Select a Restaurant':
            info = df[df['name'] == st.session_state.selected_restaurant].iloc[0]
            with st.container(border=True):
                st.subheader("Restaurant Information", divider=True)
                col1, col2 = st.columns(2)
                with col1:
                    
                    
                    st.write(f"**Restaurant Name:** {info['name']}")
                    st.write(f"**State:** {info['state']}")
                    st.write(f"**City:** {info['city']}")
                    st.write(f"**Address:** {info['address']}")
                    st.write(f"**Cuisine:** {info['categories']}")
                    st.write(f"**Rating:** {info['stars']}")
                    st.write(f"**Review Count:** {info['review_count']}")
                    
                    
                    

                    # st.write(f"**BUsiness_id:** {info['business_id']}")
                    
                with col2:
                    
                    if 'latitude' in info and 'longitude' in info:
                        latitude = info['latitude']
                        longitude = info['longitude']

                        if pd.notna(latitude) and pd.notna(longitude):
                            # Create the Folium map centered on the marker's location
                            m = folium.Map(location=[latitude, longitude], zoom_start=18, dragging=True, zoom_control=False,
                                        scrollWheelZoom=False)

                            # Add the marker to the map
                            folium.Marker(
                                [latitude, longitude], popup=info['name'], tooltip=info['name']
                            ).add_to(m)

                            # Display the map in Streamlit
                            st_folium(m, width=700, height=300)

                            # Create a link for directions
                            route = f"http://maps.google.com/maps?z=12&t=m&q=loc:{latitude}+{longitude}"
                            st.button(f"[Get Directions]({route})")




                    else:
                        st.write("Location data is not available for this restaurant.")
                
                with st.expander("Restaurant Specialities", expanded=False):
                    image_urls = get_business_image_urls(info["business_id"])
                    if image_urls:
                        # Create columns for the images
                        num_cols = 3  # Number of columns
                        cols = st.columns(num_cols)
                        for i, url in enumerate(image_urls):
                            # Determine which column to place the image in
                            col_index = i % num_cols
                            with cols[col_index]:
                                st.image(url, use_column_width=True)
                    else:
                        st.write("No images available for this restaurant.")