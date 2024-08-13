import numpy as np
import streamlit as st
import pandas as pd

from streamlit_folium import st_folium
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load your restaurant data
df = pd.read_pickle(pickled_files/restaurant_data.pkl)

def preprocess(df):
    # Combine text columns into a single feature
    df['combined_features'] = df['state'] + " " + df['categories'] + " " + df['city']
    return df

def create_feature_vectors(df):
    # Vectorize the combined text features
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    # Combine the TF-IDF matrix with numerical columns
    numerical_features = df[['stars', 'review_count', 'latitude', 'longitude']].values
    combined_features = np.hstack((tfidf_matrix.toarray(), numerical_features))
    
    return combined_features

def content_based_filtering(df, query, num=5, by='name'):
    df = preprocess(df)
    
    if by == 'name':
        # Find the index of the restaurant by name
        idx = df.index[df['name'].str.lower() == query.lower()][0]
        features = create_feature_vectors(df)
    elif by == 'category':
        # Filter the DataFrame to include only the relevant category
        df_filtered = df[df['categories'].str.contains(query, case=False)]
        if df_filtered.empty:
            return f"No restaurants found with category: {query}"
        
        # Recreate the feature vectors for the filtered DataFrame only
        features = create_feature_vectors(df_filtered)
        
        # For the category case, set the index to 0 since we're only considering the filtered DataFrame
        idx = 0
    
    # Compute the cosine similarity within the relevant subset
    cosine_sim = cosine_similarity(features[idx].reshape(1, -1), features).flatten()
    
    # Get the indices of the most similar restaurants
    similar_indices = cosine_sim.argsort()[-(num+1):][::-1]  # Exclude the first one since it's the restaurant itself
    
    # Get the top N similar restaurants, accounting for the index offset in the filtered DataFrame case
    top_restaurants = df_filtered.iloc[similar_indices[1:num+1]][['name', 'stars', 'review_count', 'address', 'city', 'state', 'categories']] if by == 'category' else df.iloc[similar_indices[1:num+1]][['name', 'stars', 'review_count', 'address', 'city', 'state', 'categories']]
    
    return top_restaurants

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
            ('Restaurant Name', 'Category')
        )

        if search_option == 'Restaurant Name':
            # Filter restaurant names based on selected state and city
            restaurants = sorted(df[(df['state'] == state) & (df['city'] == city)]['name'].unique())
            restaurant_name = st.sidebar.selectbox('Restaurant Name', ['All'] + restaurants)
        else:
            # Extract unique categories and create a selectbox
            categories = sorted(df[(df['state'] == state) & (df['city'] == city)]['categories'].unique())
            
            selected_category = st.sidebar.selectbox("Choose a Cuisine", ['All'] + categories)
            restaurant_name = selected_category

        # Add a "Recommend" button
        recommend_button = st.sidebar.button('Recommend')

        if recommend_button:
            if search_option == 'Restaurant Name' and restaurant_name != 'All':
                # Use content-based filtering to find similar restaurants by name
                st.session_state.recommendations = content_based_filtering(df, restaurant_name, by='name')
            elif search_option == 'Category' and restaurant_name != 'All':
                # Use content-based filtering to find restaurants by category
                st.session_state.recommendations = content_based_filtering(df, restaurant_name, by='category')
            else:
                st.write("Please select a valid restaurant name or category to get recommendations.")

    # Display the filtered DataFrame with selected columns
    if not st.session_state.recommendations.empty:
        st.write('### Recommended Restaurants')
        st.dataframe(st.session_state.recommendations)

        # Add a dropdown to select a restaurant from recommendations
        selected_restaurant = st.selectbox(
            'Select a Restaurant to View Details',
            options=['Select a Restaurant'] + list(st.session_state.recommendations['name']),
            index=0
        )

        # Update the session state for the selected restaurant
        st.session_state.selected_restaurant = selected_restaurant

        if st.session_state.selected_restaurant != 'Select a Restaurant':
            info = st.session_state.recommendations[st.session_state.recommendations['name'] == st.session_state.selected_restaurant].iloc[0]

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Restaurant Name:** {info['name']}")
                st.write(f"**Cuisine:** {info['categories']}")
                st.write(f"**Rating:** {info['stars']}")
                st.write(f"**Review Count:** {info['review_count']}")
                st.write(f"**Address:** {info['address']}")
                st.write(f"**City:** {info['city']}")
                st.write(f"**State:** {info['state']}")
                
            with col2:
                if 'latitude' in info and 'longitude' in info:
                    latitude = info['latitude']
                    longitude = info['longitude']

                    if pd.notna(latitude) and pd.notna(longitude):
                        m = folium.Map(location=[latitude, longitude], zoom_start=18, dragging=False, zoom_control=False,
                                    scrollWheelZoom=True)
                        folium.Marker(
                            [latitude, longitude], popup=info['name'], tooltip=info['name']
                        ).add_to(m)
                        st_folium(m, width=700, height=400)

                        route = f"http://maps.google.com/maps?z=12&t=m&q=loc:{latitude}+{longitude}"
                        st.markdown(f"[Get Directions]({route})", unsafe_allow_html=True)
                else:
                    st.write("Location data is not available for this restaurant.")
if __name__ == '__main__':
    render_home_page()
