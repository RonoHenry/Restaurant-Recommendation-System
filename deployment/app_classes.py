

import pickle
import folium
import requests
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

@st.cache_resource
def load_model():
    with open('pickled_files/svd.pkl', 'rb') as file:
        return pickle.load(file)

svd = load_model()

# Yelp API key
API_KEY = "QO9XAZfxn80KoHc2rPOj9iEhWK2r8EJXfLNH_Q1F2O04d3XpAvdxFiX0Bz1wKge_hR0IMLsbsn2-ObSe0uTx5EWttuS_Yy_6wYvew5D0GXBGru_BV2OkyQDUlQOyZnYx"

# Yelp Business Endpoint
YELP_BUSINESS_URL = "https://api.yelp.com/v3/businesses/"

# Headers for the API request
headers = {
    'Authorization': f'Bearer {API_KEY}',
}

def get_business_info(business_id):

    """
    Function to generate business information including, name, phone number, business operating hours and website url
    """
    response = requests.get(f'{YELP_BUSINESS_URL}{business_id}', headers=headers)
    
    if response.status_code == 200:
        business_data = response.json()
        
        # Extract basic information
        name = business_data.get('name', '')
        phone = business_data.get('display_phone', '')
        website = business_data.get('url', '')
        hours = business_data.get('hours', '')
        address = business_data.get('address', '')


        image_urls = business_data.get('photos', [])
        return {
            'name': name,
            'phone': phone,
            'website': website,
            'hours': hours,
            'image_urls': image_urls,
            'address': address
        }
    
    elif response.status_code == 429:
        print("Rate limit exceeded. Please try again later.")
    else:
        print(f"Failed to retrieve data for Business ID: {business_id}, Status Code: {response.status_code}")
    
    return {}

def get_yelp_reviews(business_id):
    """
    Function to generate restaruant reviews for each restaurant
    """
    
    url = f'https://api.yelp.com/v3/businesses/{business_id}/reviews'
    headers = {
        'Authorization': f'Bearer {API_KEY}',
    }

    response = requests.get(url, headers=headers)

    reviews_info = []
    if response.status_code == 200:
        reviews = response.json().get('reviews', [])
        for review in reviews:
            reviews_info.append({
                'user': review['user']['name'],
                'rating': review['rating'],
                'text': review['text'],
                'time_created': review['time_created']
            })
    else:
        print(f"Failed to retrieve reviews: {response.status_code}")

    return reviews_info


def preprocess(df):
    """
    Function to preprocess the data to combine the needed features into one column
    """
    filtered_df=df 
    filtered_df['combined_features'] = (
                                        
                                        filtered_df['categories'] + " " +
                                        filtered_df['attributes_true'] 
    )
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df

def create_feature_vectors(df):
    """
    Performing vectorization of the preprocessed categorical features 
    and combining with the numerical features
    """
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    
    numerical_features = df[['stars']].values
    combined_features = np.hstack((tfidf_matrix.toarray(), numerical_features))
    # combined_features=tfidf_matrix
    
    return combined_features

def recommendation(df, state, name=None, category=None):
    """
    Function to generate the recommendations based on restaurant names by using cosine similarity 
    as well as filtering based on cuisine types
    """
    preprocessed = preprocess(df)
    
    def cuisines(cuisine=None, state=state):
        preprocessed=df[df["state"]==state]
        cuisine_df = preprocessed[preprocessed['categories'] == cuisine]
        cuisine_df_sorted = cuisine_df.sort_values(by=["stars", "city"], ascending=False)
        return cuisine_df_sorted[['name', 'state', 'city', 'stars', 'address', 'categories']]
    
    if name:
        if name not in preprocessed['name'].values:
            raise ValueError(f"Restaurant with name '{name}' not found in the filtered data.")
        
        idx = preprocessed[preprocessed['name'] == name].index[0]
        exclude_names = [name]

        row_to_add = preprocessed.iloc[idx]
        row_to_add_df = pd.DataFrame([row_to_add])     
        specific_state= preprocessed[preprocessed["state"] == state]
        specific_state = pd.concat([specific_state, row_to_add_df]).reset_index(drop=True)
        idx = specific_state[specific_state['name'] == name].index[0]
    
        combined_features = create_feature_vectors(specific_state)
        cosine_sim = cosine_similarity(combined_features, combined_features)
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        top_indices = [i[0] for i in sim_scores]  
        
        recommended_restaurants = specific_state.iloc[top_indices]
        recommended_restaurants = recommended_restaurants[~recommended_restaurants['name'].isin(exclude_names)]        
        
        return recommended_restaurants[['name', 'state', 'city', 'stars', 'address','categories']].drop_duplicates(subset='name')[:20]
    
    elif category:
        return cuisines(category)


def pagenation(df, filter_column):
    """
    Function to create pagenation on recommended dataframe and filter the dataframe
    """
    
    filter_values = sorted(df[filter_column].unique())
    filter_values.insert(0, 'All')  
    selected_value = st.selectbox(f"Filter by {filter_column}", filter_values)

    if selected_value == 'All':
        filtered_df = df
    else:
        filtered_df = df[df[filter_column] == selected_value]
    
    rows_per_page = 10
    total_pages = len(filtered_df) // rows_per_page + (len(filtered_df) % rows_per_page > 0)

    
    page = st.number_input('Page Number:', min_value=1, max_value=total_pages, value=1)

    
    start_idx = (page - 1) * rows_per_page
    end_idx = start_idx + rows_per_page

    
    page_df = filtered_df.iloc[start_idx:end_idx]
    column_map = {
        'name': 'Restaurant Name',
        'state': 'State',
        'city': 'City',
        'stars': 'Ratings',
        'address': 'Address',
        'categories': 'Cuisine'
    }

    page_df = page_df.rename(columns=column_map)
    st.dataframe(page_df, hide_index= True)
    
    return page_df



def recommend_restaurants(user_id, rated_restaurants, all_restaurants_df, state=None):
        """
        Function perform collaborative filtering by taking in ratings and recommending using pickled SVD model 
        """
        # Filter by state if provided
        if state:
            all_restaurants_df = all_restaurants_df[all_restaurants_df["state"] == state]

        # Get all restaurant IDs
        all_restaurant_ids = all_restaurants_df['business_id'].unique()

        # Filter out the restaurants that the user has already rated
        unrated_restaurants = [rid for rid in all_restaurant_ids if rid not in [rid for rid, _ in rated_restaurants]]

        # Predict ratings for all unrated restaurants
        predictions = [svd.predict(user_id, rid) for rid in unrated_restaurants]

        # Create a DataFrame for the predictions
        pred_df = pd.DataFrame({
            'business_id': [pred.iid for pred in predictions],
            'predicted_rating': [pred.est for pred in predictions]
        })

        # Merge with the original restaurants DataFrame to get more information
        recommendations = pred_df.merge(all_restaurants_df, on='business_id', how='left')

        # Sort by predicted rating and get top recommendations
        recommendations = recommendations.sort_values(by='predicted_rating', ascending=False)
        
        return recommendations[['name', 'state', 'city', 'stars', 'address','categories']].drop_duplicates(subset='name')[:20].reset_index(drop=True)

def collect_ratings(df):
    """
    Function to collect 3 necessary ratings for the collaborative function to recommend 
    """
    num_ratings = 3

    # Initialize session state for user ratings and sampled restaurants
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = [5.0] * num_ratings  

    # Adjust num_ratings if there are fewer restaurants available
    if len(df) < num_ratings:
        num_ratings = len(df)
        st.session_state.sampled_restaurants = df
    else:
        st.session_state.sampled_restaurants = df.sample(num_ratings).reset_index(drop=True)

    # Create columns for rating inputs if there are any restaurants
    if num_ratings > 0:
        cols = st.columns(num_ratings)
    else:
        st.write("No restaurants available to rate.")
        return

    # Display restaurants and collect ratings
    for i in range(num_ratings):
        try:
            restaurant = st.session_state.sampled_restaurants.iloc[i]
            with cols[i]:
                st.write(f"**Restaurant:** {restaurant['name']} ({restaurant['city']}, {restaurant['state']})")
                st.write(f"**Cuisine:** Sample Cuisine")  # Adjust as needed
                st.session_state.user_ratings[i] = st.number_input(
                    f"Rate {restaurant['name']}",
                    min_value=1.0,  # Float
                    max_value=5.0,  # Float
                    value=float(st.session_state.user_ratings[i]),  # Convert to float
                    step=1.0,  # Float
                    key=f"rating_{restaurant['business_id']}"
                )
        except IndexError:
            st.write(f"Error: Attempted to access index {i}, but it is out of bounds.")
            break



