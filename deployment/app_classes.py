
import pandas as pd
import requests
import streamlit as st
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np







# Your Yelp API key
API_KEY = st.secrets(api_key)

# Yelp Business Endpoint
YELP_BUSINESS_URL = "https://api.yelp.com/v3/businesses/"

# Headers for the API request
headers = {
    'Authorization': f'Bearer {API_KEY}',
}

def get_business_image_urls(business_id):
    response = requests.get(f'{YELP_BUSINESS_URL}{business_id}', headers=headers)
    
    if response.status_code == 200:
        business_data = response.json()
        # Extract the image URLs
        image_urls = business_data.get('photos', [])
        return image_urls
    elif response.status_code == 429:
        st.error("Rate limit exceeded. Please try again later.")
    else:
        st.error(f"Failed to retrieve data for Business ID: {business_id}, Status Code: {response.status_code}")
    
    return []


def preprocess(df, state=None):
    """
    Preprocessing the df by combining restaurant features
    """
    filtered_df = df[df["state"] == state]    
    filtered_df['combined_features'] = (
        filtered_df['state'] + " " + 
        filtered_df['categories'] + " " + 
        filtered_df['city'] + " " + 
        filtered_df['name'] + " " +  
        filtered_df['attributes_true']
    )
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df

def create_feature_vectors(df):
    """
    Performing vectorization of the preprocessed categorical features 
    and combining with the numerical features
    """
    # Vectorize the combined text features
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    # Combine the TF-IDF matrix with numerical columns
    numerical_features = df[['stars']].values
    combined_features = np.hstack((tfidf_matrix.toarray(), numerical_features))
    # combined_features=tfidf_matrix
    
    return combined_features

def recommendation(df, state, name=None, category=None):
    preprocessed = preprocess(df, state)
    
    def cuisines(cuisine=None):
        cuisine_df = preprocessed[preprocessed['categories'] == cuisine]
        cuisine_df_sorted = cuisine_df.sort_values(by=["stars", "city"], ascending=False)
        return cuisine_df_sorted[['name', 'state', 'city', 'stars', 'address', 'categories']]
    
    if name:
        if name not in preprocessed['name'].values:
            raise ValueError(f"Restaurant with name '{name}' not found in the filtered data.")
        
        idx = preprocessed[preprocessed['name'] == name].index[0]
        exclude_names = [name]
        
        combined_features = create_feature_vectors(preprocessed)
        cosine_sim = cosine_similarity(combined_features, combined_features)
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Exclude the current restaurant and any with the same name
        top_indices = [i[0] for i in sim_scores]  # Start from 1 to exclude the provided restaurant itself
        
        recommended_restaurants = preprocessed.iloc[top_indices]
        recommended_restaurants = recommended_restaurants[~recommended_restaurants['name'].isin(exclude_names)]        
        
        return recommended_restaurants[['name', 'state', 'city', 'stars', 'address','categories']].drop_duplicates(subset='name').sort_values(by=["city", "stars"], ascending=False)[:30]
    
    elif category:
        return cuisines(category)
    
def pagenation(df, filter_column):
    # Get unique values from the filter column and sort them alphabetically
    filter_values = sorted(df[filter_column].unique())

    # Let the user select a filter value from the sorted list
    selected_value = st.selectbox(f"Filter by {filter_column}", filter_values)

    # Apply the filter to the DataFrame
    filtered_df = df[df[filter_column] == selected_value]

    # Number of rows per page
    ROWS_PER_PAGE = 10

    # Calculate total number of pages
    total_pages = len(filtered_df) // ROWS_PER_PAGE + (len(filtered_df) % ROWS_PER_PAGE > 0)

    # Streamlit page selector
    page = st.number_input('Page Number:', min_value=1, max_value=total_pages, value=1)

    # Calculate start and end indices for the DataFrame slice
    start_idx = (page - 1) * ROWS_PER_PAGE
    end_idx = start_idx + ROWS_PER_PAGE

    # Display the DataFrame slice and return it
    page_df = filtered_df.iloc[start_idx:end_idx]
    st.dataframe(page_df)
    return page_df



