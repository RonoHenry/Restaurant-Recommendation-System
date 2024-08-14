
import re
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


# Yelp API key
API_KEY = "QO9XAZfxn80KoHc2rPOj9iEhWK2r8EJXfLNH_Q1F2O04d3XpAvdxFiX0Bz1wKge_hR0IMLsbsn2-ObSe0uTx5EWttuS_Yy_6wYvew5D0GXBGru_BV2OkyQDUlQOyZnYx"

# Yelp Business Endpoint
YELP_BUSINESS_URL = "https://api.yelp.com/v3/businesses/"

# Headers for the API request
headers = {
    'Authorization': f'Bearer {API_KEY}',
}

def get_business_info(business_id):
    response = requests.get(f'{YELP_BUSINESS_URL}{business_id}', headers=headers)
    
    if response.status_code == 200:
        business_data = response.json()
        
        # Extract basic information
        name = business_data.get('name', '')
        phone = business_data.get('display_phone', '')
        website = business_data.get('url', '')
        hours = business_data.get('hours', '')
        reviews = business_data.get('reviews', '')

        image_urls = business_data.get('photos', [])
        

        return {
            'name': name,
            'phone': phone,
            'website': website,
            'reviews': reviews,
            'hours': hours,
            'image_urls': image_urls,
        }
    
    elif response.status_code == 429:
        print("Rate limit exceeded. Please try again later.")
    else:
        print(f"Failed to retrieve data for Business ID: {business_id}, Status Code: {response.status_code}")
    
    return {}


def preprocess(df):
    """
    Function to preprocess the data to combine the needed features into one column
    """
    filtered_df=df 
    filtered_df['combined_features'] = (
                                        
                                        filtered_df['attributes'] + " " +
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

    
    filter_values = sorted(df[filter_column].unique())
    filter_values.insert(0, 'All')  

    
    selected_value = st.selectbox(f"Filter by {filter_column}", filter_values)

    
    if selected_value == 'All':
        filtered_df = df
    else:
        filtered_df = df[df[filter_column] == selected_value]

    
    ROWS_PER_PAGE = 10
    total_pages = len(filtered_df) // ROWS_PER_PAGE + (len(filtered_df) % ROWS_PER_PAGE > 0)

    
    page = st.number_input('Page Number:', min_value=1, max_value=total_pages, value=1)

    
    start_idx = (page - 1) * ROWS_PER_PAGE
    end_idx = start_idx + ROWS_PER_PAGE

    
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


# Function to format each word with spaces at capital letters
def format_word(word):
    # Add a space before each capital letter that is not at the start of the word
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', word)

# Function to format attributes
def format_attributes(attributes):
    # Split the attributes by spaces
    words = attributes.split()
    # Format each word
    formatted_words = [format_word(word) for word in words]
    # Join the formatted words into a single string with new lines for each word
    formatted_string = '\n'.join(formatted_words)
    return formatted_string


