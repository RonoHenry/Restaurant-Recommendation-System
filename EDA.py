# Import Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from IPython.display import display
import json
from shapely.geometry import Point
import folium
from folium.plugins import MarkerCluster




class ComprehensiveEDA:
    """
    Base class for Exploratory Data Analysis with detailed visualizations.
    """
    def __init__(self, df):
        self.df = df




class UnivariateAnalysis:
    """
    Perform univariate analysis to understand individual features.
    This class is derived from a base class ComprehensiveEDA, which should handle basic DataFrame management.
    """
    def __init__(self, df):
        self.df = df

    def plot_histogram(self, column, title, bins=30, color='skyblue'):
        """
        Plots a histogram of a specified column using the provided title and color settings.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(self.df[column].dropna(), bins=bins, color=color)
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def plot_bar_chart(self, column, title, color='coral', limit=None):
        """
        Plots a bar chart of the value counts for a specified column.
        """
        plt.figure(figsize=(10, 6))
        value_counts = self.df[column].value_counts()
        if limit:
            value_counts = value_counts.head(limit)
        value_counts.plot(kind='bar', color=color)
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    def plot_boxplot(self, column, title):
        """
        Plots a boxplot for the specified column using the given title.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.df[column])
        plt.title(title)
        plt.grid(True)
        plt.show()

    def plot_pie_chart(self, column, title):
        """
        Plots a pie chart of the value counts for a specified column.
        """
        plt.figure(figsize=(8, 8))
        self.df[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
        plt.title(title)
        plt.ylabel('')  # Hide the y-axis label as it is not needed for pie charts
        plt.show()


class BivariateAnalysis(ComprehensiveEDA):
    """
    Perform bivariate analysis to understand relationships between two features.
    """
    def plot_scatter(self, x, y, title):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.df[x], y=self.df[y])
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(True)
        plt.show()

    def plot_boxplot(self, x, y, title):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.df[x], y=self.df[y])
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()





class MultivariateAnalysis(ComprehensiveEDA):
    """
    Perform multivariate analysis to understand complex relationships between multiple features.
    This class inherits from ComprehensiveEDA.
    """

    def plot_top_rated_restaurants(self):
        """
        Plots the top-rated restaurant in each state.
        """
        top_rated = self.df.groupby('state').apply(lambda x: x.nlargest(1, 'stars'))
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_rated, x='state', y='stars', hue='name')
        plt.title('Top Rated Restaurants per State')
        plt.xlabel('State')
        plt.ylabel('Stars')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    def plot_restaurant_locations_folium(self):
        """
        Displays the locations of restaurants on a map of the USA using Folium.
        """


        # Initialize a folium map centered at the mean latitude and longitude
        map_center = [self.df['latitude'].mean(), self.df['longitude'].mean()]
        m = folium.Map(location=map_center, zoom_start=4)

        # Add a marker cluster
        marker_cluster = MarkerCluster().add_to(m)

        # Add markers to the map
        for idx, row in self.df.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"{row['name']} ({row['stars']} stars)"
            ).add_to(marker_cluster)

        # Display the map
        return m




class UserEDA(ComprehensiveEDA):
    """
    Perform Exploratory Data Analysis specifically for the users.csv dataset.
    This class inherits from a general EDA class and focuses on user-related data.
    """


    def plot_rating_distribution(self):
        """
        Plot the distribution of star ratings.
        """
        plt.figure(figsize=(10, 6))
        self.df['stars'].value_counts().sort_index().plot(kind='bar')
        plt.title('Distribution of Ratings')
        plt.xlabel('Star Ratings')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def plot_reviews_over_time(self):
        """
        Plot the number of reviews over time, terminating the data at the end of December of the last year available in the dataset.
        """
        # Ensure 'date' is datetime
        self.df['date'] = pd.to_datetime(self.df['date'])

          # Define the cutoff date as the end of December 2021
        cutoff_date = pd.Timestamp('2021-12-31')

         # Filter the data to include only dates up to the end of December 2021
        filtered_df = self.df[self.df['date'] <= cutoff_date]

        # Resample to monthly and count
        monthly_reviews = filtered_df.resample('M', on='date').size()

        plt.figure(figsize=(12, 6))
        monthly_reviews.plot()
        plt.title('Review Counts Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Reviews')
        plt.grid(True)
        plt.show()

    def plot_wordcloud(self):
        """
        Generate a word cloud from the 'text' column in the dataset.
        """
        from wordcloud import WordCloud
        text = ' '.join(self.df['text'].dropna().tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Word Cloud of Review Texts')
        plt.axis('off')
        plt.show()
