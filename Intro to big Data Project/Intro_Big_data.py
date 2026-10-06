import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


#================QUESTION 1 

# Load the ratings data
ratings = pd.read_csv('u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Load the movies data
movies = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None)

# Load the users data
users = pd.read_csv('u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
total_users = ratings['user_id'].nunique()
print(total_users)

# Number of movies rated by each user
user_movie_counts = ratings.groupby('user_id').size()
print('the movies seen by each user are: ', user_movie_counts)


#Plot movies rated by users
plt.figure(figsize=(10, 6))
user_movie_counts.plot(kind='hist', bins=30, edgecolor='black')
plt.title('Number of Movies Rated by Each User')
plt.xlabel('Number of Movies')
plt.ylabel('Number of Users')
plt.show()

# Frequency of each rating
rating_counts = ratings['rating'].value_counts().sort_index()

# Plot
plt.figure(figsize=(10, 6))
rating_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Frequency of Each Rating')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


#================QUESTION 1 part (b)
from scipy.stats import zscore

# Calculate Z-scores
user_z_scores = zscore(user_movie_counts)

# Identify outliers of user's ratings (Z-score > 3 or Z-score<-3)
outliers = user_movie_counts[user_z_scores > 3]

print("Outlier users based on number of ratings:")
print(outliers)

# Convert to a wide format 
wide_format = ratings.pivot(index='user_id', columns='movie_id', values='rating')

# Display the pivoted table
print(wide_format)

# Save to a new CSV file if needed
wide_format.to_csv('wide_format_ratings.csv')


#================QUESTION 2

from sklearn.model_selection import train_test_split

#split data set into 80% training set and 20% testing set 
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Get top-rated movies based on the training set
top_movies = train_data.groupby('movie_id')['rating'].mean().sort_values(ascending=False).head(5).index

top_rated_movies = (
    train_data.groupby('movie_id')['rating']
    .mean()
    .sort_values(ascending=False)
    .head(5)
    .index.tolist()
)

# Recommend top-rated movies for all users in the test set
top_rated_recommendations = {user: top_rated_movies for user in test_data['user_id'].unique()}

print("Top-rated Recommendations:")
print(top_rated_recommendations)

# Get all unique movies in the training set
unique_movies = train_data['movie_id'].unique()

# Recommend random movies for all users in the test set
random_recommendations = {
    user: np.random.choice(unique_movies, 5, replace=False).tolist() for user in test_data['user_id'].unique()
}

print("Random Recommendations:")
print(random_recommendations)









