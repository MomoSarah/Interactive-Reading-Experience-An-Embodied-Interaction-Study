#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().system('pip install wordcloud')

# Set the style of seaborn
sns.set(style="whitegrid")


# In[2]:


# Define file paths
pre_experiment_path = 'Reading Habit.xlsx'
post_experiment_path = 'Responses.xlsx'

# Load the data into DataFrames
pre_df = pd.read_excel(pre_experiment_path)
post_df = pd.read_excel(post_experiment_path)

# Display the first few rows of each DataFrame
print("Pre-Experiment Data:")
print(pre_df.head())

print("\nPost-Experiment Data:")
print(post_df.head())


# In[6]:


# Merge the two datasets
df_merged = pd.concat([pre_df, post_df], axis=1)

# Insert a new ID column
df_merged.insert(0, 'ID', range(1, len(df_merged) + 1))

# Display the merged dataframe
print(df_merged.head())


# In[7]:


# Display basic information
print("Pre-Experiment Data Info:")
print(df_merged.info())



# In[ ]:


# Display statistical summary
print("Pre-Experiment Data Description:")
print(df_merged.describe(include='all'))



# In[9]:


# Check for missing values
print("Pre-Experiment Data Missing Values:")
print(df_merged.isnull().sum())



# In[13]:


# Check cleaned column names
print("Cleaned Post-Experiment Data Columns:")
print(df_merged.columns)


# # Pre Experiment Questions EDA

# In[20]:


# Convert 'What is your age?' to categorical
df_merged['What is your age?'] = df_merged['What is your age?'].astype('category')

# Convert 'How often do you read books?' to categorical
df_merged['How often do you read books?'] = df_merged['How often do you read books?'].astype('category')
# Convert 'Do you listen to music while reading?' to categorical
df_merged['What type of books do you usually read? '] = df_merged['What type of books do you usually read? '].astype('category')

# Convert 'On average, how much time do you spend reading each session?' to categorical
df_merged['On average, how much time do you spend reading each session?'] = df_merged['On average, how much time do you spend reading each session?'].astype('category')

# Convert 'When do you prefer to read?' to categorical
df_merged['When do you prefer to read?'] = df_merged['When do you prefer to read?'].astype('category')

# Convert 'Do you prefer physical books or e-books?' to categorical
df_merged['Do you prefer physical books or e-books?'] = df_merged['Do you prefer physical books or e-books?'].astype('category')

# Convert 'Do you listen to music while reading?' to categorical
df_merged['Do you listen to music while reading?'] = df_merged['Do you listen to music while reading?'].astype('category')




# In[4]:


# Set the aesthetic style of the plots
sns.set_style('whitegrid')
# Convert infinite values to NaN
df_merged.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

# Plot distribution of age
plt.figure(figsize=(12, 6))
sns.countplot(data=df_merged, x='What is your age?')
plt.title('Age Distribution of Participants')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Plot gender distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=df_merged, x='Gender')
plt.title('Gender Distribution of Participants')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Plot distribution of book reading frequency
plt.figure(figsize=(12, 6))
sns.countplot(data=df_merged, x='How often do you read books?')
plt.title('Frequency of Book Reading')
plt.xlabel('Frequency')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Plot types of books read
plt.figure(figsize=(12, 6))
sns.countplot(data=df_merged, x='What type of books do you usually read? ')
plt.title('Types of Books Read')
plt.xlabel('Book Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Plot average reading time
plt.figure(figsize=(12, 6))
sns.countplot(data=df_merged, x='On average, how much time do you spend reading each session?')
plt.title('Average Reading Time Per Session')
plt.xlabel('Reading Time')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Plot preferred reading time
plt.figure(figsize=(12, 6))
sns.countplot(data=df_merged, x='When do you prefer to read?')
plt.title('Preferred Reading Time')
plt.xlabel('Preferred Reading Time')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Plot preference for physical books vs e-books
plt.figure(figsize=(12, 6))
sns.countplot(data=df_merged, x='Do you prefer physical books or e-books?')
plt.title('Preference for Physical Books vs E-books')
plt.xlabel('Book Preference')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Plot preference for listening to music while reading
plt.figure(figsize=(12, 6))
sns.countplot(data=df_merged, x='Do you listen to music while reading?')
plt.title('Listening to Music While Reading')
plt.xlabel('Music Preference')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[22]:


# Summary of categorical columns
for column in df_merged.select_dtypes(include=['category']).columns:
    print(f"\nSummary of {column}:")
    print(pre_df[column].value_counts())


# # Post Experiment Question EDA

# In[5]:


# Summary statistics for numerical columns (ratings)
ratings_columns = [
    'On a scale of 1 to 5, how enjoyable did you find the interactive storytelling experience?\n',
    'On a scale of 1 to 5, how engaging did you find the combination of music and lighting changes with the passages?',
    'On a scale of 1 to 5, how well did the music enhance your reading experience for the sad passages?',
    'On a scale of 1 to 5, how well did the music enhance your reading experience for the not sad passages?\n',
    'On a scale of 1 to 5, how well did the lighting enhance your reading experience for the sad passages?',
    'On a scale of 1 to 5, how well did the lighting enhance your reading experience for the not sad passages?',
    'On a scale of 1 to 5, how smooth were the transitions between sad and not sad passages?',
    'On a scale of 1 to 5, how immersive was the interactive storytelling experience?\n',
    'On a scale of 1 to 5, how does this interactive storytelling experience compare to your regular reading experience?'
]

# Calculate summary statistics (mean, std, min, max)
ratings_summary = df_merged[ratings_columns].describe()
print(ratings_summary)


# In[23]:


# Summary statistics for rating columns
print(post_df[rating_columns].describe())


# In[25]:


# Group data by age, gender, and reading habits to see trends
grouped_summary = df_merged.groupby(['What is your age?', 'Gender', 'How often do you read books?'])[ratings_columns].mean()
print(grouped_summary)


# In[ ]:


# Calculate correlations between ratings
ratings_correlation = df_merged[ratings_columns].corr()
print(ratings_correlation)



# In[71]:


plt.figure(figsize=(12, 6))
sns.countplot(data=post_df, x='Would you be interested in using interactive storytelling with music and lighting changes in the future?')
plt.title('Interest in Future Use of Interactive Storytelling')
plt.xlabel('Interest Level')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[30]:


# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=post_df, 
                x='On a scale of 1 to 5, how enjoyable did you find the interactive storytelling experience?',
                y='On a scale of 1 to 5, how well did the music enhance your reading experience for the sad passages?',
                alpha=0.7)

plt.title('Comparison of Enjoyment Ratings with Music Impact on Sad Passages')
plt.xlabel('Enjoyment Rating')
plt.ylabel('Music Impact on Sad Passages')
plt.show()


# In[31]:


# Calculate correlation
correlation = post_df[rating_columns].corr().iloc[0, 1]
print(f'Correlation between enjoyment rating and music impact on sad passages: {correlation:.2f}')



# In[32]:


# Convert the relevant columns to numeric
for column in rating_columns:
    post_df[column] = pd.to_numeric(post_df[column], errors='coerce')

# Calculate the correlation matrix
correlation_matrix = post_df[rating_columns].corr()


# In[4]:


# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix of Post-Experiment Ratings')
plt.show()


# In[29]:


# Analyze the interest in future use
future_interest = df_merged['Would you be interested in using interactive storytelling with music and lighting changes in the future?\n'].value_counts(normalize=True)
print(future_interest)


# In[30]:


# Analyze the text feedback for additional comments or suggestions
text_feedback = df_merged['Please share any additional comments or suggestions about the interactive storytelling experience'].dropna()
print(text_feedback)


# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns

# Example: Visualizing average ratings for enjoyment vs engagement
plt.figure(figsize=(10,6))
sns.scatterplot(x='On a scale of 1 to 5, how enjoyable did you find the interactive storytelling experience?\n',
                y='On a scale of 1 to 5, how engaging did you find the combination of music and lighting changes with the passages?',
                data=df_merged)
plt.title('Enjoyment vs Engagement of Interactive Storytelling')
plt.show()


# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns

# Select the columns for ratings
ratings_columns = [
    'On a scale of 1 to 5, how enjoyable did you find the interactive storytelling experience?\n',
    'On a scale of 1 to 5, how engaging did you find the combination of music and lighting changes with the passages?',
    'On a scale of 1 to 5, how immersive was the interactive storytelling experience?\n'
]

# Rename columns for easier labeling in the plot
df_merged.rename(columns={
    'On a scale of 1 to 5, how enjoyable did you find the interactive storytelling experience?\n': 'Enjoyment',
    'On a scale of 1 to 5, how engaging did you find the combination of music and lighting changes with the passages?': 'Engagement',
    'On a scale of 1 to 5, how immersive was the interactive storytelling experience?\n': 'Immersion'
}, inplace=True)

# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_merged[['Enjoyment', 'Engagement', 'Immersion']])
plt.title('Distribution of Ratings for Enjoyment, Engagement, and Immersion')
plt.ylabel('Rating (1-5)')
plt.xlabel('Experience Aspect')
plt.show()


# In[33]:


# Calculate correlation between ratings columns
ratings_columns = ['Enjoyment', 'Engagement', 'Immersion']

# Create correlation heatmap
plt.figure(figsize=(8, 6))
corr_matrix = df_merged[ratings_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation between Experience Ratings')
plt.show()


# In[34]:


# Count the responses for future interest
future_interest = df_merged['Would you be interested in using interactive storytelling with music and lighting changes in the future?\n'].value_counts()

# Create bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=future_interest.index, y=future_interest.values)
plt.title('Interest in Future Use of Interactive Storytelling')
plt.ylabel('Number of Participants')
plt.xlabel('Interest Level')
plt.show()


# In[38]:


# Bar plot with error bars showing mean enjoyment ratings by age group
plt.figure(figsize=(10, 6))
sns.barplot(x='What is your age?', y='Enjoyment', data=df_merged, ci="sd", palette="Blues_d")
plt.title('Mean Enjoyment Ratings by Age Group')
plt.ylabel('Mean Enjoyment Rating (1-5)')
plt.xlabel('Age Group')
plt.show()


# In[44]:


print(df_merged.columns.tolist())


# In[52]:


# Rename columns explicitly if you know the exact names
df_merged.rename(columns={
    'On a scale of 1 to 5, how enjoyable did you find the interactive storytelling experience?\n': 'Enjoyment',
    'On a scale of 1 to 5, how well did the music enhance your reading experience for the sad passages?': 'Music Enhancement Sad',
    'On a scale of 1 to 5, how well did the music enhance your reading experience for the not sad passages?\n': 'Music Enhancement Not Sad'
}, inplace=True)


# In[46]:


print(df_merged.head())


# In[56]:


# Clean any extra spaces and newlines again
df_merged.columns = df_merged.columns.str.strip().str.replace('\n', '', regex=False)

# Print cleaned column names again to verify
print(df_merged.columns)


# In[57]:


# Extract relevant columns using the correct names
enjoyment = df_merged['enjoyment']
music_enhancement_sad = df_merged['music enhancement sad']
music_enhancement_not_sad = df_merged['on a scale of 1 to 5, how well did the music enhance your reading experience for the not sad passages?']

# Calculate correlation
corr_enjoyment_sad = enjoyment.corr(music_enhancement_sad)
corr_enjoyment_not_sad = enjoyment.corr(music_enhancement_not_sad)

# Print the results
print(f'Correlation between Enjoyment and Music Enhancement for Sad Passages: {corr_enjoyment_sad:.2f}')
print(f'Correlation between Enjoyment and Music Enhancement for Not Sad Passages: {corr_enjoyment_not_sad:.2f}')


# In[58]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame for correlation
correlation_data = {
    'Sad Passages': [enjoyment.corr(music_enhancement_sad)],
    'Not Sad Passages': [enjoyment.corr(music_enhancement_not_sad)]
}

df_corr = pd.DataFrame(correlation_data, index=['Enjoyment'])

# Plot a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation of Enjoyment with Music Enhancement')
plt.show()


# In[59]:


# Correlation values
corr_values = [corr_enjoyment_sad, corr_enjoyment_not_sad]
labels = ['Sad Passages', 'Not Sad Passages']

# Plot a bar chart
plt.figure(figsize=(8, 5))
plt.bar(labels, corr_values, color=['blue', 'orange'])
plt.ylim(-1, 1)  # Correlation ranges from -1 to 1
plt.ylabel('Correlation Coefficient')
plt.title('Correlation of Enjoyment with Music Enhancement')
plt.show()


# In[60]:


engagement = df_merged['engagement']
immersion = df_merged['immersion']
corr_engagement_immersion = engagement.corr(immersion)
print(f'Correlation between Engagement and Immersion: {corr_engagement_immersion:.2f}')


# In[62]:


df_merged.groupby('what is your age?')['enjoyment'].mean().plot(kind='bar')
plt.title('Average Enjoyment Across Age Groups')
plt.show()


# In[63]:


df_merged['enjoyment'].hist()
plt.title('Distribution of Enjoyment Ratings')
plt.show()


# In[64]:


df_merged.groupby('do you prefer physical books or e-books?')['enjoyment'].mean().plot(kind='bar')
plt.title('Enjoyment Levels by Book Preference')
plt.show()


# In[68]:


from wordcloud import WordCloud
feedback_text = ' '.join(df_merged['please share any additional comments or suggestions about the interactive storytelling experience'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(feedback_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[73]:


# Rename columns for simplicity
post_df.rename(columns={
    'On a scale of 1 to 5, how enjoyable did you find the interactive storytelling experience?\n': 'Enjoyment',
    'On a scale of 1 to 5, how engaging did you find the combination of music and lighting changes with the passages?': 'Engagement',
    'On a scale of 1 to 5, how well did the music enhance your reading experience for the sad passages?': 'Music Enhancement Sad',
    'On a scale of 1 to 5, how well did the music enhance your reading experience for the not sad passages?\n': 'Music Enhancement Not Sad',
    'On a scale of 1 to 5, how well did the lighting enhance your reading experience for the sad passages?': 'Lighting Enhancement Sad',
    'On a scale of 1 to 5, how well did the lighting enhance your reading experience for the not sad passages?': 'Lighting Enhancement Not Sad',
    'On a scale of 1 to 5, how smooth were the transitions between sad and not sad passages?': 'Transitions Smoothness',
    'On a scale of 1 to 5, how immersive was the interactive storytelling experience?\n': 'Immersion',
    'On a scale of 1 to 5, how does this interactive storytelling experience compare to your regular reading experience?': 'Experience Comparison'
}, inplace=True)


# In[74]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Select columns of interest for correlation analysis
columns_of_interest = [
    'Enjoyment',
    'Engagement',
    'Music Enhancement Sad',
    'Music Enhancement Not Sad',
    'Lighting Enhancement Sad',
    'Lighting Enhancement Not Sad',
    'Transitions Smoothness',
    'Immersion',
    'Experience Comparison'
]

# Calculate the correlation matrix
correlation_matrix = post_df[columns_of_interest].corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Ratings and Enhancements')
plt.show()


# In[11]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Join the comments from the DataFrame with additional comments
additional_comments = (
    "For sad passages, the combination of music and lighting was distracting. "
    "Music was distracting sometimes. "
    "Lighting could be much better."
)
feedback_text = ' '.join(df_merged['Please share any additional comments or suggestions about the interactive storytelling experience
s'].dropna())
feedback_text += ' ' + additional_comments  # Add the additional comments

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(feedback_text)

# Display the word cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have already loaded your data into a dataframe 'df'
# Rename columns for convenience
df = df_merged.rename(columns={
    'On a scale of 1 to 5, how well did the music enhance your reading experience for the sad passages?': 'Music_Sad',
    'On a scale of 1 to 5, how well did the music enhance your reading experience for the not sad passages?\n': 'Music_Not_Sad',
    'On a scale of 1 to 5, how well did the lighting enhance your reading experience for the sad passages?': 'Light_Sad',
    'On a scale of 1 to 5, how well did the lighting enhance your reading experience for the not sad passages?': 'Light_Not_Sad'
})

# Calculate average enjoyment levels for music and lighting (sad and not sad passages)
avg_music_sad = df['Music_Sad'].mean()
avg_music_not_sad = df['Music_Not_Sad'].mean()
avg_light_sad = df['Light_Sad'].mean()
avg_light_not_sad = df['Light_Not_Sad'].mean()

# Create a DataFrame to plot the results
enjoyment_data = pd.DataFrame({
    'Sensory Cue': ['Music - Sad', 'Music - Not Sad', 'Light - Sad', 'Light - Not Sad'],
    'Average Enjoyment': [avg_music_sad, avg_music_not_sad, avg_light_sad, avg_light_not_sad]
})

# Plot the comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Sensory Cue', y='Average Enjoyment', data=enjoyment_data, palette='Blues_d')

# Adding labels and title
plt.title('Average Enjoyment Level for Music vs Light (Sad and Not Sad Passages)')
plt.xlabel('Sensory Cue')
plt.ylabel('Average Enjoyment Level (1 to 5)')

# Show the plot
plt.show()


# In[9]:


import matplotlib.pyplot as plt

# Data for pie charts
labels = ['Enjoyment', 'Remaining']

# Calculate remaining percentages (out of 5)
music_sad_remaining = 5 - avg_music_sad
music_not_sad_remaining = 5 - avg_music_not_sad
light_sad_remaining = 5 - avg_light_sad
light_not_sad_remaining = 5 - avg_light_not_sad

# Plotting the pie charts
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Music - Sad
axs[0, 0].pie([avg_music_sad, music_sad_remaining], labels=labels, autopct='%1.1f%%', colors=['#1f77b4', '#d3d3d3'])
axs[0, 0].set_title('Music - Sad Passages')

# Music - Not Sad
axs[0, 1].pie([avg_music_not_sad, music_not_sad_remaining], labels=labels, autopct='%1.1f%%', colors=['#ff7f0e', '#d3d3d3'])
axs[0, 1].set_title('Music - Not Sad Passages')

# Lighting - Sad
axs[1, 0].pie([avg_light_sad, light_sad_remaining], labels=labels, autopct='%1.1f%%', colors=['#2ca02c', '#d3d3d3'])
axs[1, 0].set_title('Lighting - Sad Passages')

# Lighting - Not Sad
axs[1, 1].pie([avg_light_not_sad, light_not_sad_remaining], labels=labels, autopct='%1.1f%%', colors=['#9467bd', '#d3d3d3'])
axs[1, 1].set_title('Lighting - Not Sad Passages')

# Adjust layout
plt.tight_layout()
plt.show()


# In[ ]:




