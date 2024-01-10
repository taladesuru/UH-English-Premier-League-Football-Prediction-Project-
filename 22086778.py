# -*- coding: utf-8 -*-

"""
Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
"""

import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns

# Read Dataset
df = pd.read_csv('data.csv')

# Data Cleaning and Exploration
# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Explore unique values in categorical columns
for column in df.select_dtypes(include='object').columns:
    print(f"Unique values in {column}:", df[column].unique())

# Summary Statistics of Dataset
print(df.describe())

# Checking the Rows and Columns
print("Number of rows and columns:", df.shape)

# Select the Columns Needed
df1 = df[['Age', 'Gender', 'City', 'FavoriteTourismDest_india', 'FavoriteTourismDest_abroad']]

# Defining the DataFrame for Gender Count
gender_counts = df1['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']

# Setting the Plot
plt.figure(figsize=(8, 5))
sns.barplot(x='Gender', y='Count', data=gender_counts, palette='viridis')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')

# Labelling on top of the bars
for index, value in enumerate(gender_counts['Count']):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=12)

# Include summary message
summary_text = ( 
    "SUMMARY INFORMATION\n\n Higher % of male gender travel within India and Abroad compared to females.\n"
)
fig = plt.gcf()
textbox = plt.text(0.1, -0.1, summary_text, transform=fig.transFigure,
                   fontsize=10, fontweight='bold', horizontalalignment='left')



# Get the Travel Count from Mumbai to Zurich
count_mumbai_to_zurich = df1[(df1['City'] == 'Mumbai') & (df1['FavoriteTourismDest_abroad'] == 'Zurich')].shape[0]

print("Total count from Mumbai to Zurich:", count_mumbai_to_zurich)

# Create a Pie chart for Travels from Mumbai to Zurich
labels = ['Mumbai to Zurich', 'Other Destinations']
sizes = [count_mumbai_to_zurich, len(df1) - count_mumbai_to_zurich]
colors = ['red', 'green']

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)

# Create a Doughnut from a Pie Chart
centre_circle = plt.Circle((0, 0), 0.50, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')

plt.title('Distribution of Travel from Mumbai to Zurich')
# Include summary message
summary_text = ( 
    "SUMMARY INFORMATION\n\n Trend of travel to Zurich is 11.3% from Mumbai compared to other destination.\n"
)
fig = plt.gcf()
textbox = plt.text(0.1, -0.1, summary_text, transform=fig.transFigure,
                   fontsize=10, fontweight='bold', horizontalalignment='left')


# Getting the Age Range Distribution of Participants
plt.figure(figsize=(12, 6))
sns.histplot(data=df1, x='Age', bins=20, kde=True, color='skyblue')
plt.title('Age Distribution of Participants')
plt.xlabel('Age')
plt.ylabel('Count')

# Include summary message
summary_text = ( 
    "SUMMARY INFORMATION\n\n Age range 23 - 32 provided the highest number of the travel particpants with Age 27 being at the peak.\n"
)
fig = plt.gcf()
textbox = plt.text(0.1, -0.1, summary_text, transform=fig.transFigure,
                   fontsize=10, fontweight='bold', horizontalalignment='left')



# Setting the figure size
plt.figure(figsize=(14, 8))

# Get unique colors for male and female
palette_colors = {'Male': '#1f77b4', 'Female': 'red'}

# Create separate histograms for each gender
for gender, color in palette_colors.items():
    sns.histplot(df1[df1['Gender'] == gender]['Age'], bins=20, kde=True, color=color, label=gender, alpha=0.7)

plt.title('Age Distribution by Gender')
plt.xlabel('Age')
plt.ylabel('Count')

# Adding legend
plt.legend(title='Gender')

# Include summary message
summary_text = ( 
    "SUMMARY INFORMATION\n\n Male Age range 26 - 27 provided the most frequent travel count of 50\n\n Female Age range 27 also provided the highest travel count of 34"
)
fig = plt.gcf()
textbox = plt.text(0.1, -0.1, summary_text, transform=fig.transFigure,
                   fontsize=10, fontweight='bold', horizontalalignment='left')



# Get the Count for Favourite Tourism Destination Abroad
destinations_count = df1['FavoriteTourismDest_abroad'].value_counts()

print("Total count for FavoriteTourismDest_abroad:")
print(destinations_count)

# Calculate the percentages for each count
total_responses = len(df1)
percentages = destinations_count / total_responses * 100

# Bar plot with rotated x-axis labels and color background
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=destinations_count.index, y=percentages, palette='viridis')
ax.set_facecolor('#f5f5f5')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

plt.title('Distribution of Favorite Tourism Destinations Abroad')
plt.xlabel('Favorite Tourism Destinations')
plt.ylabel('Percentage')

# Set y-axis limit proper display of percentage
plt.ylim(0, 100)

# Adding the percentage labels at the top of the bars
for index, value in enumerate(percentages):
    plt.text(index, value + 1, f'{value:.1f}%', ha='center', va='bottom', fontsize=10)

# Include summary message
summary_text = ( 
    "SUMMARY INFORMATION\n\n The most visited location from India is Zurich with 33.5% of their travel destination, followed by Kyiv with 18.7%, \n\n and then Paris with 12.9% respectively. While Jerusalem with 0.7 had the lowest destination point abroad.\n"
)
fig = plt.gcf()
textbox = plt.text(0.1, -0.1, summary_text, transform=fig.transFigure,
                   fontsize=10, fontweight='bold', horizontalalignment='left')



# Get the Count for Favourite Tourism Destination India
destinations_india_count = df1['FavoriteTourismDest_india'].value_counts()

print("Total count for FavoriteTourismDest_india:")
print(destinations_india_count)

# Getting the Count for Tourism Destination India
destinations_india_count = df1['FavoriteTourismDest_india'].value_counts()

# Create random colors for each wedge
labels = destinations_india_count.index
sizes = destinations_india_count.values
random_colors = np.random.rand(len(labels), 3)

fig, ax = plt.subplots(figsize=(10, 10))
wedgeprops = dict(width=0.4, edgecolor='w')  # Adjust the width of the wedges
patches, texts, autotexts = ax.pie(sizes, labels=labels, autopct=lambda p: '{:.1f}%'.format(p) if p > 1 else '',
                                   startangle=90, colors=random_colors, wedgeprops=wedgeprops)

# Draw a white circle at the center to create a doughnut chart
centre_circle = plt.Circle((0, 0), 0.65, fc='white')
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')

plt.title('Distribution of Favorite Tourism Destinations in India', pad=20)

# Adjust the position of the subplot
box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

# Create legend with labels and corresponding colors
legend_labels = [f'{label} ({size})' for label, size in zip(labels, sizes)]
plt.legend(patches, legend_labels, loc='upper right', bbox_to_anchor=(1.3, 1))

# Include summary message
summary_text = ( 
    "SUMMARY INFORMATION\n\n Agra (Home of the Taj Mahal) is the most famous tourist destination in India with 23.8% of tourist visiting it.\n"
)
fig = plt.gcf()
textbox = plt.text(0.1, -0.1, summary_text, transform=fig.transFigure,
                   fontsize=10, fontweight='bold', horizontalalignment='left')


# Assuming df1 is your DataFrame
plt.figure(figsize=(12, 8))

# Normalize the counts to percentages
city_counts = df1['City'].value_counts(normalize=True)

# Generate random colors
random_colors = np.random.rand(len(city_counts), 3)

# Create a Pie Chart with percentages outside and random colors
wedges, texts, autotexts = plt.pie(city_counts, labels=None, autopct='', pctdistance=0.75, colors=random_colors,
                                   startangle=90, textprops=dict(color="w"))

# Add percentages outside the wedges
for autotext, percent in zip(autotexts, city_counts * 100):  # Multiply by 100 to convert to percentage
    autotext.set_text('{:.1f}%'.format(percent))

# Add legend
plt.legend(wedges, city_counts.index, title='Cities', bbox_to_anchor=(1, 0.8), loc="center left", fontsize=10)

plt.title('Distribution of Participants Across Cities')

# Include summary message
summary_text = ( 
    "SUMMARY INFORMATION\n\n Disparities in population of Indian cities is reflected with the lowest travel participants \n\ncoming from Bhopal with 0.7% travel participants compared to New Delhi with 34.9% (highest) participant\n"
)
fig = plt.gcf()
textbox = plt.text(0.1, -0.1, summary_text, transform=fig.transFigure,
                   fontsize=10, fontweight='bold', horizontalalignment='left')



# Setting the figure size
plt.figure(figsize=(10, 8))

# Get unique cities and their counts
city_counts = df1['City'].value_counts()

# Generate random colors
num_cities = len(city_counts)
random_colors = np.random.rand(num_cities, 3)

# Bar plot with random colors
sns.barplot(x=city_counts.index, y=city_counts, palette=random_colors)

plt.title('Distribution of Participants Across Cities')
plt.xlabel('City')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

# Adding count labels on top of the bars
for index, value in enumerate(city_counts):
    plt.text(index, value + 1, str(value), ha='center', va='bottom', fontsize=10)


# Include summary message
summary_text = ( 
    "SUMMARY INFORMATION\n\n Indian developed cities (Like New Delhi)with higher population contributed more travel participants \n\ncompared to towns like Bhopal with the lowest participants count\n"
)
fig = plt.gcf()
textbox = plt.text(0.1, -0.1, summary_text, transform=fig.transFigure,
                   fontsize=10, fontweight='bold', horizontalalignment='left')


# Getting Gender Counts
gender_counts = df1['Gender'].value_counts()
destinations_count = df1['FavoriteTourismDest_abroad'].value_counts()

# Figure Size for Plot
fig = plt.figure(figsize=(15, 9), dpi=300)

# Subtitle Header to Dashboard
plt.suptitle('INFOGRAPHICS FOR TRAVEL TRENDS IN INDIA & ABROAD DASHBOARD\n\nBy THOMAS ALADESURU (Student ID: 22086778)',
             weight='bold', size=19, y=1.05)

plt.subplots_adjust()

# Create gridspec object
sub_gs = gridspec.GridSpec(4, 4, wspace=0.9, hspace=1.4)

# Plotting 1
ax1 = plt.subplot(sub_gs[2:4, 0:2])
# Setting the Plot
sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='viridis', ax=ax1)
ax1.set_title('Gender Distribution')
ax1.set_xlabel('Gender')
ax1.set_ylabel('Count')

# Labelling on top of the bars
for index, value in enumerate(gender_counts):
    ax1.text(index, value, str(value), ha='center', va='bottom', fontsize=12)

# Plotting 2
ax2 = plt.subplot(sub_gs[0:2, 2:4])
# Get unique colors for male and female
palette_colors = {'Male': '#1f77b4', 'Female': 'red'}

# Create separate histograms for each gender
for gender, color in palette_colors.items():
    sns.histplot(df1[df1['Gender'] == gender]['Age'], bins=20, kde=True, color=color, label=gender, alpha=0.7, ax=ax2)

ax2.set_title('Age Distribution by Gender')
ax2.set_xlabel('Age')
ax2.set_ylabel('Count')

# Adding legend
ax2.legend(title='Gender')

# Plotting 3
ax3 = plt.subplot(sub_gs[2:4, 2:4])
# Bar plot with rotated x-axis labels and color background
ax3 = sns.barplot(x=destinations_count.index, y=destinations_count, palette='viridis', ax=ax3)
ax3.set_facecolor('#f5f5f5')

# Rotate x-axis
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')

ax3.set_title('Distribution of Favorite Tourism Destinations Abroad')
ax3.set_xlabel('Favorite Tourism Destinations')
ax3.set_ylabel('Count')

# Adding count labels on top of the bars
for index, value in enumerate(destinations_count):
    ax3.text(index, value, str(value), ha='center', va='bottom', fontsize=10)

# Plotting 4
ax4 = plt.subplot(sub_gs[0:2, 0:2])
# Assuming df1 is your DataFrame
city_counts = df1['City'].value_counts()

# Generate random colors
random_colors = np.random.rand(len(city_counts), 3)

# Create a Pie Chart with percentages outside and random colors
wedges, texts, autotexts = ax4.pie(city_counts, labels=None, autopct='', pctdistance=0.75, colors=random_colors,
                                   startangle=90, textprops=dict(color="w"))

# Add percentages outside the wedges
for autotext, percent in zip(autotexts, city_counts):
    autotext.set_text('{:.1f}%'.format(percent))

# Add legend
plt.legend(wedges, city_counts.index, title='Cities', bbox_to_anchor=(1, 0.6), loc="center left", fontsize=10)

ax4.set_title('Distribution of Participants Across Cities')

# Set the border around the dashboard
fig = plt.gcf()
fig.patch.set_linewidth(5)
fig.patch.set_edgecolor('black')

#Summary Report Text for all the plots
summary_text = (
    "Summary Report\n\nThe plots above show the VISUALIZATION OF TRAVEL TRENDS IN INDIA & ABROAD. It starts with Gender Distribution of both Male and Female. Male Gender has the\n"
    "highest count of 461 traveling within India and Abroad. Age Distribution by Gender shows the gender age range of Female between 27 years and\n"
    "28 years with the most frequent travel counts of 38 and 29 respectively, followed by ages 24 & 25 years and finally between 31 years and 32 years.\n"
    "Additionally, for Male Gender, it shows that between 26 years and 27 years of age had the highest with the most frequent travel count of above 50, followed by\n"
    "23 years with 49 counts, and then 25 years with 48 counts respectively. The Distribution Participants across Cities show that New Delhi has the highest\n"
    "participants with 252%, followed by Bangalore with 133%, and then Mumbai with 123% respectively. Finally, the Tourism Destinations Abroad also show the most\n"
    "visited location from India beginning with Zurich having the highest with 33.5%, followed by Kyiv with 18.7%, and then Paris with 12.9% respectively.\n\n"
    "In Summary, the analysis shows that the Male Gender has more counts for Trends in traveling both in India and Abroad."
)

textbox = plt.text(0.5, -0.16, summary_text, transform=fig.transFigure,
                   fontsize=12, fontweight='bold', horizontalalignment='center')

plt.subplots_adjust(bottom=0.3)

# Save the dashboard as a PNG file
plt.savefig('22086778.png', dpi=300)



