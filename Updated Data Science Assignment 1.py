import pandas as pd
import matplotlib.pyplot as plt

# Reading the data from the CSV file source
data = pd.read_csv(r'C:\\Users\\HP\\OneDrive\\Desktop\\Thomas Assignment\\10-1310-data-interim-equality-impact-he-funding-figure-4.csv', skiprows=2)

# Extraction of the data for plotting on the graph
deciles = data['Decile']
men_income = data['Men']
women_income = data['Women']

# Creation of a figure and plot the bar chart 
plt.figure(figsize=(10, 6))

# Plot barchart for men and women for income by decile
plt.bar(deciles, men_income, width=0.4, label='Men', align='center', alpha=0.7)
plt.bar([x + 0.4 for x in deciles], women_income, width=0.4, label='Women', align='center', alpha=0.7)

# Set the labels for the plot and title for the bar chart
plt.xlabel('Decile')
plt.ylabel('Income')
plt.title('Income by Decile for Men and Women')
plt.xticks([x + 0.2 for x in deciles], deciles)  # Adjust the x-tick positions

# Addition of a legend
plt.legend()

# Show the barchart
plt.tight_layout()
plt.show()

# Create a separate figure and plot the histogram of men's income
plt.figure(figsize=(10, 6))

# Create a histogram of men's income
plt.hist(men_income, bins=10, edgecolor='k', alpha=0.7, color='blue')

# Set labels and title for the histogram
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.title('Histogram of Men\'s Income')

# Show the histogram
plt.tight_layout()
plt.show()



df = pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\Thomas Assignment\\10-1309-data-interim-impact-assessment-he-funding-table-10.csv', skiprows=1, encoding='ISO-8859-1', dtype=str)
# Convert 'Size of firm (number of employees)' column to strings# Filter out rows with valid categories
valid_categories = ['Micro: 1-9 employees - Insourced', 'Micro: 1-9 employees - Outsourced',
                    'Small: 10-49 employees - Insourced', 'Small: 10-49 employees - Outsourced',
                    'Medium: 50-249 employees - Insourced', 'Medium: 50-249 employees - Outsourced',
                    'Large: 250+ employees - Insourced', 'Large: 250+ employees - Outsourced']

df_filtered = df[df['Size of firm (number of employees)'].isin(valid_categories)]

# Create a line plot
plt.figure(figsize=(10, 6))

# Plot a line graph
plt.plot(df_filtered['Size of firm (number of employees)'], df_filtered['No of enterprises employing graduates in 2014*'], marker='o', label='No of enterprises employing graduates in 2014')
plt.plot(df_filtered['Size of firm (number of employees)'], df_filtered['Mean hourly cost of personnel officer in 2014**'], marker='o', label='Mean hourly cost of personnel officer in 2014')
plt.plot(df_filtered['Size of firm (number of employees)'], df_filtered['Mean hourly cost of IT technician***'], marker='o', label='Mean hourly cost of IT technician***')

# Set labels and title for the Line graph
plt.xlabel('Number of Employees')
plt.ylabel('In 2014 Numbers of enterprises employing graduates ')
plt.title('Enterprises Employing Graduates in 2014 by Size of Firm')


plt.xticks(rotation=45)

# Addition of a legend
plt.legend()

# Show the Line plot
plt.tight_layout()
plt.show()






