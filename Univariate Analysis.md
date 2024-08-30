Univariate analysis focuses on examining a single variable at a time. Several common tasks and techniques involved in univariate analysis on a dataset:

1.Descriptive Statistics:
    Measures of Central Tendency: Mean, median, mode.
    Measures of Dispersion: Range, variance, standard deviation.
    Shape of Distribution: Skewness and kurtosis.

2.Visualization:
    Numerical Data
        Histograms: Displaying the distribution of numerical data. --> Matplotlib (plt.hist()), Seaborn (sns.histplot())
        Box Plots: Illustrating the range, median, and outliers of numerical data. --> Matplotlib (plt.boxplot()), Seaborn (sns.boxplot())
        Density Plot: A density plot (or kernel density plot) visualizes the distribution of numerical data as a smoothed curve. --> Matplotlib (plt.plot() with kind='density'), Seaborn (sns.kdeplot()).
        Violin Plot: A violin plot combines aspects of a box plot and a density plot.It displays the distribution of numerical data across different levels of a categorical variable. --> Seaborn (sns.violinplot())
        Rug Plot: Adds a small vertical tick at each data point along the x-axis. Often used in combination with other plots (like histograms or density plots) to show the distribution of data points. --> Matplotlib (plt.plot() with kind='rug'), Seaborn (sns.rugplot())
        Strip Plot and Swarm Plot: To visualize the distribution of numerical data points along a categorical axis. --> Seaborn (sns.stripplot(), sns.swarmplot())
    Categorical Data
        Bar Charts: Showing the frequency distribution of categorical data. --> Matplotlib (plt.bar()), Seaborn (sns.countplot())
        Pie Charts: Representing proportions of categorical variables. --> Matplotlib (plt.pie())
        Donut Chart: Similar to pie charts but with a hole in the center. --> Matplotlib (plt.pie() with wedgeprops={'width': 0.3})
        Stacked Bar Plot: Shows the composition of each category by stacking bars for different sub-categories. --> Matplotlib (plt.bar() with bottom parameter), Pandas (df.plot(kind='bar', stacked=True))
        Grouped Bar Plot: Displays multiple bars side by side, each representing a different category. --> Matplotlib (plt.bar() with adjustments for positioning), Pandas (df.plot(kind='bar') with adjustments)
        Pie-of-Pie Chart: Highlights specific categories or sub-categories within a larger dataset. --> Matplotlib (plt.pie() with autopct='%1.1f%%' and explode parameter)
        Treemap: Represents hierarchical data as nested rectangles, where each rectangle's size is proportional to data values.
        Word Cloud: Visualizes text data by representing word frequencies using different font sizes or colors

3.Frequency Analysis:
    Frequency Distributions: Displaying how often each value of a numerical variable occurs. 
                             Understanding the distribution and concentration of values within a numerical variable. --> value_counts(), groupby()
                             Histograms, Bar Plots, KDE Plot, Density Plot 
    Frequency Tables: Summarizing the count or percentage of each category for categorical variables. --> Bar Plot, Pie Chart, countplot()
    
4.Probability Distributions:
    Identifying and fitting theoretical distributions (e.g., normal, Poisson) to numerical data to understand underlying patterns.

5.Measuring Skewness and Kurtosis:
    Skewness: Whether the data distribution is symmetric or skewed.
    Kurtosis: How much data is in the tails of the distribution.

6.Outlier Detection:
    Identifying extreme values that deviate significantly from the rest of the data. --> Box Plots, Histograms --> Z-score, IQR

7.Data Transformation:
    Applying transformations (e.g., logarithmic, square root) to normalize the distribution or make patterns more visible.

8.Testing Assumptions:
    Checking assumptions such as normality or homogeneity of variance for further analyses.

9.Segmentation:
    Grouping data into segments based on values of the variable for comparative analysis. -->Define segments based on specific values or ranges of the variable. -->pd.qcut(),pd.cut()
                                                 For example, dividing customers into age groups (e.g., 18-25, 26-35, etc.) or income brackets (e.g., low, medium, high).

10.Missing Value Analysis:
    Checking for missing data and determining its impact on the analysis.

The goal of univariate analysis is to describe the data and understand its characteristics before moving on to more complex analyses 
like bivariate or multivariate analysis. Each technique provides insights into the dataset's structure, patterns, and potential outliers or errors.