# Univariate Analysis

Univariate analysis focuses on examining a single variable at a time. Here are several common tasks and techniques involved in univariate analysis on a dataset:

## 1. Descriptive Statistics

- **Measures of Central Tendency**: Mean, median, mode.
- **Measures of Dispersion**: Range, variance, standard deviation.
- **Shape of Distribution**: Skewness and kurtosis.

## 2. Visualization

### Numerical Data

- **Histograms**: Display the distribution of numerical data.
  - *Tools*: Matplotlib (`plt.hist()`), Seaborn (`sns.histplot()`).
- **Box Plots**: Illustrate the range, median, and outliers of numerical data.
  - *Tools*: Matplotlib (`plt.boxplot()`), Seaborn (`sns.boxplot()`).
- **Density Plot**: Visualizes the distribution of numerical data as a smoothed curve.
  - *Tools*: Matplotlib (`plt.plot()` with `kind='density'`), Seaborn (`sns.kdeplot()`).
- **Violin Plot**: Combines aspects of a box plot and a density plot to display the distribution of numerical data across different levels of a categorical variable.
  - *Tool*: Seaborn (`sns.violinplot()`).
- **Rug Plot**: Adds a small vertical tick at each data point along the x-axis, often used with other plots to show data distribution.
  - *Tools*: Matplotlib (`plt.plot()` with `kind='rug'`), Seaborn (`sns.rugplot()`).
- **Strip Plot and Swarm Plot**: Visualize the distribution of numerical data points along a categorical axis.
  - *Tools*: Seaborn (`sns.stripplot()`, `sns.swarmplot()`).

### Categorical Data

- **Bar Charts**: Show the frequency distribution of categorical data.
  - *Tools*: Matplotlib (`plt.bar()`), Seaborn (`sns.countplot()`).
- **Pie Charts**: Represent proportions of categorical variables.
  - *Tool*: Matplotlib (`plt.pie()`).
- **Donut Chart**: Similar to pie charts but with a hole in the center.
  - *Tool*: Matplotlib (`plt.pie()` with `wedgeprops={'width': 0.3}`).
- **Stacked Bar Plot**: Shows the composition of each category by stacking bars for different sub-categories.
  - *Tools*: Matplotlib (`plt.bar()` with `bottom` parameter), Pandas (`df.plot(kind='bar', stacked=True)`).
- **Grouped Bar Plot**: Displays multiple bars side by side, each representing a different category.
  - *Tools*: Matplotlib (`plt.bar()` with adjustments for positioning), Pandas (`df.plot(kind='bar')` with adjustments).
- **Pie-of-Pie Chart**: Highlights specific categories or sub-categories within a larger dataset.
  - *Tool*: Matplotlib (`plt.pie()` with `autopct='%1.1f%%'` and `explode` parameter).
- **Treemap**: Represents hierarchical data as nested rectangles, where each rectangle's size is proportional to data values.
- **Word Cloud**: Visualizes text data by representing word frequencies using different font sizes or colors.

## 3. Frequency Analysis

- **Frequency Distributions**: Display how often each value of a numerical variable occurs.
  - *Methods*: `value_counts()`, `groupby()`.
  - *Visualizations*: Histograms, Bar Plots, KDE Plot, Density Plot.
- **Frequency Tables**: Summarize the count or percentage of each category for categorical variables.
  - *Visualizations*: Bar Plot, Pie Chart, `countplot()`.

## 4. Probability Distributions

- **Identifying and Fitting**: Theoretical distributions (e.g., normal, Poisson) to numerical data to understand underlying patterns.

## 5. Measuring Skewness and Kurtosis

- **Skewness**: Measures whether the data distribution is symmetric or skewed.
- **Kurtosis**: Measures how much data is in the tails of the distribution.

## 6. Outlier Detection

- **Identifying Extreme Values**: Detect extreme values that deviate significantly from the rest of the data.
  - *Methods*: Box Plots, Histograms.
  - *Techniques*: Z-score, IQR.

## 7. Data Transformation

- **Applying Transformations**: (e.g., logarithmic, square root) to normalize the distribution or make patterns more visible.

## 8. Testing Assumptions

- **Checking Assumptions**: Such as normality or homogeneity of variance for further analyses.

## 9. Segmentation

- **Grouping Data**: Into segments based on values of the variable for comparative analysis.
  - *Methods*: Define segments based on specific values or ranges of the variable.
  - *Functions*: `pd.qcut()`, `pd.cut()`.
  - *Examples*: Dividing customers into age groups (e.g., 18-25, 26-35) or income brackets (e.g., low, medium, high).

## 10. Missing Value Analysis

- **Checking for Missing Data**: Determining its impact on the analysis.

## Goal

The goal of univariate analysis is to describe the data and understand its characteristics before moving on to more complex analyses like bivariate or multivariate analysis. Each technique provides insights into the dataset's structure, patterns, and potential outliers or errors.
