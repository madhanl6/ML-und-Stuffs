Bivariate analysis focuses on examining a two variable at a time. Several common tasks and techniques involved in bivariate analysis on a dataset:

1. Descriptive Statistics: df.describe()
    Correlation Coefficient: Measure the strength and direction of the linear relationship between two continuous variables. --> df['X'].corr(df['Y'])
                             Calculate Pearson's correlation coefficient, which ranges from -1 (perfect negative correlation) to +1 (perfect positive correlation), 
                             with 0 indicating no linear relationship.

    Covariance: Measure how much two variables change together. 
                Compute the covariance between two continuous variables to understand the direction of their linear relationship. --> np.cov(df['X'], df['Y'])

    Crosstabulation: Summarizes the frequency of occurrences between two categorical variables.

2.Visualization:
    Numerical vs Numerical Data
        Scatter Plot: Displays individual data points in a two-dimensional space to visualize the relationship between two numerical variables. --> Matplotlib (plt.scatter()), Seaborn (sns.scatterplot())
        Joint Plot: Combines a scatter plot with marginal distributions (histograms or density plots) of each variable. --> Seaborn (sns.jointplot())
        Pair Plot: Grid of scatter plots showing pairwise relationships in a dataset with multiple numerical variables. --> Seaborn (sns.pairplot())
    Categorical vs Numerical Data
        Box Plot: Illustrates the distribution of a numerical variable across different categories of a categorical variable. --> Matplotlib (plt.boxplot()), Seaborn (sns.boxplot())
        Violin Plot: Displays the distribution of a numerical variable across levels of a categorical variable using kernel density estimation. --> Seaborn (sns.violinplot())
    Categorical vs Categorical Data
        Stacked Bar Plot: Shows the composition of each category in one variable by stacking bars for different categories of another variable. --> Matplotlib (plt.bar() with bottom parameter), Pandas (df.plot(kind='bar', stacked=True))
        Grouped Bar Plot: Displays multiple bars side by side, each representing a category in one variable across categories of another variable. --> Matplotlib (plt.bar() with adjustments for positioning), Pandas (df.plot(kind='bar'))

3.Interaction Effects:
    Interaction Plots: Visualizes how the relationship between two variables changes based on the levels of a third variable. --> Seaborn (sns.interactplot()), StatsModels (statsmodels.graphics.interactionplot())

4. Segmentation and Stratification:
    Segmentation: Segmentation involves dividing data into segments based on categories of one variable to compare relationships with another variable.
        This helps in understanding how the relationship between two variables differs across different segments.
            Segmentation by Age Groups: Dividing customers into age groups (e.g., 18-25, 26-35, etc.) to analyze how their income levels (numerical variable) vary with age.
            Technique: Use pd.cut() or pd.qcut() in Python to create age groups and then visualize or analyze the relationship between age groups and income levels.
    Stratification: Stratification involves dividing data into homogeneous groups (strata) based on a categorical variable for more focused analysis. 
        This technique ensures that comparisons are made within groups that are similar in terms of the stratifying variable.
            Stratification by Gender: Analyzing the relationship between educational attainment (numerical variable) and income levels within different gender groups.
            Technique: Use groupby() in Python to create groups based on gender and then perform separate analyses or visualizations for each group.
   
    