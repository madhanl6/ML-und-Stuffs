Multivariate analysis involves examining relationships among multiple variables simultaneously to understand patterns, dependencies, and interactions within a dataset.

1. Descriptive Statistics:
    Correlation Matrix: Examines pairwise relationships between numerical variables.
    Covariance Matrix: Measures how much pairs of variables change together.

2. Visualization:
        Heatmaps: Visualizes the correlation matrix where each cell represents the correlation coefficient between two variables. df.corr()
        Highlights the strength and direction of relationships between multiple variables. --> Seaborn (sns.heatmap()), Matplotlib (plt.imshow() with color mapping)
    Numerical vs. Numerical Data
        Scatter Plot Matrix (Pair Plot): Grid of scatter plots for pairwise relationships among multiple numerical variables. --> Seaborn (sns.pairplot()), Pandas (pd.plotting.scatter_matrix())
    Numerical vs. Categorical Data
        Box Plot with Hue: Shows distribution of numerical data across categories of a categorical variable. --> Seaborn (sns.boxplot() with hue parameter)
        Violin Plot with Hue: Displays distribution of numerical data across categories using kernel density estimation. --> Seaborn (sns.violinplot() with hue parameter)
    Categorical vs. Categorical Data
        Clustered Bar Chart: Compares frequency distribution of categories across multiple categorical variables. --> Matplotlib (plt.bar()), Seaborn (sns.countplot())
        Stacked Bar Chart with Hue: Shows composition of categories in one variable across categories of another variable. -->Matplotlib (plt.bar() with adjustments), Pandas (df.plot(kind='bar', stacked=True))

3. Interaction Effects:
    Interaction Plots in Multivariate Space: Visualizes how relationships among multiple variables change based on the levels of other variables. --> Seaborn (sns.interactplot()), StatsModels (statsmodels.graphics.interactionplot())

5. Advanced Techniques:
    Factor Analysis: Identifies latent variables that explain correlations among observed variables. --> StatsModels (statsmodels.api.FactorAnalysis()), FactorAnalyzer (factor_analyzer.FactorAnalyzer())
    Canonical Correlation Analysis (CCA): Analyzes relationships between sets of variables from two different data sets. --> StatsModels (statsmodels.api.CCA())