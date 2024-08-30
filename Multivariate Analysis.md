# Multivariate Analysis

Multivariate analysis involves examining relationships among multiple variables simultaneously to understand patterns, dependencies, and interactions within a dataset.

## 1. Descriptive Statistics

- **Correlation Matrix**: Examines pairwise relationships between numerical variables.
- **Covariance Matrix**: Measures how much pairs of variables change together.

## 2. Visualization

- **Heatmaps**: Visualize the correlation matrix, where each cell represents the correlation coefficient between two variables, highlighting the strength and direction of relationships between multiple variables.
  - *Tools*: Seaborn (`sns.heatmap()`), Matplotlib (`plt.imshow()` with color mapping).

### Numerical vs. Numerical Data

- **Scatter Plot Matrix (Pair Plot)**: Grid of scatter plots for pairwise relationships among multiple numerical variables.
  - *Tools*: Seaborn (`sns.pairplot()`), Pandas (`pd.plotting.scatter_matrix()`).

### Numerical vs. Categorical Data

- **Box Plot with Hue**: Shows the distribution of numerical data across categories of a categorical variable.
  - *Tool*: Seaborn (`sns.boxplot()` with `hue` parameter).
  
- **Violin Plot with Hue**: Displays the distribution of numerical data across categories using kernel density estimation.
  - *Tool*: Seaborn (`sns.violinplot()` with `hue` parameter).

### Categorical vs. Categorical Data

- **Clustered Bar Chart**: Compares frequency distribution of categories across multiple categorical variables.
  - *Tools*: Matplotlib (`plt.bar()`), Seaborn (`sns.countplot()`).
  
- **Stacked Bar Chart with Hue**: Shows the composition of categories in one variable across categories of another variable.
  - *Tools*: Matplotlib (`plt.bar()` with adjustments), Pandas (`df.plot(kind='bar', stacked=True)`).

## 3. Interaction Effects

- **Interaction Plots in Multivariate Space**: Visualizes how relationships among multiple variables change based on the levels of other variables.
  - *Tools*: Seaborn (`sns.interactplot()`), StatsModels (`statsmodels.graphics.interactionplot()`).

## 4. Advanced Techniques

- **Factor Analysis**: Identifies latent variables that explain correlations among observed variables.
  - *Tools*: StatsModels (`statsmodels.api.FactorAnalysis()`), FactorAnalyzer (`factor_analyzer.FactorAnalyzer()`).
  
- **Canonical Correlation Analysis (CCA)**: Analyzes relationships between sets of variables from two different datasets.
  - *Tool*: StatsModels (`statsmodels.api.CCA()`).

## Goal

The goal of multivariate analysis is to uncover complex relationships among multiple variables, identify patterns and dependencies, and provide a deeper understanding of the data’s structure and interactions.
