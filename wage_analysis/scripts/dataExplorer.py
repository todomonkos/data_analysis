import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class DataExplorer:
    def __init__(self, df):
        self.df = df

    def explore_data(self):
        """
        Data exploration with detailed statistical analysis and visualizations for econometric analysis.
        """
        if self.df is None:
            print("Please load data first.")
            return

        print("\n=== DATA EXPLORATION ===")

        # Menu for exploration options
        print("\n--- Exploration Options ---")
        print("1. Basic statistics and missing values")
        print("2. Distribution analysis")
        print("3. Correlation analysis")
        print("4. Time series analysis (if applicable)")
        print("5. Outlier detection")
        print("6. Econometric tests")
        print("7. All of the above")

        option = input("\nSelect exploration option(s) (comma-separated numbers): ").strip()
        options = [int(opt) for opt in option.split(',')]

        # 1. Basic statistics and missing values
        if 1 in options or 7 in options:
            self._basic_statistics()

        # 2. Distribution analysis
        if 2 in options or 7 in options:
            self._distribution_analysis()

        # 3. Correlation analysis
        if 3 in options or 7 in options:
            self._correlation_analysis()

        # 4. Time series analysis
        if 4 in options or 7 in options:
            self._time_series_analysis()

        # 5. Outlier detection
        if 5 in options or 7 in options:
            self._outlier_detection()

        # 6. Econometric tests
        if 6 in options or 7 in options:
            self._econometric_tests()

        # Ask if user wants to save exploratory results
        save_results = input("\nWould you like to save the exploratory analysis results? (y/n): ").strip().lower()
        if save_results == 'y':
            self._save_report()

    def _basic_statistics(self):
        print("\n--- Basic Information ---")
        print(f"Data shape: {self.df.shape}")
        print("\n--- Data Types ---")
        print(self.df.dtypes)

        # Check for missing values
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print("\n--- Missing Values ---")
            print(missing_values[missing_values > 0])

            # Calculate percentage of missing values
            missing_percent = (missing_values / len(self.df)) * 100
            print("\n--- Missing Values (%) ---")
            print(missing_percent[missing_percent > 0])

            # Ask if user wants to handle missing values
            handle_missing = input("\nWould you like to handle missing values? (y/n): ").strip().lower()
            if handle_missing == 'y':
                self._handle_missing_values()
        else:
            print("\nNo missing values found.")

        # Summary statistics
        print("\n--- Summary Statistics ---")
        print(self.df.describe(include='all').transpose())

    def _distribution_analysis(self):
        print("\n--- Distribution Analysis ---")

        # Analyze numeric columns
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

        if len(numeric_cols) > 0:
            # Check if user wants to see histograms
            show_hist = input("\nShow histograms for numeric variables? (y/n): ").strip().lower()
            if show_hist == 'y':
                for col in numeric_cols[:min(5, len(numeric_cols))]:
                    plt.figure(figsize=(10, 4))

                    # Histogram with KDE
                    plt.subplot(1, 2, 1)
                    sns.histplot(self.df[col], kde=True)
                    plt.title(f'Distribution of {col}')

                    # Q-Q Plot
                    plt.subplot(1, 2, 2)
                    stats.probplot(self.df[col].dropna(), dist="norm", plot=plt)
                    plt.title(f'Q-Q Plot of {col}')

                    plt.tight_layout()
                    plt.show()

                # Display skewness and kurtosis
                print("\n--- Skewness and Kurtosis ---")
                skew_kurt = pd.DataFrame({
                    'Skewness': self.df[numeric_cols].skew(),
                    'Kurtosis': self.df[numeric_cols].kurtosis()
                })
                print(skew_kurt)

                # Recommend transformations for skewed variables
                high_skew = skew_kurt[abs(skew_kurt['Skewness']) > 1].index.tolist()
                if high_skew:
                    print("\nVariables with high skewness (|skew| > 1):")
                    for var in high_skew:
                        skew_val = skew_kurt.loc[var, 'Skewness']
                        print(f"- {var} (skew = {skew_val:.2f})")

                        if skew_val > 1:
                            print("  Recommended transformation: log, sqrt, or inverse")
                        else:  # skew < -1
                            print("  Recommended transformation: squared or cubed")

                # Test for normality
                print("\n--- Normality Tests (Shapiro-Wilk) ---")
                for col in numeric_cols[:min(5, len(numeric_cols))]:
                    if len(self.df[col].dropna()) <= 5000:  # Shapiro-Wilk works best for smaller samples
                        stat, p = stats.shapiro(self.df[col].dropna())
                        print(f"{col}: W = {stat:.4f}, p-value = {p:.4f}")
                        if p < 0.05:
                            print(f"  {col} is not normally distributed (p < 0.05)")
                        else:
                            print(f"  {col} appears normally distributed (p >= 0.05)")
                    else:
                        print(f"{col}: Sample too large for Shapiro-Wilk test")

        # Analyze categorical columns
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns

        if len(cat_cols) > 0:
            print("\n--- Categorical Variables Analysis ---")

            for col in cat_cols[:min(5, len(cat_cols))]:
                value_counts = self.df[col].value_counts()
                print(f"\n{col} - Unique values: {len(value_counts)}")

                if len(value_counts) <= 10:
                    # Value counts
                    print(value_counts)

                    # Percentage
                    print("\nPercentage:")
                    print(value_counts / len(self.df) * 100)

                    # Plot bar chart
                    plt.figure(figsize=(10, 5))
                    sns.countplot(y=col, data=self.df, order=value_counts.index)
                    plt.title(f'Distribution of {col}')
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"Too many unique values ({len(value_counts)}) to display")
                    print("Top 10 most frequent values:")
                    print(value_counts.head(10))

    def _correlation_analysis(self):
        print("\n--- Correlation Analysis ---")

        # Correlation methods
        print("\nAvailable correlation methods:")
        print("1. Pearson (linear relationships)")
        print("2. Spearman (monotonic relationships)")
        print("3. Kendall's Tau (ordinal relationships)")

        corr_method = input("Select correlation method (1, 2, or 3): ").strip()
        method_map = {'1': 'pearson', '2': 'spearman', '3': 'kendall'}

        if corr_method in method_map:
            method = method_map[corr_method]
            numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

            if len(numeric_cols) > 1:
                # Calculate correlation matrix
                corr_matrix = self.df[numeric_cols].corr(method=method)

                # Print correlation matrix
                print(f"\n{method.capitalize()} Correlation Matrix:")
                print(corr_matrix.round(2))

                # Visualize correlation matrix
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
                plt.title(f'{method.capitalize()} Correlation Matrix')
                plt.tight_layout()
                plt.show()

                # Identify strong correlations
                print("\nStrong Correlations (|r| > 0.7):")
                strong_corr = []

                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            strong_corr.append((
                                corr_matrix.columns[i], 
                                corr_matrix.columns[j], 
                                corr_matrix.iloc[i, j]
                            ))

                if strong_corr:
                    for var1, var2, corr in strong_corr:
                        print(f"- {var1} & {var2}: {corr:.2f}")

                        # Create scatterplot for strongly correlated variables
                        plt.figure(figsize=(8, 5))
                        sns.scatterplot(x=var1, y=var2, data=self.df)
                        plt.title(f'Scatterplot of {var1} vs {var2} (r = {corr:.2f})')
                        plt.tight_layout()
                        plt.show()
                else:
                    print("No strong correlations (|r| > 0.7) found.")

                # Correlation with potential dependent variables
                if len(numeric_cols) > 5:
                    print("\nWould you like to identify potential dependent variables?")
                    find_dep = input("(y/n): ").strip().lower()

                    if find_dep == 'y':
                        print("\nSelect a potential dependent variable:")
                        for i, col in enumerate(numeric_cols, 1):
                            print(f"{i}. {col}")

                        dep_idx = int(input("Enter number: ").strip()) - 1
                        if 0 <= dep_idx < len(numeric_cols):
                            dep_var = numeric_cols[dep_idx]

                            # Calculate correlation with all other variables
                            correlations = corr_matrix[dep_var].drop(dep_var).sort_values(ascending=False)

                            print(f"\nCorrelations with {dep_var}:")
                            print(correlations)

                            # Visualize top correlations
                            plt.figure(figsize=(10, 6))
                            correlations.plot(kind='bar')
                            plt.title(f'Correlation with {dep_var}')
                            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                            plt.tight_layout()
                            plt.show()

                            # Create pairplot for top correlated variables
                            top_vars = list(correlations.index[:min(5, len(correlations))])
                            top_vars.append(dep_var)

                            print("\nCreating pairplot for top correlated variables...")
                            sns.pairplot(self.df[top_vars])
                            plt.suptitle(f'Pairplot of {dep_var} with Top Correlated Variables', y=1.02)
                            plt.tight_layout()
                            plt.show()
            else:
                print("Need at least two numeric columns for correlation analysis.")

    def _time_series_analysis(self):
        print("\n--- Time Series Analysis ---")

        is_time_series = input("Does this dataset contain time series data? (y/n): ").strip().lower()

        if is_time_series == 'y':
            # Ask for time variable
            print("\nAvailable columns:")
            for i, col in enumerate(self.df.columns, 1):
                print(f"{i}. {col} ({self.df[col].dtype})")

            time_var_idx = int(input("\nSelect the time variable (enter number): ").strip()) - 1
            time_var = self.df.columns[time_var_idx]

            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(self.df[time_var]):
                try:
                    self.df[time_var] = pd.to_datetime(self.df[time_var])
                    print(f"Converted {time_var} to datetime format")
                except:
                    print(f"Could not convert {time_var} to datetime. Using as is.")

            # Set time variable as index
            ts_df = self.df.set_index(time_var)

            # Select variable to analyze
            numeric_cols = ts_df.select_dtypes(include=['int64', 'float64']).columns

            print("\nSelect variable for time series analysis:")
            for i, col in enumerate(numeric_cols, 1):
                print(f"{i}. {col}")

            ts_var_idx = int(input("Enter number: ").strip()) - 1
            ts_var = numeric_cols[ts_var_idx]

            # Time series plot
            plt.figure(figsize=(12, 6))
            ts_df[ts_var].plot()
            plt.title(f'Time Series: {ts_var}')
            plt.tight_layout()
            plt.show()

            # Check for stationarity
            try:
                print("\n--- Stationarity Test (Augmented Dickey-Fuller) ---")
                result = adfuller(ts_df[ts_var].dropna())

                print(f'ADF Statistic: {result[0]:.4f}')
                print(f'p-value: {result[1]:.4f}')

                if result[1] < 0.05:
                    print("Series is stationary (p < 0.05)")
                else:
                    print("Series is non-stationary (p >= 0.05)")
                    print("Consider differencing the series for regression")

                    # Ask if user wants to see differenced series
                    show_diff = input("\nShow differenced series? (y/n): ").strip().lower()
                    if show_diff == 'y':
                        # Calculate first difference
                        ts_diff = ts_df[ts_var].diff().dropna()

                        # Plot differenced series
                        plt.figure(figsize=(12, 6))
                        ts_diff.plot()
                        plt.title(f'First Difference of {ts_var}')
                        plt.tight_layout()
                        plt.show()

                        # Test stationarity of differenced series
                        diff_result = adfuller(ts_diff.dropna())
                        print("\nStationarity Test for Differenced Series:")
                        print(f'ADF Statistic: {diff_result[0]:.4f}')
                        print(f'p-value: {diff_result[1]:.4f}')

                        if diff_result[1] < 0.05:
                            print("Differenced series is stationary (p < 0.05)")
                        else:
                            print("Differenced series is still non-stationary (p >= 0.05)")
                            print("Consider higher order differencing or other transformations")
            except Exception as e:
                print(f"Could not perform stationarity test: {e}")

            # Autocorrelation and partial autocorrelation
            try:
                # Plot ACF and PACF
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

                plot_acf(ts_df[ts_var].dropna(), ax=ax1, lags=20)
                ax1.set_title(f'Autocorrelation Function for {ts_var}')

                plot_pacf(ts_df[ts_var].dropna(), ax=ax2, lags=20)
                ax2.set_title(f'Partial Autocorrelation Function for {ts_var}')

                plt.tight_layout()
                plt.show()

                print("\nInterpretation of ACF/PACF:")
                print("- Significant spikes in ACF indicate potential AR terms")
                print("- Significant spikes in PACF indicate potential MA terms")
                print("- For regression with time series data, consider including lagged terms")
            except Exception as e:
                print(f"Could not plot ACF/PACF: {e}")

    def _outlier_detection(self):
        print("\n--- Outlier Detection ---")

        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

        if len(numeric_cols) > 0:
            # Methods for outlier detection
            print("\nOutlier detection methods:")
            print("1. Z-score method (assumes normal distribution)")
            print("2. IQR method (robust to non-normal distributions)")
            print("3. Both methods")

            outlier_method = input("Select method (1, 2, or 3): ").strip()

            if outlier_method in ['1', '3']:
                # Z-score method
                print("\n--- Z-score Method ---")
                z_scores = pd.DataFrame()

                for col in numeric_cols:
                    z_scores[col] = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())

                # Find outliers (Z-score > 3)
                z_outliers = (z_scores > 3).any(axis=1)
                z_outlier_count = z_outliers.sum()

                print(f"Found {z_outlier_count} potential outliers using Z-score method (|Z| > 3)")

                if z_outlier_count > 0:
                    # Show outlier summary
                    print("\nOutlier summary by variable (Z-score method):")
                    for col in numeric_cols:
                        col_outliers = (z_scores[col] > 3).sum()
                        if col_outliers > 0:
                            print(f"- {col}: {col_outliers} outliers ({col_outliers/len(self.df)*100:.2f}%)")

                    # Visualize outliers with boxplots
                    plt.figure(figsize=(12, 6))
                    self.df[numeric_cols].boxplot(vert=False)
                    plt.title('Boxplot for Numeric Variables')
                    plt.tight_layout()
                    plt.show()

            if outlier_method in ['2', '3']:
                # IQR method
                print("\n--- IQR Method ---")
                outlier_counts = {}

                for col in numeric_cols:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound))