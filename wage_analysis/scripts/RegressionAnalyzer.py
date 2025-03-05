import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from scipy import stats
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class RegressionAnalyzer:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.model_results = None
        self.file_path = None
        self.dependent_var = None
        self.independent_vars = []
        self.categorical_vars = []
        self.interaction_terms = []
        self.polynomial_vars = {}
        
    def load_data(self, interactive=True):
        """
        Load data from a file. If interactive is True, prompts the user for the file path.
        
        Parameters:
        -----------
        interactive : bool
            Whether to prompt the user for input or use the provided file_path
        
        Returns:
        --------
        pandas.DataFrame
            The loaded dataset
        """
        if interactive:
            print("\n=== DATA LOADING ===")
            file_path = input("Enter the path to your data file: ").strip()
            self.file_path = file_path
        
        if not os.path.exists(self.file_path):
            print(f"File '{self.file_path}' not found. Please check the path.")
            return None
        
        # Determine file type by extension
        if self.file_path.endswith('.csv'):
            self.df = pd.read_csv(self.file_path)
        elif self.file_path.endswith('.dta'):
            self.df = pd.read_stata(self.file_path)
        elif self.file_path.endswith('.xlsx') or self.file_path.endswith('.xls'):
            self.df = pd.read_excel(self.file_path)
        else:
            print(f"Unsupported file format. Supported formats: .csv, .dta, .xlsx, .xls")
            return None
        
        print(f"\nSuccessfully loaded data with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
        return self.df
    
    def explore_data(self):
        """
        Provide basic data exploration, including summary statistics,
        missing values, and correlation analysis.
        """
        if self.df is None:
            print("Please load data first.")
            return
        
        print("\n=== DATA EXPLORATION ===")
        
        # Basic info and summary statistics
        print("\n--- Basic Information ---")
        print(f"Data shape: {self.df.shape}")
        print("\n--- Data Types ---")
        print(self.df.dtypes)
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print("\n--- Missing Values ---")
            print(missing_values[missing_values > 0])
            
            # Ask if user wants to handle missing values
            handle_missing = input("\nWould you like to handle missing values? (y/n): ").strip().lower()
            if handle_missing == 'y':
                self._handle_missing_values()
        else:
            print("\nNo missing values found.")
        
        # Summary statistics
        print("\n--- Summary Statistics ---")
        print(self.df.describe())
        
        # Correlation analysis for numeric columns
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1:
            print("\n--- Correlation Matrix ---")
            corr_matrix = self.df[numeric_cols].corr()
            print(corr_matrix.round(2))
            
            # Visualize correlation matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.show()
    
    def _handle_missing_values(self):
        """
        Handle missing values based on user input.
        """
        print("\n--- Missing Value Handling Options ---")
        print("1. Drop rows with missing values")
        print("2. Fill numeric missing values with mean")
        print("3. Fill numeric missing values with median")
        print("4. Fill categorical missing values with mode")
        
        option = input("Select an option (1-4): ").strip()
        
        if option == '1':
            self.df = self.df.dropna()
            print(f"Dropped rows with missing values. New shape: {self.df.shape}")
        elif option == '2':
            # Fill numeric columns with mean
            numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
            print("Filled numeric missing values with mean.")
        elif option == '3':
            # Fill numeric columns with median
            numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
            print("Filled numeric missing values with median.")
        elif option == '4':
            # Fill categorical columns with mode
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            print("Filled categorical missing values with mode.")
        else:
            print("Invalid option. No changes made.")
    
    def specify_model(self, interactive=True):
        """
        Specify the dependent and independent variables for the regression model.
        If interactive is True, prompts the user for input.
        
        Parameters:
        -----------
        interactive : bool
            Whether to prompt the user for input
        """
        if self.df is None:
            print("Please load data first.")
            return
        
        if interactive:
            print("\n=== MODEL SPECIFICATION ===")
            
            # Display available columns
            print("\nAvailable columns:")
            for i, col in enumerate(self.df.columns, 1):
                print(f"{i}. {col} ({self.df[col].dtype})")
            
            # Specify dependent variable
            dep_var_idx = int(input("\nSelect the dependent variable (enter number): ")) - 1
            self.dependent_var = self.df.columns[dep_var_idx]
            
            # Check if log transformation is needed for dependent variable
            if self.df[self.dependent_var].min() > 0:
                log_transform = input(f"Would you like to log-transform {self.dependent_var}? (y/n): ").strip().lower()
                if log_transform == 'y':
                    log_var_name = f"log_{self.dependent_var}"
                    self.df[log_var_name] = np.log(self.df[self.dependent_var])
                    self.dependent_var = log_var_name
                    print(f"Created log-transformed variable: {log_var_name}")
            
            # Specify independent variables
            print("\nSelect independent variables (comma-separated numbers, e.g., 1,3,5): ")
            indep_var_indices = [int(idx) - 1 for idx in input().strip().split(',')]
            self.independent_vars = [self.df.columns[idx] for idx in indep_var_indices]
            
            # Check for categorical variables
            print("\nAre any of these variables categorical? (comma-separated numbers or 'none'): ")
            cat_input = input().strip().lower()
            if cat_input != 'none':
                cat_indices = [int(idx) - 1 for idx in cat_input.split(',')]
                self.categorical_vars = [self.df.columns[idx] for idx in cat_indices]
                
                # Convert categorical variables to dummies
                self._handle_categorical_vars()
            
            # Check for polynomial terms
            poly_input = input("\nDo you want to add polynomial terms for any variables? (y/n): ").strip().lower()
            if poly_input == 'y':
                self._add_polynomial_terms()
                
            # Check for interaction terms
            interact_input = input("\nDo you want to add interaction terms? (y/n): ").strip().lower()
            if interact_input == 'y':
                self._add_interaction_terms()
                
            print(f"\nDependent variable: {self.dependent_var}")
            print(f"Independent variables: {', '.join(self.independent_vars)}")
            
            # Update X and y
            self._prepare_variables()
            
            return self.dependent_var, self.independent_vars
    
    def _handle_categorical_vars(self):
        """
        Handle categorical variables by creating dummy variables.
        """
        for cat_var in self.categorical_vars:
            # Check if it's already dummy-coded
            unique_values = self.df[cat_var].nunique()
            if unique_values <= 2:
                print(f"{cat_var} appears to be already dummy-coded (binary).")
                continue
                
            # Ask if user wants to drop first category (to avoid dummy variable trap)
            drop_first = input(f"Drop first category for {cat_var} to avoid dummy variable trap? (y/n): ").strip().lower()
            drop_first = True if drop_first == 'y' else False
            
            # Create dummies
            dummies = pd.get_dummies(self.df[cat_var], prefix=cat_var, drop_first=drop_first)
            
            # Add dummy columns to dataframe
            self.df = pd.concat([self.df, dummies], axis=1)
            
            # Update independent variables list
            self.independent_vars.remove(cat_var)
            self.independent_vars.extend(dummies.columns.tolist())
            
            print(f"Created dummy variables for {cat_var}: {', '.join(dummies.columns)}")
    
    def _add_polynomial_terms(self):
        """
        Add polynomial terms for selected variables.
        """
        print("\nAvailable variables for polynomial terms:")
        for i, var in enumerate(self.independent_vars, 1):
            if var not in self.categorical_vars and pd.api.types.is_numeric_dtype(self.df[var]):
                print(f"{i}. {var}")
        
        poly_vars = input("Select variables for polynomial terms (comma-separated numbers): ").strip()
        if poly_vars:
            selected_indices = [int(idx) - 1 for idx in poly_vars.split(',')]
            for idx in selected_indices:
                if idx < len(self.independent_vars):
                    var = self.independent_vars[idx]
                    degree = int(input(f"Enter polynomial degree for {var} (2-5): ").strip())
                    
                    if 2 <= degree <= 5:
                        self.polynomial_vars[var] = degree
                        # Create polynomial terms
                        for p in range(2, degree + 1):
                            poly_var_name = f"{var}{p}"
                            self.df[poly_var_name] = self.df[var] ** p
                            self.independent_vars.append(poly_var_name)
                            print(f"Created polynomial term: {poly_var_name}")
    
    def _add_interaction_terms(self):
        """
        Add interaction terms between selected variables.
        """
        print("\nSelect pairs of variables for interaction terms:")
        for i, var in enumerate(self.independent_vars, 1):
            print(f"{i}. {var}")
        
        interact_pairs = input("Enter pairs as 'num1,num2;num3,num4' (e.g., '1,2;3,4'): ").strip()
        if interact_pairs:
            pairs = interact_pairs.split(';')
            for pair in pairs:
                if ',' in pair:
                    idx1, idx2 = [int(idx) - 1 for idx in pair.split(',')]
                    if idx1 < len(self.independent_vars) and idx2 < len(self.independent_vars):
                        var1 = self.independent_vars[idx1]
                        var2 = self.independent_vars[idx2]
                        
                        # Create interaction term
                        interact_name = f"{var1}_x_{var2}"
                        self.df[interact_name] = self.df[var1] * self.df[var2]
                        self.independent_vars.append(interact_name)
                        self.interaction_terms.append((var1, var2))
                        print(f"Created interaction term: {interact_name}")
    
    def _prepare_variables(self):
        """
        Prepare X and y variables for regression.
        """
        # Prepare y (dependent variable)
        self.y = self.df[self.dependent_var]
        
        # Prepare X (independent variables)
        self.X = self.df[self.independent_vars]
        
        # Add constant term for intercept
        self.X = sm.add_constant(self.X)
        
        print(f"\nPrepared data: X shape = {self.X.shape}, y shape = {self.y.shape}")
    
    def check_assumptions(self):
        """
        Check assumptions for OLS regression.
        """
        if self.X is None or self.y is None:
            print("Please specify model first.")
            return
        
        print("\n=== REGRESSION ASSUMPTIONS CHECK ===")
        
        # 1. Check for multicollinearity (VIF)
        print("\n--- Multicollinearity Check (VIF) ---")
        vif_data = pd.DataFrame()
        vif_data["Variable"] = self.X.columns
        vif_data["VIF"] = [variance_inflation_factor(self.X.values, i) for i in range(self.X.shape[1])]
        print(vif_data)
        
        # Highlight problematic VIF values
        high_vif = vif_data[vif_data["VIF"] > 10].shape[0]
        if high_vif > 0:
            print(f"\nWarning: {high_vif} variables have VIF > 10, indicating multicollinearity.")
            
            # Suggest remedies
            print("\nPossible remedies for multicollinearity:")
            print("- Remove one of the highly correlated variables")
            print("- Use Ridge Regression instead of OLS")
            print("- Create a composite variable")
            
            # Ask if user wants to continue
            continue_anyway = input("\nContinue with regression despite multicollinearity? (y/n): ").strip().lower()
            if continue_anyway != 'y':
                return False
        
        print("\nMulticollinearity check complete.")
        return True
    
    def run_ols(self):
        """
        Run OLS regression and display results.
        """
        if self.X is None or self.y is None:
            print("Please specify model first.")
            return
        
        print("\n=== OLS REGRESSION ANALYSIS ===")
        
        # Run OLS regression
        model = sm.OLS(self.y, self.X)
        self.model_results = model.fit()
        
        # Display results
        print(self.model_results.summary())
        
        # Test for heteroskedasticity
        self._check_heteroskedasticity()
        
        # Plot diagnostic plots
        self._plot_diagnostics()
        
        # Check if user wants to try other regression models
        try_other = input("\nWould you like to try other regression models (Ridge, Lasso, ElasticNet)? (y/n): ").strip().lower()
        if try_other == 'y':
            self._run_regularized_regression()
        
        return self.model_results
    
    def _check_heteroskedasticity(self):
        """
        Test for heteroskedasticity using White's test and Breusch-Pagan test.
        """
        print("\n--- Heteroskedasticity Tests ---")
        
        # White's test
        try:
            white_test = het_white(self.model_results.resid, self.X)
            lm_stat, lm_pval, f_stat, f_pval = white_test
            
            print("\nWhite's test for heteroskedasticity:")
            print(f"LM statistic: {lm_stat:.4f}")
            print(f"LM test p-value: {lm_pval:.4f}")
            
            if lm_pval < 0.05:
                print("Evidence of heteroskedasticity detected (p < 0.05)")
                
                # Ask if user wants to use robust standard errors
                use_robust = input("\nWould you like to use robust standard errors? (y/n): ").strip().lower()
                if use_robust == 'y':
                    # Re-run with robust standard errors
                    self.model_results = sm.OLS(self.y, self.X).fit(cov_type='HC3')
                    print("\nModel re-estimated with heteroskedasticity-robust standard errors (HC3):")
                    print(self.model_results.summary())
            else:
                print("No significant evidence of heteroskedasticity (p >= 0.05)")
        except:
            print("Could not perform White's test due to an error.")
        
        # Breusch-Pagan test
        try:
            bp_test = het_breuschpagan(self.model_results.resid, self.X)
            lm_stat, lm_pval, f_stat, f_pval = bp_test
            
            print("\nBreusch-Pagan test for heteroskedasticity:")
            print(f"LM statistic: {lm_stat:.4f}")
            print(f"LM test p-value: {lm_pval:.4f}")
            
            if lm_pval < 0.05:
                print("Evidence of heteroskedasticity detected (p < 0.05)")
            else:
                print("No significant evidence of heteroskedasticity (p >= 0.05)")
        except:
            print("Could not perform Breusch-Pagan test due to an error.")
    
    def _plot_diagnostics(self):
        """
        Create diagnostic plots for the regression analysis.
        """
        print("\n--- Diagnostic Plots ---")
        
        # Create a figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Residuals vs Fitted plot
        ax1.scatter(self.model_results.fittedvalues, self.model_results.resid, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Fitted values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted Values')
        
        # 2. Q-Q plot
        stats.probplot(self.model_results.resid, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')
        
        # 3. Scale-Location plot (sqrt of abs residuals)
        ax3.scatter(self.model_results.fittedvalues, np.sqrt(np.abs(self.model_results.resid)), alpha=0.5)
        ax3.set_xlabel('Fitted values')
        ax3.set_ylabel('Sqrt(|Residuals|)')
        ax3.set_title('Scale-Location Plot')
        
        # 4. Residuals vs Leverage plot
        try:
            from statsmodels.graphics.regressionplots import plot_leverage_resid2
            plot_leverage_resid2(self.model_results, ax=ax4)
            ax4.set_title('Residuals vs Leverage')
        except:
            # If leverage plot fails, show histogram of residuals instead
            ax4.hist(self.model_results.resid, bins=20)
            ax4.set_xlabel('Residuals')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Histogram of Residuals')
        
        # Adjust the layout
        plt.tight_layout()
        plt.show()
        
        # Plot distribution of residuals
        plt.figure(figsize=(10, 6))
        sns.histplot(self.model_results.resid, kde=True)
        plt.title('Distribution of Residuals')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.show()
    
    def _run_regularized_regression(self):
        """
        Run regularized regression models (Ridge, Lasso, ElasticNet).
        """
        print("\n=== REGULARIZED REGRESSION MODELS ===")
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            self.X.values, self.y.values, test_size=0.2, random_state=42
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Ask for regularization parameter
        alpha = float(input("\nEnter regularization parameter (alpha) value (e.g., 0.1, 1.0): ").strip())
        
        print("\n--- Ridge Regression ---")
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        ridge_pred = ridge.predict(X_test_scaled)
        ridge_r2 = r2_score(y_test, ridge_pred)
        ridge_mse = mean_squared_error(y_test, ridge_pred)
        print(f"Ridge R² on test data: {ridge_r2:.4f}")
        print(f"Ridge MSE on test data: {ridge_mse:.4f}")
        
        print("\n--- Lasso Regression ---")
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train_scaled, y_train)
        lasso_pred = lasso.predict(X_test_scaled)
        lasso_r2 = r2_score(y_test, lasso_pred)
        lasso_mse = mean_squared_error(y_test, lasso_pred)
        print(f"Lasso R² on test data: {lasso_r2:.4f}")
        print(f"Lasso MSE on test data: {lasso_mse:.4f}")
        
        print("\n--- ElasticNet Regression ---")
        l1_ratio = float(input("Enter L1 ratio for ElasticNet (0 to 1, where 0=Ridge, 1=Lasso): ").strip())
        elastic = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        elastic.fit(X_train_scaled, y_train)
        elastic_pred = elastic.predict(X_test_scaled)
        elastic_r2 = r2_score(y_test, elastic_pred)
        elastic_mse = mean_squared_error(y_test, elastic_pred)
        print(f"ElasticNet R² on test data: {elastic_r2:.4f}")
        print(f"ElasticNet MSE on test data: {elastic_mse:.4f}")
        
        # Compare coefficients
        print("\n--- Coefficient Comparison ---")
        coef_df = pd.DataFrame({
            'Variable': self.X.columns,
            'OLS': self.model_results.params,
            'Ridge': np.r_[ridge.intercept_, ridge.coef_[1:]],
            'Lasso': np.r_[lasso.intercept_, lasso.coef_[1:]],
            'ElasticNet': np.r_[elastic.intercept_, elastic.coef_[1:]]
        })
        print(coef_df)
        
        # Plot coefficient comparison
        plt.figure(figsize=(12, 8))
        
        # Transpose for easier plotting
        coef_plot = coef_df.set_index('Variable').T
        
        # Plot
        coef_plot.plot(kind='bar', figsize=(15, 8))
        plt.title('Coefficient Comparison Across Models')
        plt.ylabel('Coefficient Value')
        plt.xlabel('Model')
        plt.xticks(rotation=0)
        plt.legend(loc='best')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Recommend best model
        print("\n--- Model Comparison Summary ---")
        print("Model       R²      MSE")
        print(f"OLS         {self.model_results.rsquared:.4f}   {np.mean(self.model_results.resid**2):.4f}")
        print(f"Ridge       {ridge_r2:.4f}   {ridge_mse:.4f}")
        print(f"Lasso       {lasso_r2:.4f}   {lasso_mse:.4f}")
        print(f"ElasticNet  {elastic_r2:.4f}   {elastic_mse:.4f}")
        
        # Find best model
        models = {
            'OLS': self.model_results.rsquared,
            'Ridge': ridge_r2,
            'Lasso': lasso_r2,
            'ElasticNet': elastic_r2
        }
        best_model = max(models, key=models.get)
        print(f"\nBased on R² values, the {best_model} model performs best.")