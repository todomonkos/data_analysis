import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.stattools import durbin_watson

class RegressionDiagnostics:
    """
    Comprehensive econometric diagnostics of OLS regression models.
    Allows users to selectively run tests for various regression assumptions.
    """
    
    def __init__(self, X=None, y=None):
        """
        Initialize the RegressionDiagnostics class.
        
        Parameters:
        ----------
        X : pandas DataFrame
            Independent variables (including constant term)
        y : pandas Series or array-like
            Dependent variable
        """
        self.X = X
        self.y = y
        self.X_original = None
        self.results = None
    
    def set_data(self, X, y):
        """
        Set or update the data for analysis.
        
        Parameters:
        ----------
        X : pandas DataFrame
            Independent variables (including constant term)
        y : pandas Series or array-like
            Dependent variable
        """
        self.X = X
        self.y = y
        return self
    
    def set_results(self, results):
        """Set statsmodels regression results directly"""
        self.results = results
        return self
    
    def fit_model(self):
        """
        Fit the OLS model and store results.
        """
        if self.X is None or self.y is None:
            print("Please set data first using set_data() method.")
            return None
        
        model = sm.OLS(self.y, self.X)
        self.results = model.fit()
        return self.results
    
    def check_assumptions(self, checks=None, interactive=True):
        """
        Check specified OLS regression assumptions with comprehensive econometric tests.
        
        Parameters:
        ----------
        checks : list, optional
            List of assumption checks to run. Options:
            ['linearity', 'multicollinearity', 'heteroskedasticity', 'autocorrelation', 'normality', 'all']
            Default is None, which will prompt user if interactive=True or run all if False.
        interactive : bool, optional
            Whether to prompt user for inputs during analysis (default True)
        
        Returns:
        -------
        dict
            Dictionary containing test results
        """
        if self.X is None or self.y is None:
            print("Please set data first using set_data() method.")
            return None
        
        # If results don't exist, fit the model
        if self.results is None:
            self.fit_model()
        
        # Initialize results dictionary
        test_results = {}
        
        print("\n=== REGRESSION ASSUMPTIONS CHECK ===")
        
        # Determine which checks to run
        if checks is None:
            if interactive:
                print("\nAvailable assumption checks:")
                print("1. Linearity (Ramsey RESET Test)")
                print("2. Multicollinearity (VIF)")
                print("3. Heteroskedasticity (White's and Breusch-Pagan tests)")
                print("4. Autocorrelation (Durbin-Watson test)")
                print("5. Normality of residuals (Jarque-Bera test)")
                print("6. All of the above")
                
                choice = input("\nEnter numbers for checks to run (comma-separated, e.g., '1,3,5'): ").strip()
                if choice == "6":
                    checks = ['linearity', 'multicollinearity', 'heteroskedasticity', 'autocorrelation', 'normality']
                else:
                    check_map = {
                        "1": "linearity", 
                        "2": "multicollinearity", 
                        "3": "heteroskedasticity",
                        "4": "autocorrelation", 
                        "5": "normality"
                    }
                    checks = [check_map[c.strip()] for c in choice.split(",") if c.strip() in check_map]
            else:
                checks = ['linearity', 'multicollinearity', 'heteroskedasticity', 'autocorrelation', 'normality']
        elif 'all' in checks:
            checks = ['linearity', 'multicollinearity', 'heteroskedasticity', 'autocorrelation', 'normality']
        
        # Run selected checks
        if 'linearity' in checks:
            test_results['linearity'] = self.check_linearity()
            
        if 'multicollinearity' in checks:
            test_results['multicollinearity'] = self.check_multicollinearity(interactive)
            
        if 'heteroskedasticity' in checks:
            test_results['heteroskedasticity'] = self.check_heteroskedasticity()
            
        if 'autocorrelation' in checks:
            test_results['autocorrelation'] = self.check_autocorrelation(interactive)
            
        if 'normality' in checks:
            test_results['normality'] = self.check_normality()
        
        return test_results
    
    def check_linearity(self):
        """
        Check the linearity assumption using Ramsey RESET test.
        
        Returns:
        -------
        dict
            Dictionary containing test results
        """
        print("\n--- Linearity Check (Ramsey RESET Test) ---")
        
        try:
            # Get fitted values
            y_hat = self.results.fittedvalues
            
            # Create powers of fitted values for RESET test
            X_reset = self.X.copy()
            X_reset['y_hat_2'] = y_hat**2
            X_reset['y_hat_3'] = y_hat**3
            
            # Run auxiliary regression
            reset_model = sm.OLS(self.y, X_reset)
            reset_results = reset_model.fit()
            
            # Calculate F statistic
            restricted = self.results
            unrestricted = reset_results
            df_resid_restricted = restricted.df_resid
            df_resid_unrestricted = unrestricted.df_resid
            df_diff = df_resid_restricted - df_resid_unrestricted
            ss_restricted = sum(restricted.resid**2)
            ss_unrestricted = sum(unrestricted.resid**2)
            f_value = ((ss_restricted - ss_unrestricted) / df_diff) / (ss_unrestricted / df_resid_unrestricted)
            
            # Calculate p-value
            p_value = 1 - stats.f.cdf(f_value, df_diff, df_resid_unrestricted)
            
            print(f"Ramsey RESET Test: F-statistic = {f_value:.4f}, p-value = {p_value:.4f}")
            
            if p_value < 0.05:
                print("Model may have incorrect functional form (p < 0.05)")
                print("Consider adding nonlinear transformations or interaction terms")
                conclusion = "Nonlinearity detected"
            else:
                print("No significant evidence of nonlinearity (p >= 0.05)")
                conclusion = "Linearity assumption satisfied"
                
            return {
                'test': 'Ramsey RESET',
                'f_value': f_value,
                'p_value': p_value,
                'conclusion': conclusion
            }
            
        except Exception as e:
            print(f"Could not perform RESET test due to an error: {e}")
            return {
                'test': 'Ramsey RESET',
                'error': str(e)
            }
    
    def check_multicollinearity(self, interactive=True):
        """
        Check for multicollinearity using Variance Inflation Factors (VIF).
        
        Parameters:
        ----------
        interactive : bool, optional
            Whether to prompt user for inputs (default True)
            
        Returns:
        -------
        dict
            Dictionary containing test results
        """
        print("\n--- Multicollinearity Check (VIF) ---")
        
        try:
            vif_data = pd.DataFrame()
            vif_data["Variable"] = self.X.columns
            vif_data["VIF"] = [variance_inflation_factor(self.X.values, i) for i in range(self.X.shape[1])]
            print(vif_data)
            
            # Highlight problematic VIF values
            high_vif = vif_data[vif_data["VIF"] > 10].shape[0]
            high_vif_vars = vif_data[vif_data["VIF"] > 10]["Variable"].tolist()
            
            if high_vif > 0:
                print(f"\nWarning: {high_vif} variables have VIF > 10, indicating multicollinearity.")
                print("Problematic variables:", ", ".join(high_vif_vars))
                
                # Suggest remedies
                print("\nPossible remedies for multicollinearity:")
                print("- Remove one of the highly correlated variables")
                print("- Use Ridge Regression instead of OLS")
                print("- Create a composite variable through PCA")
                
                # Ask if user wants to try PCA
                if interactive:
                    try_pca = input("\nWould you like to try PCA to address multicollinearity? (y/n): ").strip().lower()
                    if try_pca == 'y':
                        self._apply_pca(interactive)
                
                conclusion = "Multicollinearity detected"
            else:
                print("No variables with VIF > 10. Multicollinearity does not appear to be a concern.")
                conclusion = "No significant multicollinearity detected"
            
            return {
                'test': 'VIF',
                'vif_data': vif_data,
                'high_vif_variables': high_vif_vars if high_vif > 0 else [],
                'conclusion': conclusion
            }
            
        except Exception as e:
            print(f"Could not perform VIF calculation due to an error: {e}")
            return {
                'test': 'VIF',
                'error': str(e)
            }
    
    def check_heteroskedasticity(self):
        """
        Check for heteroskedasticity using White's and Breusch-Pagan tests.
        
        Returns:
        -------
        dict
            Dictionary containing test results
        """
        print("\n--- Heteroskedasticity Tests ---")
        results = {}
        
        # White's test
        try:
            white_test = het_white(self.results.resid, self.X)
            lm_stat, lm_pval, f_stat, f_pval = white_test
            
            print("\nWhite's test for heteroskedasticity:")
            print(f"LM statistic: {lm_stat:.4f}")
            print(f"LM test p-value: {lm_pval:.4f}")
            
            if lm_pval < 0.05:
                print("Evidence of heteroskedasticity detected (p < 0.05)")
                white_conclusion = "Heteroskedasticity detected"
            else:
                print("No significant evidence of heteroskedasticity (p >= 0.05)")
                white_conclusion = "Homoskedasticity assumption satisfied"
                
            results['white_test'] = {
                'lm_stat': lm_stat,
                'lm_pval': lm_pval,
                'f_stat': f_stat,
                'f_pval': f_pval,
                'conclusion': white_conclusion
            }
            
        except Exception as e:
            print(f"Could not perform White's test due to an error: {e}")
            results['white_test'] = {'error': str(e)}
        
        # Breusch-Pagan test
        try:
            bp_test = het_breuschpagan(self.results.resid, self.X)
            lm_stat, lm_pval, f_stat, f_pval = bp_test
            
            print("\nBreusch-Pagan test for heteroskedasticity:")
            print(f"LM statistic: {lm_stat:.4f}")
            print(f"LM test p-value: {lm_pval:.4f}")
            
            if lm_pval < 0.05:
                print("Evidence of heteroskedasticity detected (p < 0.05)")
                print("\nRecommended solutions:")
                print("- Use heteroskedasticity-robust standard errors (HC0, HC1, HC2, or HC3)")
                print("- Transform dependent variable (e.g., log)")
                print("- Use weighted least squares (WLS)")
                bp_conclusion = "Heteroskedasticity detected"
            else:
                print("No significant evidence of heteroskedasticity (p >= 0.05)")
                bp_conclusion = "Homoskedasticity assumption satisfied"
                
            results['bp_test'] = {
                'lm_stat': lm_stat,
                'lm_pval': lm_pval,
                'f_stat': f_stat,
                'f_pval': f_pval,
                'conclusion': bp_conclusion
            }
            
        except Exception as e:
            print(f"Could not perform Breusch-Pagan test due to an error: {e}")
            results['bp_test'] = {'error': str(e)}
            
        # Overall conclusion
        if 'white_test' in results and 'bp_test' in results:
            if ('conclusion' in results['white_test'] and 'Heteroskedasticity detected' in results['white_test']['conclusion']) or \
               ('conclusion' in results['bp_test'] and 'Heteroskedasticity detected' in results['bp_test']['conclusion']):
                results['overall_conclusion'] = "Heteroskedasticity detected in at least one test"
            else:
                results['overall_conclusion'] = "Homoskedasticity assumption satisfied in all tests"
                
        return results
    
    def check_autocorrelation(self, interactive=True):
        """
        Check for autocorrelation using Durbin-Watson test.
        
        Parameters:
        ----------
        interactive : bool, optional
            Whether to prompt user for inputs (default True)
            
        Returns:
        -------
        dict
            Dictionary containing test results
        """
        if interactive:
            time_series = input("\nIs this a time series dataset? (y/n): ").strip().lower()
            run_test = (time_series == 'y')
        else:
            # If not interactive, run the test anyway
            run_test = True
            
        if run_test:
            print("\n--- Autocorrelation Test (Durbin-Watson) ---")
            try:
                dw_stat = durbin_watson(self.results.resid)
                print(f"Durbin-Watson statistic: {dw_stat:.4f}")
                
                if dw_stat < 1.5:
                    print("Evidence of positive autocorrelation (DW < 1.5)")
                    print("\nRecommended solutions:")
                    print("- Use Newey-West standard errors")
                    print("- Include lagged dependent variables")
                    print("- Use ARIMA or other time series models")
                    conclusion = "Positive autocorrelation detected"
                elif dw_stat > 2.5:
                    print("Evidence of negative autocorrelation (DW > 2.5)")
                    conclusion = "Negative autocorrelation detected"
                else:
                    print("No significant evidence of autocorrelation (1.5 <= DW <= 2.5)")
                    conclusion = "No significant autocorrelation detected"
                    
                # Ask if user wants to use Newey-West standard errors
                if interactive and (dw_stat < 1.5 or dw_stat > 2.5):
                    use_nw = input("\nWould you like to use Newey-West standard errors? (y/n): ").strip().lower()
                    if use_nw == 'y':
                        lags = int(input("Enter number of lags for Newey-West (usually 1-3): ").strip())
                        self.results = sm.OLS(self.y, self.X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
                        print("\nModel re-estimated with Newey-West standard errors:")
                        print(self.results.summary())
                
                return {
                    'test': 'Durbin-Watson',
                    'dw_stat': dw_stat,
                    'conclusion': conclusion
                }
                
            except Exception as e:
                print(f"Could not perform Durbin-Watson test due to an error: {e}")
                return {
                    'test': 'Durbin-Watson',
                    'error': str(e)
                }
        else:
            print("\nSkipping autocorrelation test as data is not time series.")
            return {
                'test': 'Durbin-Watson',
                'result': 'Test skipped (not time series data)'
            }
    
    def check_normality(self):
        """
        Check normality of residuals using Jarque-Bera test.
        
        Returns:
        -------
        dict
            Dictionary containing test results
        """
        print("\n--- Normality Test (Jarque-Bera) ---")
        
        try:
            jb_test = stats.jarque_bera(self.results.resid)
            jb_stat, jb_pval = jb_test[0], jb_test[1]
            
            print(f"Jarque-Bera statistic: {jb_stat:.4f}")
            print(f"Jarque-Bera p-value: {jb_pval:.4f}")
            
            if jb_pval < 0.05:
                print("Residuals are not normally distributed (p < 0.05)")
                print("\nNote: Non-normality is less concerning with large samples (n > 30) due to CLT")
                print("For small samples, consider bootstrapping confidence intervals")
                conclusion = "Residuals are not normally distributed"
            else:
                print("Residuals appear normally distributed (p >= 0.05)")
                conclusion = "Residuals are normally distributed"
                
            return {
                'test': 'Jarque-Bera',
                'jb_stat': jb_stat,
                'jb_pval': jb_pval,
                'conclusion': conclusion
            }
            
        except Exception as e:
            print(f"Could not perform Jarque-Bera test due to an error: {e}")
            return {
                'test': 'Jarque-Bera',
                'error': str(e)
            }
    
    def _apply_pca(self, interactive=True):
        """
        Apply Principal Component Analysis to address multicollinearity.
        
        Parameters:
        ----------
        interactive : bool, optional
            Whether to prompt user for inputs (default True)
            
        Returns:
        -------
        pandas.DataFrame
            DataFrame with PCA components
        """
        print("\n--- Applying PCA to Address Multicollinearity ---")
        
        # Standardize features
        X_no_const = self.X.iloc[:, 1:] if 'const' in self.X.columns else self.X  # Remove constant term if exists
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_no_const)
        
        # Get variance retention percentage
        if interactive:
            var_retain = float(input("Enter percentage of variance to retain (e.g., 95 for 95%): ").strip()) / 100
        else:
            var_retain = 0.95  # Default to 95% if not interactive
        
        # Apply PCA
        pca = PCA(n_components=var_retain, svd_solver='full')
        X_pca = pca.fit_transform(X_scaled)
        
        # Create new dataframe with PCA components
        component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        X_pca_df = pd.DataFrame(X_pca, columns=component_names)
        
        # Add constant term
        X_pca_df = sm.add_constant(X_pca_df)
        
        # Print information
        explained_var = sum(pca.explained_variance_ratio_) * 100
        print(f"\nPCA reduced dimensions from {X_no_const.shape[1]} to {X_pca.shape[1]} components")
        print(f"Retained {explained_var:.2f}% of original variance")
        
        # Display variance explained by each component
        var_df = pd.DataFrame({
            'Component': component_names,
            'Variance Explained (%)': pca.explained_variance_ratio_ * 100,
            'Cumulative Variance (%)': np.cumsum(pca.explained_variance_ratio_) * 100
        })
        print("\nVariance Explained by Components:")
        print(var_df)
        
        # Display loadings (relationship between original variables and components)
        loadings = pd.DataFrame(
            pca.components_.T, 
            columns=component_names,
            index=X_no_const.columns
        )
        print("\nComponent Loadings (contribution of original variables to components):")
        print(loadings)
        
        # Update X with PCA components
        self.X_original = self.X.copy()  # Store original X
        self.X = X_pca_df
        
        print("\nX data replaced with PCA components for regression analysis")
        print("Original X data stored in self.X_original if needed")
        
        # Refit the model with PCA components
        if interactive:
            refit = input("\nWould you like to refit the model with PCA components? (y/n): ").strip().lower()
            if refit == 'y':
                self.fit_model()
                print("\nModel refitted with PCA components:")
                print(self.results.summary())
                
        return X_pca_df
    
    def get_robust_se(self, cov_type='HC3'):
        """
        Refit the model with robust standard errors.
        
        Parameters:
        ----------
        cov_type : str, optional
            Type of robust standard errors. Options: 'HC0', 'HC1', 'HC2', 'HC3', 'HAC'
            Default is 'HC3' which is more conservative
            
        Returns:
        -------
        statsmodels.regression.linear_model.RegressionResults
            Results with robust standard errors
        """
        if self.X is None or self.y is None:
            print("Please set data first using set_data() method.")
            return None
            
        if cov_type == 'HAC':
            results = sm.OLS(self.y, self.X).fit(cov_type=cov_type, cov_kwds={'maxlags': 1})
        else:
            results = sm.OLS(self.y, self.X).fit(cov_type=cov_type)
            
        print(f"\nModel estimated with {cov_type} robust standard errors:")
        print(results.summary())
        
        self.results = results
        return results
    
    def summary(self):
        """
        Print a summary of the regression results.
        """
        if self.results is None:
            print("No regression results available. Please fit the model first.")
            return None
            
        return self.results.summary()