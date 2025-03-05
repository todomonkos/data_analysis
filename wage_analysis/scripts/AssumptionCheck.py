def check_assumptions(self):
    """
    Check assumptions for OLS regression with comprehensive econometric tests.
    """
    if self.X is None or self.y is None:
        print("Please specify model first.")
        return
    
    print("\n=== REGRESSION ASSUMPTIONS CHECK ===")
    
    # 1. Check for linearity using RESET test
    print("\n--- Linearity Check (Ramsey RESET Test) ---")
    try:
        # Run initial model
        model = sm.OLS(self.y, self.X)
        results = model.fit()
        
        # Get fitted values
        y_hat = results.fittedvalues
        
        # Create powers of fitted values for RESET test
        X_reset = self.X.copy()
        X_reset['y_hat_2'] = y_hat**2
        X_reset['y_hat_3'] = y_hat**3
        
        # Run auxiliary regression
        reset_model = sm.OLS(self.y, X_reset)
        reset_results = reset_model.fit()
        
        # Perform F-test
        from statsmodels.stats.anova import anova_lm
        restricted = results
        unrestricted = reset_results
        
        # Calculate F statistic
        df_resid_restricted = restricted.df_resid
        df_resid_unrestricted = unrestricted.df_resid
        df_diff = df_resid_restricted - df_resid_unrestricted
        ss_restricted = sum(restricted.resid**2)
        ss_unrestricted = sum(unrestricted.resid**2)
        f_value = ((ss_restricted - ss_unrestricted) / df_diff) / (ss_unrestricted / df_resid_unrestricted)
        
        # Calculate p-value
        from scipy import stats
        p_value = 1 - stats.f.cdf(f_value, df_diff, df_resid_unrestricted)
        
        print(f"Ramsey RESET Test: F-statistic = {f_value:.4f}, p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            print("Model may have incorrect functional form (p < 0.05)")
            print("Consider adding nonlinear transformations or interaction terms")
        else:
            print("No significant evidence of nonlinearity (p >= 0.05)")
    except Exception as e:
        print(f"Could not perform RESET test due to an error: {e}")
    
    # 2. Check for multicollinearity (VIF)
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
        print("- Create a composite variable through PCA")
        
        # Ask if user wants to try PCA
        try_pca = input("\nWould you like to try PCA to address multicollinearity? (y/n): ").strip().lower()
        if try_pca == 'y':
            self._apply_pca()
        else:
            # Ask if user wants to continue
            continue_anyway = input("\nContinue with regression despite multicollinearity? (y/n): ").strip().lower()
            if continue_anyway != 'y':
                return False
    
    # 3. Check for heteroskedasticity
    print("\n--- Heteroskedasticity Tests ---")
    
    # Run basic model for tests
    model = sm.OLS(self.y, self.X)
    results = model.fit()
    
    # White's test
    try:
        white_test = het_white(results.resid, self.X)
        lm_stat, lm_pval, f_stat, f_pval = white_test
        
        print("\nWhite's test for heteroskedasticity:")
        print(f"LM statistic: {lm_stat:.4f}")
        print(f"LM test p-value: {lm_pval:.4f}")
        
        if lm_pval < 0.05:
            print("Evidence of heteroskedasticity detected (p < 0.05)")
        else:
            print("No significant evidence of heteroskedasticity (p >= 0.05)")
    except:
        print("Could not perform White's test due to an error.")
    
    # Breusch-Pagan test
    try:
        bp_test = het_breuschpagan(results.resid, self.X)
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
        else:
            print("No significant evidence of heteroskedasticity (p >= 0.05)")
    except:
        print("Could not perform Breusch-Pagan test due to an error.")
    
    # 4. Check for autocorrelation (if time series)
    time_series = input("\nIs this a time series dataset? (y/n): ").strip().lower()
    if time_series == 'y':
        print("\n--- Autocorrelation Test (Durbin-Watson) ---")
        try:
            from statsmodels.stats.stattools import durbin_watson
            dw_stat = durbin_watson(results.resid)
            print(f"Durbin-Watson statistic: {dw_stat:.4f}")
            
            if dw_stat < 1.5:
                print("Evidence of positive autocorrelation (DW < 1.5)")
                print("\nRecommended solutions:")
                print("- Use Newey-West standard errors")
                print("- Include lagged dependent variables")
                print("- Use ARIMA or other time series models")
            elif dw_stat > 2.5:
                print("Evidence of negative autocorrelation (DW > 2.5)")
            else:
                print("No significant evidence of autocorrelation (1.5 <= DW <= 2.5)")
                
            # Ask if user wants to use Newey-West standard errors
            use_nw = input("\nWould you like to use Newey-West standard errors? (y/n): ").strip().lower()
            if use_nw == 'y':
                lags = int(input("Enter number of lags for Newey-West (usually 1-3): ").strip())
                results = sm.OLS(self.y, self.X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
                print("\nModel re-estimated with Newey-West standard errors:")
                print(results.summary())
        except Exception as e:
            print(f"Could not perform Durbin-Watson test due to an error: {e}")
    
    # 5. Check for normality of residuals
    print("\n--- Normality Test (Jarque-Bera) ---")
    try:
        jb_test = stats.jarque_bera(results.resid)
        jb_stat, jb_pval = jb_test[0], jb_test[1]
        
        print(f"Jarque-Bera statistic: {jb_stat:.4f}")
        print(f"Jarque-Bera p-value: {jb_pval:.4f}")
        
        if jb_pval < 0.05:
            print("Residuals are not normally distributed (p < 0.05)")
            print("\nNote: Non-normality is less concerning with large samples (n > 30) due to CLT")
            print("For small samples, consider bootstrapping confidence intervals")
        else:
            print("Residuals appear normally distributed (p >= 0.05)")
    except:
        print("Could not perform Jarque-Bera test due to an error.")
    
    print("\nMulticollinearity check complete.")
    return True

def _apply_pca(self):
    """
    Apply Principal Component Analysis to address multicollinearity.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    print("\n--- Applying PCA to Address Multicollinearity ---")
    
    # Standardize features
    X_no_const = self.X.iloc[:, 1:]  # Remove constant term
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_no_const)
    
    # Ask for variance retention
    var_retain = float(input("Enter percentage of variance to retain (e.g., 95 for 95%): ").strip()) / 100
    
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
    
    return X_pca_df

