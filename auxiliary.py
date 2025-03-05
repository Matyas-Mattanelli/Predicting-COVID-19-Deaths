import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Function calculating and reshaping the correlation matrix
def get_correlations(df: pd.DataFrame):
    corr_mat = df.corr().stack().reset_index()
    corr_mat.columns = ['FEATURE_1', 'FEATURE_2', 'CORRELATION'] # Rename the columns
    mask_dups = (corr_mat[['FEATURE_1', 'FEATURE_2']].apply(frozenset, axis=1).duplicated()) | (corr_mat['FEATURE_1']==corr_mat['FEATURE_2']) # Create a mask to identify rows with duplicate features
    corr_mat = corr_mat[~mask_dups]
    corr_mat = corr_mat.iloc[(-corr_mat['CORRELATION'].abs()).argsort().values, :]

    return corr_mat

# Function estimating a linear regression and printing the results
def OLS(df: pd.DataFrame, dep_var: str, vars: list[str] | None = None, mask: list[bool] | None = None, add_constant: bool = True, plot_title: str = ''):
    # Add a constant to the data set if required
    df = sm.add_constant(df)
    
    # Specify all variables if not provided
    if vars is None:
        vars = list[df.columns]
    elif add_constant:
        vars = ['const'] + vars
    
    # Create mask if not provided
    if mask is None:
        mask = [True] * df.shape[0]

    # Specify the model
    model = sm.OLS(df.loc[mask, dep_var], df.loc[mask, vars])
    model_res = model.fit()

    # Print summary
    print(model_res.summary())
    print('\n')

    # Show residual plot
    model_res.resid.plot(style='.', title=plot_title)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.xlabel('')

    # Show MSE and MAE
    print(f'MAE: {model_res.resid.abs().mean()}')
    print(f'MSE: {(model_res.resid ** 2).mean()}')

    # Show pvalues
    print('\np-values:')
    print(model_res.pvalues.sort_values())

    return model_res