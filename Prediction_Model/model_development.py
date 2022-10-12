### import packages
# import os
# import sys
import pandas as pd
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
import numpy as np
# import seaborn as sns
# import statsmodels
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import datetime
from datetime import datetime
# from datetime import date, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
# from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LassoCV
# from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings('ignore')
# from pandas.tseries.offsets import DateOffset
import pmdarima as pm
from IPython.display import display



### import pet sales data
data = pd.read_csv("C:\\Users\\zhwenxin\\OneDrive - Boehringer Ingelheim\\Desktop\\Code from Tong\\Data\\df_pet_sales.csv")
print("Data Overview")
display(data.head(1))



### SKU pivot table
SKU_pv = data.pivot_table(index=['Date'], columns=['sku_code'], aggfunc='sum' )[['gross_sales', 'cog_qty']]
# brand_pv = data.pivot_table(index=['Date'], columns=['prd_grp_level3'], aggfunc='sum' )[['gross_sales', 'cog_qty']]
SKU_qty = SKU_pv['cog_qty'].reset_index().reset_index()
SKU_qty.columns = ['index', 'Date'] + [str(int(x)) for x in SKU_qty.columns[2:]]
print("SKU pivot table:")
display(SKU_qty.head(1))



### SKU list
Nexgard_SKU_lst = ['142833', '142835', '142831', '142829', '142832', '142834', '142830', '142828']
Heartgard_SKU_lst = ['145450', '145448','145449','141360','141348','141354']
print(f"Nexgard SKU List: {Nexgard_SKU_lst}")
print(f"Heargard SKU List: {Heartgard_SKU_lst}")

### address date
Nexgard = SKU_qty[['Date'] + Nexgard_SKU_lst]
Heartgard = SKU_qty[['Date'] + Heartgard_SKU_lst]
Nexgard['Date'] = Nexgard['Date'].apply(lambda x: datetime(int(x//100), int(x%100), 1))
Heartgard['Date'] = Heartgard['Date'].apply(lambda x: datetime(int(x//100), int(x%100), 1))
Nexgard_invoice = Nexgard.set_index('Date')
Heartgard_invoice = Heartgard.set_index('Date')
# print("Nexgard Revenue data:")
# display(Nexgard_invoice.head(1))



### External data
ext = pd.read_csv("C:\\Users\\zhwenxin\\OneDrive - Boehringer Ingelheim\\Desktop\\Code from Tong\\Data\\external_forecast_new.csv")
# print(ext['Date']) # 3/1/2023
ext['Date'] = ext['Date'].apply(lambda x: datetime(int(x.split('/')[2]), int(x.split('/')[0]), 1))
print(f"External data columns: {ext.columns.tolist()}")
print("External data:")
display(ext.head(1))

ext.columns = ['Date', 'Puppies', 'Adult Dogs', 'Kittens', 'Adult Cats', 'n_promo', 'disposable_income_pct', 'Unemployment Rate',
       'CPI_US', 'HW Risk Index', 'temperature', 'humidity', 'precip_in_mm',
       'FT Risk Index', 'COMPETITOR_SALES_Heartgard', 'COMPETITOR_SALES_Nexgard',
       'COMPETITOR_SALES_FLP', 'COMPETITOR_SALES_FLG']
numerical_var = ext[['Date', 'disposable_income_pct', 'Unemployment Rate',
       'CPI_US', 'HW Risk Index', 'temperature', 'humidity', 'precip_in_mm',
       'FT Risk Index', 'Puppies', 'Adult Dogs', 'Kittens',
       'Adult Cats']]
num_var_lst = ['disposable_income_pct', 'Unemployment Rate',
       'CPI_US', 'HW Risk Index', 'temperature', 'humidity', 'precip_in_mm',
       'FT Risk Index', 'Puppies', 'Adult Dogs', 'Kittens',
       'Adult Cats']

### lag, diff, pct_diff, moving average
lags_lst = [1, 2, 3]
for var in num_var_lst:
    for lag in lags_lst:
        lag_col = var + '_lag' + str(lag)
        diff_col = var + '_diff' + str(lag)
        pct_dff_col = var + '_pct_diff' + str(lag)
        ma_col = var + '_ma' + str(lag)
        numerical_var[lag_col] = numerical_var[var].shift(lag) # shift index by lag
        numerical_var[diff_col] = numerical_var[var].diff(lag) # difference with lag_th previous row
        numerical_var[pct_dff_col] = numerical_var[diff_col]/numerical_var[lag_col] # percentage difference with lag_th previous row
        numerical_var[ma_col] = numerical_var[var].rolling(window=lag).mean() # rolling windows average

### Leap Shift
leaps_lst = [-1, -2, -3]
leap_var = ['HW Risk Index', 'temperature', 'humidity', 'precip_in_mm', 'FT Risk Index']
for var in leap_var:
    for leap in leaps_lst:
        name = var + '_leap' + str(leap)
        numerical_var[name] = numerical_var[var].shift(leap) # shift index by lag
var_lst = list(numerical_var.columns[1:])

### numerical features standardization
print("External data before standardization:")
display(numerical_var.head(1))
scale = StandardScaler()
numerical_var_scale = numerical_var[numerical_var['Date'].apply(lambda x: x.year > 2017)]
numerical_var_scale = numerical_var_scale.dropna()
numerical_var_scale[var_lst] = scale.fit_transform(numerical_var_scale[var_lst])
external_var = numerical_var_scale.copy()
ext_var_lst = external_var.columns[1:] 
print("External data after standardization:")
display(external_var.head(1))

### Add dummies representing month
for month in range(1, 13):
    external_var[str(month)] = 0
    external_var.loc[external_var['Date'].apply(lambda x: x.month == month), str(month)] = 1
print("External data after adding month as dummies")
display(external_var.head(1))


### plot forecast
def forecast_plot(Actual, Insample, OutofSample, title):
    fig, ax = plt.subplots(figsize=(15,10))
    npre = 4
    ax.set(title=title, xlabel='Date', ylabel='Revenues')
    Actual.plot(ax=ax, style='y-o', label='Observed') # observations
    Insample.plot(ax=ax, style='r--o', label='In-Sample Forecast') # predictions
    OutofSample.plot(ax=ax, style='g-o', label='Out of Sample Forecast') # predictions
    legend = ax.legend(loc='lower right')
    forecast = pd.DataFrame(Actual)
    forecast['Insample'] = Insample
    forecast['OutofSample'] = OutofSample



class Demand_Forecast():
    def __init__(self, df_all, ext_var_lst):
        self.df_all = df_all
        self.ext_var_lst = ext_var_lst


    def lasso_selection(self, target, var_select=None):
        if not var_select:
            print('Lasso on Total')
            print(ext_var_lst)
            y = self.df_all[target]
            X = self.df_all[ext_var_lst]

            clf = Lasso(alpha=100).fit(X, y) # alpha: constant multiplies the L1 term; controlling regularization strength
            select = (clf.coef_ > 0) # clf.coef_: parameter vector
            X = X[np.asarray(X.columns)[select]] # select features 
            X2 = sm.add_constant(X) # add a column of ones to X as consant parameter
            lasso_est = sm.OLS(y, X2).fit() # get the fitted linear model using Weighted Least Squares
        else:
            print('Lasso on Partial')
            y = self.df_all[target]
            X = self.df_all[var_select]
            clf = Lasso(alpha=100).fit(X, y)
            select = (clf.coef_ > 0)
            X = X[np.asarray(X.columns)[select]]
            # reg = LinearRegression().fit(X, y)
            X2 = sm.add_constant(X)
            lasso_est = sm.OLS(y, X2).fit()
        return lasso_est


    def linear_regression(self, target, var_select, start, end, period, is_plot=False):
        df_train = self.df_all[(self.df_all.index>=start) & (self.df_all.index<=end)]
        X, y = df_train[var_select], df_train[target]
        reg = LinearRegression().fit(X, y) # LR model built on observations between start & end
        Insample = reg.predict(X) # predictions between start & end
        result = pd.DataFrame(y)
        result['forecast'] = Insample 
        X2 = sm.add_constant(X) 
        est = sm.OLS(y, X2).fit() # OLS model
        summary = est.summary() # summary table 
        start_f = pd.to_datetime(end) +  pd.DateOffset(months=1) 
        end_f = pd.to_datetime(end) + pd.DateOffset(months=period)
        x_forecast = self.df_all[start_f:end_f][var_select] # X between end+1_month & end+period_month
        forecast = reg.predict(x_forecast) # predictions between end+1_month & end+period_month

        reg_forecast = pd.DataFrame(self.df_all.loc[start_f:end_f, target])
        OOS_lm = reg_forecast[target] # observations between end+1_month & end+period_month
        observed = self.df_all[target] # all observations

        if is_plot:
            forecast_plot(observed, result['forecast'], OOS_lm, target+'Linear Regression')

        df_forecast = pd.DataFrame(observed) # all observations
        df_forecast['Insample'] = result['forecast'] # predictions between start & end 
        df_forecast['OutofSample'] = OOS_lm # observations between end+1_month & end+period_month 

        return est, summary, observed, Insample, forecast, df_forecast
        # est: OLS model
        # summary: OLS summary
        # observed: all observations
        # Insample: predictions between start & end
        # forecast: predictions between end+1_month & end+period_month
        # dataframe: [target]all observations, [Insample]predictions between start & end, [OutOfSample]observations between end+1_month & end+period_month


    def ARIMAX_regression_training(self, target, var_select, start, end, vanilla=[0, 0, 0] , seasonal=[0, 0, 0, 0]):
        df_train = self.df_all[(self.df_all.index>=start) & (self.df_all.index<=end)]
        X, y = df_train[var_select], df_train[target]
        X2 = sm.add_constant(X)
        
        ### Parameter Optimization
        current_aic = float('inf')
        est = None
        i = 0
        total = (vanilla[0] + 1) * (vanilla[1] + 1) * (vanilla[2] + 1) * (seasonal[0] + 1) * (seasonal[1] + 1) * (seasonal[2] + 1)
        for p in range(vanilla[0] + 1):
            for d in range(vanilla[1] + 1):
                for q in range(vanilla[2] + 1):
                    for P in range(seasonal[0] + 1):
                        for D in range(seasonal[1] + 1):
                            for Q in range(seasonal[2] + 1):
                                i += 1
                                print("%.2f" % (i/total*100),'%', end='\r')
                                try:
                                    if P + D + Q == 0:
                                        mod = sm.tsa.statespace.SARIMAX(y, exog=X2, order=(p,d,q), freq='MS')
                                    else:
                                        mod = sm.tsa.statespace.SARIMAX(y, exog=X2, order=(p,d,q), seasonal_order=(P, D, Q, 12), freq='MS')
                                    res = mod.fit(disp=False)
                                    if res.aic < current_aic:
                                        current_aic = res.aic
                                        est = [(p, d, q), (P, D, Q, 12)]
                                except:
                                    pass
        print('****************************')
        print(est)
        ### Finish Parameter Optimization
        return est


    def ARIMAX_regression(self, target, var_select, start, end, est, period, is_plot=False):
        df_train = self.df_all[(self.df_all.index>=start) & (self.df_all.index<=end)]
        X_all = df_train[var_select]
        X_all = sm.add_constant(X_all)
        endog = df_train[target]
        res_all = sm.tsa.statespace.SARIMAX(endog, exog=X_all, order=est[0], seasonal_order=est[1], freq='MS').fit()

        predict = res_all.get_prediction()
        Insample = predict.predicted_mean[start:end]
        start_f = pd.to_datetime(end) + pd.DateOffset(months=1)
        end_f = pd.to_datetime(end) + pd.DateOffset(months=period)

        x_forecast = self.df_all[start_f:end_f][var_select]
        x_forecast = sm.add_constant(x_forecast)

        OOS = res_all.get_prediction(start=start_f, end=end_f, exog=x_forecast).predicted_mean
        observed = self.df_all[target]
        
        if is_plot:
            forecast_plot(observed, Insample, OOS, target+'ARIMAX')

        ### Consolidate Results
        df_forecast = pd.DataFrame(observed)
        df_forecast['Insample'] = Insample
        df_forecast['OutofSample'] = OOS
        
        return res_all.summary(), observed, Insample, OOS, df_forecast
        # res_all: SARIMAX model
        # summary: SARIMAX model summary
        # observed: all observations
        # Insample: predictions between start & end
        # OOS: predictions between end+1_month & end+period_month
        # df_forecast: [target]all observations, [Insample]predictions between start & end, [OutOfSample]observations between end+1_month & end+period_month



def sku_revenue_prediction(brand, sku, num_var, dummy_var):

    ### select the revenue data of sku
    if brand == "Nexgard":
        df_revenue = Nexgard_invoice[[sku]] # consider the specific case
    else:
        df_revenue = Heartgard_invoice[[sku]] # consider the specific case
    print(f"Revenue data for {sku}") 
    display(df_revenue.head(1))
    df_revenue.index.freq = 'MS' # for what use?

    ### merge the revenue data with the external data
    df_all = df_revenue.merge(external_var, on=['Date'], how='outer')
    df_all = df_all.set_index('Date')
    df_all = df_all.fillna(0)
    df_all.index.freq = 'MS'
    print("Merged revenue and external data:")
    display(df_all.head(1))

    ### annual revenue plot and data (multiple years/lines)
    df_revenue_copy = df_revenue.copy()
    df_revenue_copy['year'] = df_revenue_copy.index.year
    df_revenue_copy['month'] = df_revenue_copy.index.month
    annual_stack = df_revenue_copy.pivot_table(index=['month'], columns=['year'])[sku]
    print("Annual revenue data:")
    display(annual_stack.head(1))

    if brand == "Nexgard":
        annual_stack.to_csv('C:\\Users\\zhwenxin\\OneDrive - Boehringer Ingelheim\\Desktop\\Code from Tong\\Result\\Nexgard\\'+sku+'_annual.csv')
    else:
        annual_stack.to_csv('C:\\Users\\zhwenxin\\OneDrive - Boehringer Ingelheim\\Desktop\\Code from Tong\\Result\\Heartgard\\'+sku+'_annual.csv')
   
    ### apply data to forecast 
    DF_class = Demand_Forecast(df_all, ext_var_lst) 

    # ### conduct lasso regression to select features
    # lasso_est = DF_class.lasso_selection(sku) # select features
    # # print(lasso_est.pvalues) # p_values of all coefficients (start with const)
    # var_select = list(lasso_est.pvalues.index[1:]) + [str(x) for x in range(1,13)] # features + months(1-12)
    # lasso_est = DF_class.lasso_selection(sku, var_select) 
    # var_select = list(lasso_est.pvalues.index[1:]) 
    # # print(f"pvalues list: {lasso_est.pvalues}")
    # print('final variables:', var_select)

    ### the lasso regression helps narrow down the canidate variables
    ### then we manually tried different combinations and select the best ones considering MAPE & plots
    ### also consider the business sense on the sku sales
    ### eg: exclue cat vaccination if the sku is for dog
    var_select = num_var + dummy_var
    # df_all.plot(y=num_var, figsize=(15, 10), style='--')
    # ax = df_all[sku].plot(secondary_y=True, color='k', marker='o')

    ### ARIMAX regression
    ARIMAX_train_start = '2018-06-01' 
    ARIMAX_train_end = '2020-12-01'
    ARIMAX_regression_start = '2018-06-01' 
    ARIMAX_regression_end = '2021-05-01'
    period = 19

    est = DF_class.ARIMAX_regression_training(sku, var_select, ARIMAX_train_start, ARIMAX_train_end, vanilla=[3, 0, 1], seasonal=[2, 0, 0, 12]) # parameter optimization
    summary, Insample, observed, forecast, df_result = DF_class.ARIMAX_regression(sku, var_select, ARIMAX_regression_start, ARIMAX_regression_end, est, period, True) # prediction 
    # res_all: SARIMAX model
    # summary: SARIMAX model summary
    # observed: all observations
    # Insample: predictions between start & end
    # OOS: predictions between end+1_month & end+period_month
    # df_forecast: [target]all observations, [Insample]predictions between start & end, [OutOfSample]observations between end+1_month & end+period_month

    ### 3 years forecast
    series = df_revenue.iloc[df_revenue.index<'2021-06-01'][sku] 
    target_final = series.dropna()
    stepwise_fit = pm.auto_arima(target_final, m=12, seasonal=True, trace=True, random_state=10, D=1)  # can adjust parameters here
    SARIMA_Forecast = pd.DataFrame(stepwise_fit.predict(n_periods=60))

    start_date = '6/1/2021' # Start Date of Datasets
    end_date = '12/1/2025' # Forecasts until

    Forecast_Forecast_Period = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='MS'))
    Forecast_Forecast_Period.columns = ['Date']
    Forecast_Forecast_Period = Forecast_Forecast_Period.merge(SARIMA_Forecast, left_index=True, right_index=True)
    Forecast_Forecast_Period = Forecast_Forecast_Period.set_index('Date')
    Forecast_Forecast_Period.columns = ['SARIMA']

    df_result = df_result.merge(Forecast_Forecast_Period, how='outer', left_index=True, right_index=True)

    if brand == "Nexgard":
        # df_result['brand'] = ['Nexgard']
        df_result.to_csv('C:\\Users\\zhwenxin\\OneDrive - Boehringer Ingelheim\\Desktop\\Code from Tong\\Result\\Nexgard\\' + sku + '.csv')
    else:
        # df_result['brand'] = ['Heartgard']
        df_result.to_csv('C:\\Users\\zhwenxin\\OneDrive - Boehringer Ingelheim\\Desktop\\Code from Tong\\Result\\Heartgard\\' + sku + '.csv')
      


nexgard_sku = {'142828': [['Puppies_lag3', 'FT Risk Index_pct_diff3', 'Adult Dogs_pct_diff1'], []], \
    '142829': [['Puppies_diff1', 'FT Risk Index_pct_diff3', 'Adult Dogs_pct_diff1'], ['3']], \
    '142830': [['Adult Dogs', 'CPI_US_lag3', 'humidity_lag3', 'Puppies_lag1'], []], \
    '142831': [['Puppies', 'FT Risk Index_pct_diff3', 'Adult Dogs_pct_diff1'], ['3']], \
    '142832': [['Adult Dogs', 'CPI_US_lag3', 'humidity_lag3', 'Puppies_lag1', 'Unemployment Rate_diff2'], ['3']], \
    "142833": [['Puppies_lag2', 'FT Risk Index_pct_diff3', 'Adult Dogs_pct_diff1'], ['3']], \
    '142834': [['Adult Dogs', 'CPI_US_lag3', 'humidity_lag3', 'Puppies_lag1', 'Unemployment Rate_diff2'], ['3']], \
    '142835': [['Puppies', 'FT Risk Index_pct_diff3', 'Adult Dogs_pct_diff1'], ['3']]}
for keys, values in nexgard_sku.items():
    print(keys, values)
    sku_revenue_prediction(brand="Nexgard", sku=keys, num_var=values[0], dummy_var=values[1])



heartgard_sku = {'145448': [['Adult Dogs_pct_diff1'], ['3']], \
    '145449': [['Adult Dogs_pct_diff1'], ['3']], \
    '145450': [[ 'HW Risk Index_pct_diff3', 'Adult Dogs_ma1'], ['3']], \
    '141348': [['Puppies_diff1', 'HW Risk Index_pct_diff3', 'Adult Dogs_pct_diff1'], ['3']], \
    '141354': [['Adult Dogs', 'CPI_US_lag3', 'humidity_lag3', 'Puppies_lag1'], ['3']], \
    '141360': [['Puppies_diff1', 'HW Risk Index_pct_diff3', 'Adult Dogs_pct_diff1'], ['3']], \
        }
for keys, values in heartgard_sku.items():
    print(keys, values)
    sku_revenue_prediction(brand="Heartgard", sku=keys, num_var=values[0], dummy_var=values[1])