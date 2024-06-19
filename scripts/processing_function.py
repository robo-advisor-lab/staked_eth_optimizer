import pandas as pd 

def data_processing(prices_data):
    prices_data = prices_data.dropna()



    prices_data['hour'] = pd.to_datetime(prices_data['hour']).dt.tz_localize(None)
    prices_data = prices_data.ffill()


    aggregated_data = prices_data.groupby(['hour', 'symbol'], as_index=False).mean()

    pivot_prices = aggregated_data.pivot(index='hour', columns='symbol', values='price').reset_index()

    price_timeseries = pivot_prices.copy()

    price_timeseries.set_index('hour', inplace=True)

    price_timeseries.describe()

    SFRXETH_price = price_timeseries['SFRXETH']

    #SFRXETH_price.to_csv('../data/sfrx.csv')

    SFRXETH_price_pct = SFRXETH_price.pct_change()

    anomolies = SFRXETH_price_pct.where(SFRXETH_price_pct.values > SFRXETH_price_pct.mean()*0.5).dropna()

    anomolies.describe()

    percentile_9 = anomolies.quantile(0.99)
    anomalies_above_90th = anomolies > percentile_9

    anomalies2 = anomalies_above_90th.where(anomalies_above_90th.values == True).dropna()

    anomalies_dates = anomalies2.index.unique()

    anomalies_dates_list = anomalies_dates.tolist()

    weird_dates = price_timeseries.loc[anomalies_dates]

    #weird_dates['SFRXETH']

    #weird_dates['SFRXETH'].plot()

    price_timeseries['SFRXETH_MA'] = price_timeseries['SFRXETH'].rolling(window=30, min_periods=1).mean()

    price_timeseries = price_timeseries.reset_index()

    # price_timeseries.loc[price_timeseries['HOUR'].isin(anomalies_dates_list), 'SFRXETH'] = price_timeseries.loc[price_timeseries['HOUR'].isin(anomalies_dates_list), 'SFRXETH_MA']

    # price_timeseries['HOUR']

    # print(price_timeseries[price_timeseries['HOUR'].isin(weird_dates)])

    additional_dates = pd.to_datetime([
        '2023-04-20 07:00:00', '2023-04-20 03:00:00', '2023-04-20 05:00:00', 
        '2023-04-18 12:00:00', '2023-04-18 13:00:00', '2023-04-18 08:00:00',
        '2023-04-18 09:00:00', '2023-04-19 00:00:00', '2023-04-19 01:00:00',
        '2023-04-19 02:00:00','2023-04-19 03:00:00','2023-04-19 04:00:00',
        '2023-04-19 05:00:00', '2023-04-20 14:00:00','2023-04-18 01:00:00',
        '2023-04-19 01:00:00','2023-04-19 02:00:00','2023-04-19 03:00:00',
        '2023-04-19 04:00:00', '2023-04-19 05:00:00','2023-04-20 07:00:00',
        '2023-04-20 14:00:00','2023-04-17 22:00:00','2023-04-17 23:00:00',
        '2023-04-18 00:00:00','2023-04-18 13:00:00','2023-04-18 14:00:00',
        '2023-04-18 15:00:00','2023-04-18 23:00:00','2023-04-19 00:00:00',
        '2023-04-19 01:00:00','2023-04-19 02:00:00','2023-04-19 03:00:00',
        '2023-04-19 04:00:00','2023-04-19 05:00:00','2023-04-19 12:00:00',
        '2023-04-19 14:00:00','2023-04-20 04:00:00','2023-04-20 05:00:00',
        '2023-04-20 06:00:00','2023-04-20 07:00:00','2023-04-20 08:00:00',
        '2023-04-20 09:00:00','2023-04-20 10:00:00','2023-04-20 11:00:00',
        '2023-04-20 12:00:00','2023-04-20 13:00:00','2023-04-20 14:00:00',
        '2023-04-20 15:00:00','2023-04-20 17:00:00'
    ])

    all_dates_to_replace = anomalies_dates.union(additional_dates)

    anomalies_dates_list = all_dates_to_replace.tolist()

    # Calculate the 30-day moving average for the SFRXETH column
    price_timeseries['SFRXETH_MA'] = price_timeseries['SFRXETH'].rolling(window=30, min_periods=1).mean()

    # Debug: Print the DataFrame before replacement
    print("Before replacement:")
    print(price_timeseries[price_timeseries['hour'].isin(anomalies_dates_list)])

    # Replace the values at the anomalies dates with the 30-day moving average values
    price_timeseries.loc[price_timeseries['hour'].isin(anomalies_dates_list), 'SFRXETH'] = price_timeseries.loc[price_timeseries['hour'].isin(anomalies_dates_list), 'SFRXETH_MA']

    # Drop the unnecessary columns if not needed
    price_timeseries.drop(columns=['level_0', 'index', 'SFRXETH_MA'], inplace=True, errors='ignore')

    # Debug: Print the DataFrame after replacement
    print("After replacement:")
    print(price_timeseries[price_timeseries['hour'].isin(anomalies_dates_list)])


    # In[ ]:





    # In[33]:


    #price_timeseries.set_index('HOUR').to_csv('../data/sfrx.csv')


    # In[34]:


    price_timeseries.set_index('hour', inplace=True)


    # In[35]:


    price_timeseries.plot()


    # In[36]:


    SFRXETH_CLEANED = price_timeseries['SFRXETH'].to_frame('SFRXETH')
    SFRXETH_CLEANED_filtered = SFRXETH_CLEANED[SFRXETH_CLEANED.index <= '2023-07-01']


    # In[37]:


    #SFRXETH_CLEANED_filtered[SFRXETH_CLEANED_filtered.values > 2500].to_csv('../data/sfrx_4000.csv')


    # In[38]:


    SFRXETH_CLEANED_filtered[SFRXETH_CLEANED_filtered.values > 2500].plot()


    # In[39]:


    SFRXETH_CLEANED_filtered[SFRXETH_CLEANED_filtered.values > 2500]


    # In[40]:


    price_timeseries


    # ## Forecasting
    # 
    # For sake of time, will likely not include features other than prices and moving day averages

    # In[41]:


    price_timeseries.reset_index(inplace=True)  # Reset index to use 'HOUR' as a column
    price_timeseries.rename(columns={'hour': 'ds'}, inplace=True)  # Rename 'HOUR' to 'ds'

    # Display the transformed dataframe
    print(price_timeseries.head())
    return price_timeseries

