import pandas as pd
import pytz

def data_processing(prices_data):
    prices_data = prices_data.dropna()

    # Convert the 'hour' column to Panama time if it is already timezone-aware
    #if pd.to_datetime(prices_data['hour']).dt.tz is not None:
    prices_data['hour'] = pd.to_datetime(prices_data['hour']).dt.tz_convert('America/Panama')
    #else:
        # Localize the 'hour' column to UTC if it is timezone-naive and then convert to Panama time
        #prices_data['hour'] = pd.to_datetime(prices_data['hour']).dt.tz_localize('UTC').dt.tz_convert('America/Panama')
    print(f'price latest date {prices_data["hour"].iloc[-1]}')
    # Forward fill any missing values
    prices_data = prices_data.ffill()

    # Aggregate and pivot the data
    aggregated_data = prices_data.groupby(['hour', 'symbol'], as_index=False).mean()
    pivot_prices = aggregated_data.pivot(index='hour', columns='symbol', values='price').reset_index()
    price_timeseries = pivot_prices.copy()
    price_timeseries.set_index('hour', inplace=True)
    price_timeseries.describe()

    SFRXETH_price = price_timeseries['SFRXETH']
    SFRXETH_price_pct = SFRXETH_price.pct_change()
    anomalies = SFRXETH_price_pct.where(SFRXETH_price_pct.values > SFRXETH_price_pct.mean() * 0.5).dropna()
    anomalies.describe()
    percentile_9 = anomalies.quantile(0.99)
    anomalies_above_90th = anomalies > percentile_9
    anomalies2 = anomalies_above_90th.where(anomalies_above_90th.values == True).dropna()
    anomalies_dates = anomalies2.index.unique()

    price_timeseries['SFRXETH_MA'] = price_timeseries['SFRXETH'].rolling(window=30, min_periods=1).mean()
    price_timeseries = price_timeseries.reset_index()

    # Ensure additional_dates are also timezone-aware
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
    ]).tz_localize('UTC').tz_convert('America/Panama')

    all_dates_to_replace = anomalies_dates.union(additional_dates)
    anomalies_dates_list = all_dates_to_replace.tolist()

    price_timeseries['SFRXETH_MA'] = price_timeseries['SFRXETH'].rolling(window=30, min_periods=1).mean()
    print("Before replacement:")
    print(price_timeseries[price_timeseries['hour'].isin(anomalies_dates_list)])

    # price_timeseries.loc[price_timeseries['hour'].isin(anomalies_dates_list), 'SFRXETH'] = price_timeseries.loc[price_timeseries['hour'].isin(anomalies_dates_list), 'SFRXETH_MA']
    price_timeseries.drop(columns=['level_0', 'index', 'SFRXETH_MA'], inplace=True, errors='ignore')

    print("After replacement:")
    print(price_timeseries[price_timeseries['hour'].isin(anomalies_dates_list)])

    price_timeseries.set_index('hour', inplace=True)
    price_timeseries.plot()

    SFRXETH_CLEANED = price_timeseries['SFRXETH'].to_frame('SFRXETH')
    SFRXETH_CLEANED_filtered = SFRXETH_CLEANED[SFRXETH_CLEANED.index <= '2023-07-01']

    SFRXETH_CLEANED_filtered[SFRXETH_CLEANED_filtered.values > 2500].plot()
    print(SFRXETH_CLEANED_filtered[SFRXETH_CLEANED_filtered.values > 2500])
    print(price_timeseries)

    price_timeseries.reset_index(inplace=True)
    price_timeseries.rename(columns={'hour': 'ds'}, inplace=True)
    print(price_timeseries.head())

    return price_timeseries
