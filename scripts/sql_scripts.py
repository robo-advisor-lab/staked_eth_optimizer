from datetime import timedelta

def sql(today, days=23):
    beginning = today 
    print('beginning', beginning)
    lst_prices_query = f"""
    WITH lsts AS (
        SELECT column1 AS token_address FROM VALUES
        ('0xae78736cd615f374d3085123a210448e74fc6393'), -- rETH
        ('0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0'), -- sfrxETH
        ('0xac3e018457b222d93114458476f3e3416abbe38f'), -- wstETH
        ('0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2')  -- WETH
    )
    SELECT 
        hour,
        CASE 
            WHEN symbol = 'WETH' THEN 'ETH' 
            ELSE symbol 
        END AS symbol,
        price
    FROM ethereum.price.ez_prices_hourly
    WHERE token_address IN (SELECT token_address FROM lsts)
    AND date_trunc('day', hour) >= date('{beginning}')
    ORDER BY hour ASC, symbol
    """
    return lst_prices_query

def eth_price(today, days=23):
    beginning = today - timedelta(days)
    eth_prices_query = f"""
    select 
        hour, 
        CASE 
            WHEN symbol = 'WETH' THEN 'ETH' 
            ELSE symbol 
        END AS symbol,
        price
    from ethereum.price_ez_prices_hourly
    where symbol == 'WETH'
    AND date_trunc('day', hour) >= date('{beginning}')
    order by hour asc, symbol


"""
    return eth_prices_query

    