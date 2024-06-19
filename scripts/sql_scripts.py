from datetime import timedelta

def sql(today, days=23):
    
    beginning = today - timedelta(days)
    print('beginning', beginning)
    lst_prices_query = f"""

    with lsts as (
        select column1 as token_address from values
        ('0xae78736cd615f374d3085123a210448e74fc6393'),
        ('0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0'),
        ('0xac3e018457b222d93114458476f3e3416abbe38f')
    )

    select hour,
        symbol,
        price
    from ethereum.price.ez_prices_hourly
    where token_address in (select token_address from lsts)
    AND date_trunc('day', hour) >= date('{beginning}')
    order by hour asc, symbol

    """
    return lst_prices_query


    