import pandas as pd
import tpqoa

api = tpqoa.tpqoa("oanda.cfg")

print(api.get_account_summary())
print(api.account_type)
print(api.account_id)

print(api.get_instruments())

instr = api.get_instruments()
print(len(instr))

# historical data
df = api.get_history(instrument = "EUR_USD", start = "2020-07-01", end = "2020-07-31",
                granularity = "D", price = "B")
df.info

#streaming data
streamed_data = api.stream_data('EUR_USD', stop=10)
print(streamed_data)

# create orders
myorder = api.create_order(instrument = "EUR_USD", units = 1000, sl_distance= 0.1) #open the position
print(myorder)

# wait to see the change
input("press any key...")


myorder = api.create_order(instrument = "EUR_USD", units = -1000, sl_distance= 0.1) # close the position
print(myorder)

# get details about a precise transaction
print(api.get_transactions(tid = 4))

# all transactions
print(api.print_transactions())#tid = 5-1)
