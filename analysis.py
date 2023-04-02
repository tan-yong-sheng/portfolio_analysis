import quantstats as qs
from openbb_terminal.sdk import openbb
import pandas

excel_file_path = r"./input/portfolio_data.xlsx" # change to your input excel path
benchmark_symbol = "0820EA.KL" # change the benchmark you want, for example SPY

# Load portfolio data from ".xlsx" excel file 
p = openbb.portfolio.load(transactions_file_path = excel_file_path)
p.set_benchmark(benchmark_symbol) # set benchmark

## calculate daily return of your portfolio
daily_returns = openbb.portfolio.dret(portfolio_engine=p) 
## change the date string to datetime format
daily_returns.index = pandas.to_datetime(daily_returns.index) 
## change the column name from "benchmark" to the benchmark symbol you set
daily_returns.rename(columns={"benchmark":benchmark_symbol}, inplace=True) 

# calculate quantitative analysis of the assets
qs.reports.html(daily_returns["portfolio"], daily_returns[benchmark_symbol], output="/output/portfolio_tearsheet.html")


