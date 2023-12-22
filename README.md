# Time_Series_Analyer
Using Time-Series models to evaluate user-data, and to identify consumer trends and price trends in the e-commerce space.


##  Background 

You’re a growth analyst at MercadoLibreLinks to an external site.. With over 200 million users, MercadoLibre is the most popular e-commerce site in Latin America. You've been tasked with analyzing the company's financial and user data in clever ways to make the company grow. So, you want to find out if the ability to predict search traffic can translate into the ability to successfully trade the stock.



## Overview and Purpose:

In a bid to drive revenue, you’ll produce a Jupyter notebook that contains your data preparation, analysis, and visualizations for all the time series data that the company needs to understand. You’ll use text and comments to document your findings, and you’ll answer the question prompts in the instructions. Specifically, this file should contain the following:

Visual depictions of seasonality (as measured by Google Search traffic) that are of interest to the company.

An evaluation of how the company’s stock price correlates to its Google Search traffic.

A Prophet forecast model that can predict hourly user search traffic.

Answers to questions in the instructions that you write in your Jupyter Notebook.

(Optional) A plot of a forecast for the company’s future revenue.

Push your final notebook to your GitHub repository so that others can review your work.

--
## Step 1: Find Unusual Patterns in Hourly Google Search Traffic

The data science manager asks if the Google search traffic for the company links to any financial events at the company. Or, does the search traffic data just present random noise? To answer this question, pick out any unusual patterns in the Google search data for the company, and connect them to the corporate financial events.

To do so, complete the following steps:

1. Read the search data into a DataFrame, and then slice the data to just the month of May 2020. (During this month, MercadoLibre released its quarterly financial results.) Use hvPlot to visualize the results. Do any unusual patterns exist?

2. Calculate the total search traffic for the month, and then compare the value to the monthly median across all months. Did the Google search traffic increase during the month that MercadoLibre released its financial results?

```
# Upload the "google_hourly_search_trends.csv" file into Colab, then store in a Pandas DataFrame
# Set the "Date" column as the Datetime Index.

Path = "Resources/google_hourly_search_trends.csv"

# Store data in dataframe
df_mercado_trends = pd.read_csv(Path, index_col="Date", parse_dates=True)

# Slice the DataFrame to just the month of May 2020
df_may_2020 = df_mercado_trends.loc["2020-05"]

# Use hvPlot to visualize the data for May 2020
df_may_2020.hvplot(title="Search trends MercadoLibre May 2020")

# Calculate the sum of the total search traffic for May 2020
traffic_may_2020 = df_may_2020.loc["2020-05"].sum()

# Calcluate the monhtly median search traffic across all months 
# Group the DataFrame by index year and then index month, chain the sum and then the median functions
median_monthly_traffic = df_mercado_trends["Search Trends"].groupby(by=[df_mercado_trends.index.year, df_mercado_trends.index.month]).median()
```


```
# Review the data types of the DataFrame using the info function
median_monthly_traffic.info()
```


<class 'pandas.core.series.Series'>
MultiIndex: 52 entries, (2016, 6) to (2020, 9)
Series name: Search Trends
Non-Null Count  Dtype  
--------------  -----  
52 non-null     float64
dtypes: float64(1)
memory usage: 762.0 bytes

```
# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Slice the DataFrame to just the month of May 2020
df_may_2020 = df_mercado_trends.loc["2020-05"]

# Use hvPlot to visualize the data for May 2020
df_may_2020.hvplot(title="Google search trends MercadoLibre - May 2020")
```

![an image showing the Google search trends for May of 2020](./images/may_trends.png)
**Question:** 
Did the Google search traffic increase during the month that MercadoLibre released its financial results?

**Answer:** 
Yes, the Google Search traffic had increased by around 8.5%, meaning that there positive movement in the MercadoLibre search traffic. The financial statement would have likely brought the company to the attention of many investors, who would have ben keeping an eye on the financial statements of many of technology firms. Though MercadoLibre saw a positive change in the month of change, we cannot yet assume that the financial statement was the cause of this deviaion, as correlation does not neccesarily equate to causation. We would need more information to asses whether or not the financial statement release was the cause of the rising search traffic- most likely, we would need information on the users who had searched for MercadoLibre, and see if those users are investors or financial analysts. Regular users generally do not pay cose attention to the financial statements of firms, so if we could isolate the group that had driven the percentage increase, we would be able to identify whether the correlation equates to a causation.

---

## Step 2: Mine the Search Traffic Data for Seasonality

Marketing realizes that they can use the hourly search data, too. If they can track and predict interest in the company and its platform for any time of day, they can focus their marketing efforts around the times that have the most traffic. This will get a greater return on investment (ROI) from their marketing budget.

To that end, you want to mine the search traffic data for predictable seasonal patterns of interest in the company. To do so, complete the following steps:

1. Group the hourly search data to plot the average traffic by the day of the week (for example, Monday vs. Friday).

2. Using hvPlot, visualize this traffic as a heatmap, referencing the `index.hour` as the x-axis and the `index.dayofweek` as the y-axis. Does any day-of-week effect that you observe concentrate in just a few hours of that day?

3. Group the search data by the week of the year. Does the search traffic tend to increase during the winter holiday period (weeks 40 through 52)?

#### Step 1: Group the hourly search data to plot the average traffic by the day of the week (for example, Monday vs. Friday).
```
# Holoviews extension to render hvPlots in Colab
# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Group the hourly search data to plot (use hvPlot) the average traffic by the day of week 
group_level_weeks = df_mercado_trends.index.dayofweek
df_mercado_trends.groupby(group_level_weeks).mean().hvplot(title = "Mercadolibre Google search traffic grouped by day of week")
```
#### Step 2: Using hvPlot, visualize this traffic as a heatmap, referencing the `index.hour` as the x-axis and the `index.dayofweek` as the y-axis. Does any day-of-week effect that you observe concentrate in just a few hours of that day?

```
import numpy as np
# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Use hvPlot to visualize the hour of the day and day of week search traffic as a heatmap.
weekly_trends = df_mercado_trends.index.dayofweek
df_mercado_trends.groupby(weekly_trends).mean().hvplot()

df_mercado_trends.hvplot.heatmap(
    x='index.hour',
    y='index.dayofweek',
    C='Search Trends',
    cmap='reds',
    title= "MercadoLibre average search traffic during Hour of the day grouped by Days of the week"
).aggregate(function=np.mean)

```
**Question:** Does any day-of-week effect that you observe concentrate in just a few hours of that day?

**Answer:** The heatmap shows that the search traffic is highest on Friday, and lowest on Sunday. The heatmap also shows that the search traffic is highest between 12:00 and 15:00 on Friday, and lowest between 0:00 and 3:00 on Sunday. This means that the search traffic is highest on Friday afternoon, and lowest on Sunday morning. This makes sense, as most people are at work on Friday afternoon, and are likely to be searching for MercadoLibre during their lunch break. On Sunday morning, most people are likely to be sleeping in, and are not likely to be searching for MercadoLibre.

#### Step 3: Group the search data by the week of the year. Does the search traffic tend to increase during the winter holiday period (weeks 40 through 52)?

```
# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Group the hourly search data to plot (use hvPlot) the average traffic by the week of the year
group_level_week_num = df_mercado_trends.index.isocalendar().week
df_mercado_trends.groupby(group_level_week_num).mean().hvplot(title="MercadoLibre Google Search Trends by Week of Year")
```
**Question:** Does the search traffic tend to increase during the winter holiday period (weeks 40 through 52)?

**Answer:** Yes, the search traffic tends to increase during the winter holiday period.

---

## Step 3: Relate the Search Traffic to Stock Price Patterns

You mention your work on the search traffic data during a meeting with people in the finance group at the company. They want to know if any relationship between the search data and the company stock price exists, and they ask if you can investigate.

To do so, complete the following steps:

1. Read in and plot the stock price data. Concatenate the stock price data to the search data in a single DataFrame.

2. Market events emerged during the year of 2020 that many companies found difficult. But, after the initial shock to global financial markets, new customers and revenue increased for e-commerce platforms. Slice the data to just the first half of 2020 (`2020-01` to `2020-06` in the DataFrame), and then use hvPlot to plot the data. Do both time series indicate a common trend that’s consistent with this narrative?

3. Create a new column in the DataFrame named “Lagged Search Trends” that offsets, or shifts, the search traffic by one hour. Create two additional columns:

    * “Stock Volatility”, which holds an exponentially weighted four-hour rolling average of the company’s stock volatility

    * “Hourly Stock Return”, which holds the percent change of the company's stock price on an hourly basis

4. Review the time series correlation, and then answer the following question: Does a predictable relationship exist between the lagged search traffic and the stock volatility or between the lagged search traffic and the stock price returns?

#### Step 1: Read in and plot the stock price data. Concatenate the stock price data to the search data in a single DataFrame.

```
# # Upload the "mercado_stock_price.csv" file into Colab, then store in a Pandas DataFrame
# # Set the "date" column as the Datetime Index.
# from google.colab import files
# uploaded = files.upload()

Path = "Resources/mercado_stock_price.csv"
df_mercado_stock = pd.read_csv(Path, index_col="date", parse_dates=True)

# Review
df_mercado_stock.info()
```
```
# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Use hvPlot to visualize the closing price of the df_mercado_stock DataFrame
df_mercado_stock.hvplot(title="MercadoLibre Closing Price")
```

#### Step 2: Market events emerged during the year of 2020 that many companies found difficult. But, after the initial shock to global financial markets, new customers and revenue increased for e-commerce platforms. Slice the data to just the first half of 2020 (`2020-01` to `2020-06` in the DataFrame), and then use hvPlot to plot the data. Do both time series indicate a common trend that’s consistent with this narrative?

```
# For the combined dataframe, slice to just the first half of 2020 (2020-01 through 2020-06) 
first_half_2020 = mercado_stock_trends_df.loc["2020-01":"2020-06"]

# Use hvPlot to visualize the close and Search Trends data
# Plot each column on a separate axes
first_half_2020.hvplot(shared_axes=False, subplots=True).cols(1)
```

---

```
# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Use hvPlot to visualize the stock volatility
mercado_stock_trends_df['Stock Volatility'].hvplot(title="Stock Volatility")
```




---

## Technologies

### Instructions
First, configure a Google Colab workspace as follows:

Open Google ColabLinks to an external site. and upload your starter notebook.

Run the provided code in the Install and import the required libraries and dependencies section.

The first cell will install the necessary libraries into the Google Colab runtime.
The second cell will import the dependencies for use in the notebook.
With your workspace configured, you can begin the Challenge. The instructions are divided into four steps and an optional fifth step, as follows:

Step 1: Find unusual patterns in hourly Google search traffic

Step 2: Mine the search traffic data for seasonality

Step 3: Relate the search traffic to stock price patterns

Step 4: Create a time series model with Prophet

Step 5 (optional): Forecast revenue by using time series models


### Required Packages :



The following python modules are also used in the application. Remember to install these packages via Terminal for MacOS/Linux or GitBash for windows clients. 



*[tokenize]
(https://docs.python.org/3/library/tokenize.html)
- The tokenize module provides a lexical scanner for Python source code, implemented in Python. The scanner in this module returns comments as tokens as well, making it useful for implementing “pretty-printers”, including colorizers for on-screen displays.

*[datetime]
(https://docs.python.org/3/library/datetime.html)
- The datetime module supplies classes for manipulating dates and times.

While date and time arithmetic is supported, the focus of the implementation is on efficient attribute extraction for output formatting and manipulation.



*[fbprophet]
(https://pypi.org/project/fbprophet/)
-Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.


* [pandas](https://github.com/pandas-dev/pandas) 
- pandas is used to interact with data packages, plot data frames, create new dataframes, describe abailable data, and helps traders and fintech proffesionals organize financial data to perform advanced decisionmaking. 

* [pathlib](https://github.com/python/cpython/blob/main/Lib/pathlib.py) - Allows the user to specify the path to a data frame / any data in a csv file. 

* [hvplot.pandas](https://hvplot.holoviz.org/user_guide/Pandas_API.html)
- If hvplot and pandas are both installed, then we can use the pandas.options.plotting.backend to control the output of pd.DataFrame.plot and pd.Series.plot. This notebook is meant to recreate the pandas visualization docs.

* [numpy]
(https://github.com/numpy/numpy)
- NumPy is the fundamental package for scientific computing with Python. It provides: a powerful N-dimensional array object, sophisticated (broadcasting) functions tools for integrating C/C++ and Fortran code, useful linear algebra, Fourier transform, and random number capabilities. 

* [os]
(https://docs.python.org/3/library/os.html)
- This module provides a portable way of using operating system dependent functionality. If you just want to read or write a file see open(), if you want to manipulate paths, see the os.path module, and if you want to read all the lines in all the files on the command line see the fileinput module. For creating temporary files and directories see the tempfile module, and for high-level file and directory handling see the shutil module..

* [io]
(https://docs.python.org/3/library/io.html)
- The io module provides Python’s main facilities for dealing with various types of I/O. There are three main types of I/O: text I/O, binary I/O and raw I/O. These are generic categories, and various backing stores can be used for each of them. A concrete object belonging to any of these categories is called a file object. Other common terms are stream and file-like object.

* [sys]
(https://docs.python.org/3/library/sys.html)
- This module provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter. It is always available.

* [setuptools]
(https://pypi.org/project/setuptools/)
- setuptools is a pypi component included in the Tidelift Subscription
Tidelift is working with the maintainers of setuptools and a growing network of open source maintainers to ensure your open source software supply chain meets enterprise standards now and into the future.



## Installation Guide

There are four installations neccesary for the program to run. Install all modules and libraries neccesary by first activating your conda dev environment via terminal (for MacOS) or GitBash (Windows/Linux) command : ''' conda activate dev'''. 

After activating the dev environment, install the following libraries via. the command line :

'''python
    pip3 install pandas
    pip3 install numpy
    pip3 install pathlib
    conda install -c pyviz hvplot geoviews
    conda install -c conda-forge voila
    pip install SQLAlchemy
    pip install json
    conda install os
    conda install requests
    pip install alpaca_trade_api
'''

Then, in google colab, import the following modules in the program :

'''
!pip install pystan~=2.14
!pip install fbprophet
!pip install hvplot
!pip install holoviews
!pip install cmdstanpy>=1.0.4
'''
## Usage

In this application, you’ll create charts with hvPlot and use the Facebook Prophet libraryLinks to an external site. to analyze time series data.

The Facebook Prophet library can be difficult to install on some machines, so in this module you'll also get acquainted with Google Colab - an IDE that allows you to run Jupyter Notebooks in the cloud. Before beginning the module, take a moment to watch the following demonstration video showing how to open a notebook in Google Colab, install the necessary libraries and import data.

(https://www.youtube.com/watch?v=nHaDm_CFCwA)



NOTE : remember to be in the application folder (clone) when inputing the command.
---


## Contributors

The sole contributor for this project is:

**NAJIB ABOU NASR**
 no instagram or linkedin yet!
---

## License

Using the 'MIT' license!
--- 

