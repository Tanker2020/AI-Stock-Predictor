import tensorflow as tf
import httpx,os,time
from datetime import datetime,timedelta
from dotenv import load_dotenv
global apikey,stocks
import numpy as np

load_dotenv()
apikey= os.getenv("API_KEY")

#dictionary for AI having multivariate linear regression
stocks = {
    'AAPL': {'Time': [],'Sentiment': [],'Prices': {'avgprice': [],'volume': []},'Income': [],'Cash_Flow': [],'GDP': [],'Inflation': [],'Unemployment': []},
    'AMZN': {'Time': [],'Sentiment': [],'Prices': {'avgprice': [],'volume': []},'Income': [],'Cash_Flow': [],'GDP': [],'Inflation': [],'Unemployment': []},
    'NVDA': {'Time': [],'Sentiment': [],'Prices': {'avgprice': [],'volume': []},'Income': [],'Cash_Flow': [],'GDP': [],'Inflation': [],'Unemployment': []},
    'GOOGL': {'Time': [],'Sentiment': [],'Prices': {'avgprice': [],'volume': []},'Income': [],'Cash_Flow': [],'GDP': [],'Inflation': [],'Unemployment': []},
    'MSFT': {'Time': [],'Sentiment': [],'Prices': {'avgprice': [],'volume': []},'Income': [],'Cash_Flow': [],'GDP': [],'Inflation': [],'Unemployment': []}
}


#make new array at end before converting to dataframe to turn into one dict by using stocks[currStock] and list.update('avgprice',stock[currStock]['Prices']['avgprice']) and then set time as another key with list.update('time',stock[currStock]['Prices']['time'])

#get and Compile Data 
def getPandData(currStock):
    #static data
        inflation = f'https://www.alphavantage.co/query?function=INFLATION&apikey={apikey}'
        gdp = f'https://www.alphavantage.co/query?function=REAL_GDP&interval=annual&apikey={apikey}'
        unemployment = f'https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey={apikey}'  
        
        #get data from API
        inflation = httpx.get(inflation)
        print(inflation.status_code)
        inflationData = inflation.json()
        
        gdp = httpx.get(gdp)
        print(gdp.status_code)
        gdpData = gdp.json()
        
        unemployment = httpx.get(unemployment)
        print(unemployment.status_code)
        unemploymentData = unemployment.json()
        print("Waiting 60 seconds for API to reset")
        time.sleep(60)
    
    #start compilation of data
    #for i in range(len(stocks)):
        #has a lot of variability in the data
        #currStock = list(stocks)[i]
        prices = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={currStock}&apikey={apikey}'
        cashflow = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={currStock}&apikey={apikey}'
        income = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={currStock}&apikey={apikey}'

        #get data from API
        income = httpx.get(income)
        print(income.status_code)
        incomeData = income.json()


        cashflow = httpx.get(cashflow)
        print(cashflow.status_code)
        cashflowData = cashflow.json()

        prices = httpx.get(prices)
        print(prices.status_code)
        pricesData=prices.json()

        print("Waiting 60 seconds for API to reset")
        time.sleep(60)
        #set counters for data
        cashCount = 0
        unemployCount = 0
        gdpCount = 0
        
        #split data into usable format of epoch time for AI model For prices
        lastrefPrice = datetime.strptime(pricesData['Meta Data']["3. Last Refreshed"],'%Y-%m-%d')
        clastrefPrice =lastrefPrice.date()
        numdaysPrice = timedelta(days=1)

        #collect data for AI model and compile it into usable format
        while 2009!=clastrefPrice.year:
            lastrefUnemploy = datetime.strptime(inflationData['data'][unemployCount]['date'],'%Y-%m-%d')
            clastrefUnemploy = lastrefUnemploy.date()
            
            lastrefGDP = datetime.strptime(gdpData['data'][gdpCount]['date'],'%Y-%m-%d')
            clastrefGDP = lastrefGDP.date()
            
            try:
                lastrefCashflow = datetime.strptime(cashflowData['annualReports'][cashCount]["fiscalDateEnding"],'%Y-%m-%d')
                clastrefCashflow = lastrefCashflow.date()
            except IndexError:
                pass
            
            #nested if statments and try/except to make sure data is collected and compiled correctly with correct masking values placed in case of missing data
            try:
                stockdata = pricesData['Time Series (Daily)'][str(clastrefPrice)]
                stocks[currStock]['Prices']['avgprice'].append((float(stockdata['1. open'])+float(stockdata["2. high"])+float(stockdata["3. low"])+float(stockdata["4. close"]))/4)
                stocks[currStock]['Prices']['volume'].append(stockdata['6. volume'])
                if clastrefPrice==clastrefCashflow:
                    stocks[currStock]['Time'].append(lastrefPrice.timestamp())
                    stocks[currStock]['Cash_Flow'].append(float(cashflowData['annualReports'][cashCount]['operatingCashflow']))
                    stocks[currStock]['Income'].append(float(incomeData['annualReports'][cashCount]['grossProfit']))
                    cashCount+=1
                    if clastrefPrice==clastrefGDP:
                        stocks[currStock]['GDP'].append(float(gdpData['data'][gdpCount]['value'])*1000000000)
                        stocks[currStock]['Inflation'].append(float(inflationData['data'][gdpCount]['value'])*100)
                        gdpCount+=1
                        if clastrefPrice==clastrefUnemploy:
                            stocks[currStock]['Unemployment'].append(float(unemploymentData['data'][unemployCount]['value'])*100)
                            unemployCount+=1
                        elif clastrefPrice!=clastrefUnemploy:
                            stocks[currStock]['Unemployment'].append(0)
                    elif clastrefPrice!=clastrefGDP:
                        stocks[currStock]['GDP'].append(0)
                        stocks[currStock]['Inflation'].append(0)
                        if clastrefPrice==clastrefUnemploy:
                            stocks[currStock]['Unemployment'].append(float(unemploymentData['data'][unemployCount]['value'])*100)
                            unemployCount+=1
                        elif clastrefPrice!=clastrefUnemploy:
                            stocks[currStock]['Unemployment'].append(0)
                elif clastrefPrice!=clastrefCashflow:
                    stocks[currStock]['Time'].append(lastrefPrice.timestamp())
                    stocks[currStock]['Cash_Flow'].append(0)
                    stocks[currStock]['Income'].append(0)
                    if clastrefPrice==clastrefGDP:
                        stocks[currStock]['GDP'].append(float(gdpData['data'][gdpCount]['value'])*1000000000)
                        stocks[currStock]['Inflation'].append(float(inflationData['data'][gdpCount]['value'])*100)
                        gdpCount+=1
                        if clastrefPrice==clastrefUnemploy:
                            stocks[currStock]['Unemployment'].append(float(unemploymentData['data'][unemployCount]['value'])*100)
                            unemployCount+=1
                        elif clastrefPrice!=clastrefUnemploy:
                            stocks[currStock]['Unemployment'].append(0)
                    elif clastrefPrice!=clastrefGDP:
                        stocks[currStock]['GDP'].append(0)
                        stocks[currStock]['Inflation'].append(0)
                        if clastrefPrice==clastrefUnemploy:
                            stocks[currStock]['Unemployment'].append(float(unemploymentData['data'][unemployCount]['value'])*100)
                            unemployCount+=1
                        elif clastrefPrice!=clastrefUnemploy:
                            stocks[currStock]['Unemployment'].append(0)
            except KeyError:
                stocks[currStock]['Prices']['avgprice'].append(0)
                stocks[currStock]['Prices']['volume'].append(0)
                if clastrefCashflow==clastrefPrice:
                    stocks[currStock]['Time'].append(lastrefPrice.timestamp())
                    stocks[currStock]['Cash_Flow'].append(float(cashflowData['annualReports'][cashCount]['operatingCashflow']))
                    stocks[currStock]['Income'].append(float(incomeData['annualReports'][cashCount]['grossProfit']))
                    cashCount+=1
                    if clastrefPrice==clastrefGDP:
                        stocks[currStock]['GDP'].append(float(gdpData['data'][gdpCount]['value'])*1000000000)
                        stocks[currStock]['Inflation'].append(float(inflationData['data'][gdpCount]['value'])*100)
                        gdpCount+=1
                        if clastrefPrice==clastrefUnemploy:
                            stocks[currStock]['Unemployment'].append(float(unemploymentData['data'][unemployCount]['value'])*100)
                            unemployCount+=1
                        elif clastrefPrice!=clastrefUnemploy:
                            stocks[currStock]['Unemployment'].append(0)
                    elif clastrefPrice!=clastrefGDP:
                        stocks[currStock]['GDP'].append(0)
                        stocks[currStock]['Inflation'].append(0)
                        if clastrefPrice==clastrefUnemploy:
                            stocks[currStock]['Unemployment'].append(float(unemploymentData['data'][unemployCount]['value'])*100)
                            unemployCount+=1
                        elif clastrefPrice!=clastrefUnemploy:
                            stocks[currStock]['Unemployment'].append(0)
                elif clastrefPrice!=clastrefPrice:
                    stocks[currStock]['Time'].append(lastrefPrice.timestamp())
                    stocks[currStock]['Cash_Flow'].append(0)
                    stocks[currStock]['Income'].append(0)
                    if clastrefPrice==clastrefGDP:
                        stocks[currStock]['GDP'].append(float(gdpData['data'][gdpCount]['value'])*1000000000)
                        stocks[currStock]['Inflation'].append(float(inflationData['data'][gdpCount]['value'])*100)
                        gdpCount+=1
                        if clastrefPrice==clastrefUnemploy:
                            stocks[currStock]['Unemployment'].append(float(unemploymentData['data'][unemployCount]['value'])*100)
                            unemployCount+=1
                        elif clastrefPrice!=clastrefUnemploy:
                            stocks[currStock]['Unemployment'].append(0)
                    elif clastrefPrice!=clastrefGDP:
                        stocks[currStock]['GDP'].append(0)
                        stocks[currStock]['Inflation'].append(0)   
                        if clastrefPrice==clastrefUnemploy:
                            stocks[currStock]['Unemployment'].append(float(unemploymentData['data'][unemployCount]['value'])*100)
                            unemployCount+=1
                        elif clastrefPrice!=clastrefUnemploy:
                            stocks[currStock]['Unemployment'].append(0)     
            #reduce time by one day every iteration to get the previous day's data
            lastrefPrice = lastrefPrice-numdaysPrice
            clastrefPrice = clastrefPrice-numdaysPrice
        
        #reverse all of the lists due to them needing to be ordered by time
        stocks[currStock]['Prices']['avgprice'].reverse()
        stocks[currStock]['Prices']['volume'].reverse()
        stocks[currStock]['Income'].reverse()
        stocks[currStock]['Cash_Flow'].reverse()
        stocks[currStock]['GDP'].reverse()
        stocks[currStock]['Inflation'].reverse()
        stocks[currStock]['Unemployment'].reverse()
        
        #call the function to get the sentiment data and add it to the dict
        getIncomeSentimentVal(currStock)
        
        gdpCount=0
        unemployCount=0
        cashCount=0
        
        fake="""
        df = pd.DataFrame(stocks)
        df["1. open"]=df["1. open"].astype(float)
        df["2. high"]=df["2. high"].astype(float)
        df["3. low"]=df["3. low"].astype(float)
        df["4. close"]=df["4. close"].astype(float)
        df["5. volume"]=df["5. volume"].astype(float)
        return df"""

#get the average of the bullish vs barrish vs. nuetral score for each day and then add that to the dict
def getIncomeSentimentVal(currStock):
    Techurl = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={currStock}&topics=technology&limit=200&sort=relevance&apikey={apikey}"
    IPOurl = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={currStock}&topics=IPO&limit=200&sort=relevance&apikey={apikey}"
    Mergerurl = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={currStock}&topics=mergers_and_acquisitions&limit=200&sort=relevance&apikey={apikey}"
    Marketurl = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={currStock}&topics=financial_markets&limit=200&sort=relevance&apikey={apikey}"
    Fiscalurl = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={currStock}&topics=economy_fiscal&limit=200&sort=relevance&apikey={apikey}"
    Earningurl = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={currStock}&topics=earnings&limit=200&sort=relevance&apikey={apikey}"
    Monetaryurl = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={currStock}&topics=economy_monetary&limit=200&sort=relevance&apikey={apikey}"

    data = [httpx.get(Techurl).json(),httpx.get(IPOurl).json(),httpx.get(Mergerurl).json(),httpx.get(Marketurl).json(),httpx.get(Fiscalurl).json()]
    print("starting sentiment data collection")
    time.sleep(60)
    data.append(httpx.get(Earningurl).json())
    data.append(httpx.get(Monetaryurl).json())

    # turn all data into masked data beforehand
    for i in range(len(stocks[currStock]['Time'])):
        stocks[currStock]['Sentiment'].append(0)
        
    # change masked data to actual data if it exists
    for i in range(len(data)):
        try:
            for c in range(len(data[i]['feed'])):
                timeData = data[i]['feed'][c]['time_published']
                timeData = timeData.split('T')[0]
                timeData = timeData[:4]+"-"+timeData[4:6]+"-"+timeData[6:]
                timeData = datetime.strptime(timeData,'%Y-%m-%d')
                if (timeData.timestamp() not in stocks[currStock]['Time']):
                    stocks[currStock]['Time'].append(timeData.timestamp())
                    stocks[currStock]['Time'].sort()
                    index = stocks[currStock]['Time'].index(timeData.timestamp())
                    stocks[currStock]['Prices']['avgprice'].insert(index,0)
                    stocks[currStock]['Prices']['volume'].insert(index,0)
                    stocks[currStock]['Income'].insert(index,0)
                    stocks[currStock]['Cash_Flow'].insert(index,0)
                    stocks[currStock]['GDP'].insert(index,0)
                    stocks[currStock]['Inflation'].insert(index,0)
                    stocks [currStock]['Unemployment'].insert(index,0)
                    stocks[currStock]['Sentiment'].insert(index,data[i]['feed'][c]['overall_sentiment_score'])
                else:
                    index = stocks[currStock]['Time'].index(timeData.timestamp())
                    stocks[currStock]['Sentiment'][index]=data[i]['feed'][c]['overall_sentiment_score']
        except IndexError:
            pass    


#due to timings of datasets not matching up there will need to be masking 
#Actual AI code for multivariate linear regression
def AI(currStock):
    #df = getPandData()
    #print(tf.convert_to_tensor(df))
    getPandData(currStock)
    xs = stocks[currStock]['Time']   
    ys = np.stack((
        np.array(stocks[currStock]['Prices']['avgprice']),
        np.array(stocks[currStock]['Prices']['volume']),
        np.array(stocks[currStock]['Income']),
        np.array(stocks[currStock]['Cash_Flow']),
        np.array(stocks[currStock]['GDP']),
        np.array(stocks[currStock]['Inflation']),
        np.array(stocks[currStock]['Unemployment'])
    ),1)
    
     
     
    model = tf.keras.Sequential([
        tf.keras.layers.Normalization(axis=-1),
        tf.keras.layers.Dense(units=1, input_shape=[8])
    ])
    model.add(tf.keras.layers.Masking(mask_value=0))
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs,ys,epochs=500)
    y = model.predict(1671513866)
    print(model)
    print(y)
    
def StartAIPrediction():
    #AI("AAPL")
    getPandData("AAPL")
    print(stocks['AAPL'])
    print(len(stocks['AAPL']['Time']))
    print (len(stocks['AAPL']['Prices']['avgprice']))
    print (len(stocks['AAPL']['Prices']['volume']))
    print (len(stocks['AAPL']['Income']))
    print (len(stocks['AAPL']['Cash_Flow']))
    print (len(stocks['AAPL']['GDP']))
    print (len(stocks['AAPL']['Inflation']))
    print (len(stocks['AAPL']['Unemployment']))
    print (len(stocks['AAPL']['Sentiment']))
    
StartAIPrediction()