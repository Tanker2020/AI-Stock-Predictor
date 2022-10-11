import tensorflow as tf
import httpx,os
import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from dotenv import load_dotenv
global apikey

load_dotenv()
apikey= os.getenv("API_KEY")

#get user wanted Stock
def getsymbol():
    searchinput = input("Enter stock name: ")

    search = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={searchinput}&apikey={apikey}'
    r = httpx.get(search)
    searchdata = r.json()
    bestmatches = searchdata["bestMatches"]
    for i in range(len(bestmatches)):
        if bestmatches[i]["1. symbol"]==searchinput:
            symbol = bestmatches[i]["1. symbol"]
            print(f"Results found for {searchinput}")
            break
        else:
            print("Did you mean: ",bestmatches[i]["1. symbol"])
    try:
        return symbol
    except Exception:
        SystemExit
#get API Key 
def getPandDataStocks():
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={getsymbol()}&apikey={apikey}'

        data = httpx.get(url)
        print(data.status_code)
        data=data.json()
        #lastref = datetime.strptime(data['Meta Data']["3. Last Refreshed"],'%Y-%m-%d').date()
        lastref = datetime.strptime("2022-10-07",'%Y-%m-%d')
        clastref =lastref.date()
        numdays = timedelta(days=7)
        pandData = []

        while 2015!=clastref.year:
            try:
                stockdata = data['Weekly Time Series'][str(clastref)]
                stockdata.update({'6. Time':lastref.timestamp()})
                if stockdata!="":
                    pandData.append(stockdata)
                lastref = lastref-numdays
                clastref = clastref-numdays
            except Exception:
                lastref = lastref-numdays
                clastref = clastref-numdays
        df = pd.DataFrame(pandData)
        df["1. open"]=df["1. open"].astype(float)
        df["2. high"]=df["2. high"].astype(float)
        df["3. low"]=df["3. low"].astype(float)
        df["4. close"]=df["4. close"].astype(float)
        df["5. volume"]=df["5. volume"].astype(float)
        return df


#Actual Nueral Network and Machine Code
def AI():
    df = getPandDataStocks()
    train_df = df.sample(frac=0.8, random_state=0)
    test_df = df.drop(train_df.index)
    
    target = df.pop('1. open')
    print(tf.convert_to_tensor(df))
    #model = tf.keras.Sequential([
    #    tf.keras.layers.Flatten(input_shape="dfsf")
    #])
    
AI()