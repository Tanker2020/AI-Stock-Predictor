import httpx,os,alpha
from dotenv import load_dotenv
global urls,articleTitles,apikey,relevance,summarys

load_dotenv()
apikey= os.getenv("API_KEY")

urls = []
articleTitles = []
relevance = []
summarys = []
sentiment = []

def geturls():
    symbol = alpha.getsymbol()
    multdata = []
    topics = ["blockchain","earnings","ipo","mergers_and_acquisitions","financial_markets","economy_fiscal","economy_monetary","economy_macro","energy_transportation","finance","life_sciences","manufacturing","real_estate","retail_wholesale","technology"]
    for i in topics:
        data = httpx.get(f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&topics={i}&sort=RELEVANCE&time_from=2022&apikey={apikey}").json()
        multdata.append(data)
    return multdata
def split():
    multdata = geturls()