from fastapi import FastAPI, Request
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import pickle
import uvicorn
import requests
import json
from fastapi import FastAPI
  # type: ignore # Example indicator (replace with your choice)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=".") # Assuming your HTML file is in a 'templates' folder



# Load model and scaler from pickle file
def load_model():
    with open("stock_prediction_model_1.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]

model, scaler = load_model()

#Login
@app.get("/", response_class=HTMLResponse)
async def log_in(request: Request):
    return templates.TemplateResponse("login_page.html",{"request": request})

# Function to predict next day's opening and closing prices
@app.get("/predict_next_day")
async def predict_next_day():
  try:
        symbol = "GC=F"
        today = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')
        
        new_data = yf.download(symbol, start=start_date, end=today)
        latest_closing_price = new_data["Close"].iloc[-1]

        time_step = 60 
        data = new_data["Close"][-time_step:]
        scaled_data = scaler.transform(data.values.reshape(-1, 1))
        x_test = scaled_data.reshape(1, -1, 1)

        prediction = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(prediction)
        predicted_opening_price, predicted_closing_price = predicted_prices.reshape(-1)[0], predicted_prices.reshape(-1)[1]

        return {
            "symbol": symbol,
            "latest_closing_price": latest_closing_price.item(),
            "predicted_opening_price": predicted_opening_price.item(),
            "predicted_closing_price": predicted_closing_price.item(),
            "dates": new_data.index.strftime('%Y-%m-%d').tolist(),
            "opens": new_data["Open"].tolist(),
            "highs": new_data["High"].tolist(),
            "lows": new_data["Low"].tolist(),
            "closes": new_data["Close"].tolist()
        }
  except Exception as e:
        return {"error": str(e)}

@app.get("/chart", response_class=HTMLResponse)
async def read_item(request: Request):
    prediction_data = predict_next_day()
    return templates.TemplateResponse("indexx.html", {"request": request, "data": prediction_data})
def gold_news():
  url = ('https://newsapi.org/v2/everything?'
        'q=Gold stock price&'
        'from=2024-04-25&'
        'sortBy=popularity&'
        'apiKey=f8ef99980cd740e5a9ef2896a039f74a')

  response = requests.get(url)
  # Check if request was successful
  if response.status_code == 200:
    # Extract only 'author' and 'description' fields with data cleaning
    filtered_data = [
      {'author': article.get('author', 'Unknown').strip(),
       'description': (article.get('description', 'No description available')
                        .strip()
                        .replace('\n', ' '))  # Remove newlines and extra spaces
      }
      for article in response.json()['articles']
      if article.get('description') and 'Gold' in article.get('description')
    ]

    # Write filtered data to a JSON file
    with open('filtered_data.json', 'w') as f:
      json.dump(filtered_data, f, indent=4)

    print("Filtered data saved successfully.")
    # Read data from the JSON file
  else:
    print("Error occurred while fetching data:", response.status_code)

  ##############################################################################################

  with open('filtered_data.json', 'r') as f:
    filtered_data = json.load(f)

  # Iterate through each record and format
  gold_lst = []
  for record in filtered_data:
    author = record['author']
    description = record['description']
    # Combine author and description with a colon (:) and newline
    formatted_record = f"{author}:{description}\n"
    gold_lst.append(formatted_record)
    # Join formatted records with a special delimiter
    delimited_data = '\n---\n'.join(gold_lst)
    

  return delimited_data
# def gold_news():
#     url = ('https://newsapi.org/v2/everything?'
#        'q=Gold stock price&'
#        'from=2024-04-25&'
#        'sortBy=popularity&'
#        'apiKey=f8ef99980cd740e5a9ef2896a039f74a')

#     response = requests.get(url)
#     #Check if request was successful
#     if response.status_code == 200:
#         # Extract only 'author' and 'description' fields from each article
#         filtered_data = [{'author': article.get('author', 'Unknown'), 'description': article.get('description', 'No description available')} for article in response.json()['articles'] if article.get('description') and 'Gold' in article.get('description')]

#         # Write filtered data to a JSON file
#         with open('filtered_data.json', 'w') as f:
#             json.dump(filtered_data, f, indent=4)

#         print("Filtered data saved successfully.")
#         # Read data from the JSON file
#     else: 
#         print("Error occurred while fetching data:", response.status_code)

#     ##############################################################################################

#     with open('filtered_data.json', 'r') as f:
#         filtered_data = json.load(f)

#     # Iterate through each record and print
#     gold_lst=[]
#     for record in filtered_data:
#         gold_lst.append(record)
#     key_values = []
#     for d in gold_lst:
#         for key, value in d.items():
#             key_values.append(f"{key}:{value}\n")
#     return key_values

#fetch live value
# def fetch_gold_price():
#     api_key = 'XPBYWCTLL30N220H'  # Replace 'YOUR_API_KEY' with your actual API key

#     try:
#         response = requests.get(f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=XAU&apikey={api_key}')
#         data = response.json()

#         # Extract the gold price from the response
#         gold_price = data['Global Quote']['05. price']
#         return gold_price
#     except Exception as e:
#         print('Error fetching gold price:', e)
#         return None


@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    gold=gold_news()
    prediction_data = predict_next_day()
    #live_price=fetch_gold_price()
    return templates.TemplateResponse("home.html", {"request": request,'data':gold,'prediction_data ':prediction_data})
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
