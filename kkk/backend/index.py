from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class SentimentResult(BaseModel):
    text: str
    sentiment: str

def preprocess_text(text: str) -> str:
    # Remove URLs, special characters, and numbers
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(filtered_tokens)

def analyze_sentiment(text: str) -> str:
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"

@app.post("/analyze-csv/")
async def analyze_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file.file)
        
        # Check if the CSV has at least one column
        if df.shape[1] < 1:
            raise HTTPException(status_code=400, detail="CSV must contain at least one column")
        
        # Use the first column as comments
        comment_column = df.columns[0]
        comments = df[comment_column].dropna().tolist()  # Drop NaN values and convert to list
        
        # Analyze sentiment for each comment
        results = []
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        
        for comment in comments:
            cleaned_text = preprocess_text(comment)
            sentiment = analyze_sentiment(cleaned_text)
            results.append({"text": comment, "sentiment": sentiment})
            sentiment_counts[sentiment] += 1
        
        return {"results": results, "sentiment_counts": sentiment_counts}
    
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the CSV file: {str(e)}")

@app.get("/")
async def read_root():
    return {"message": "Sentiment Analysis API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)