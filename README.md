# Real-Time-Twitter-Stock-Sentiment-Transformer-Model
The Real-Time Twitter Stock Sentiment Analysis used Python, Transformers, and Twitter API to analyze stock sentiments from real-time tweets. It involved data acquisition, preprocessing, Transformer-based model training, real-time sentiment classification, and visualizing sentiment trends. Key technologies included TensorFlow, Keras, and Tweepy. This tool integrates sentiment analysis and stock price prediction into a web application that allows users to visualize predicted stock prices, alongside sentiment analysis of relevant tweets. The tool leverages the Twitter API for sentiment analysis, and Yahoo Finance's stock price data to train a machine learning model for stock price prediction.

# Technologies & Libraries
The following technologies and libraries are used:

- **Python**: The tool is written in Python programming language.

* **Flask**: Flask is used to set up a web server that handles user requests and responses. This includes form submissions and rendering HTML templates with the calculated data.

+ **Tweepy**: Tweepy is a Python library for accessing the Twitter API. It is used to fetch recent tweets related to the specified stock.

- **Transformers**: The Transformers library by Hugging Face is used for sentiment analysis of the fetched tweets. Specifically, the tool uses the pre-trained DistilBERT model fine-tuned on the SST-2 task, which involves classifying text as expressing positive or negative sentiment.

* **yfinance**: Yahoo Finance's yfinance Python library is used to fetch historical stock price data for the specified stock.

+ **Scikit-learn**: Scikit-learn's Linear Regression model is used for predicting future stock prices. Also, the StandardScaler utility is used to standardize the feature set.

- **NumPy**: NumPy is used for data manipulation and mathematical computations, particularly reshaping and combining arrays.

* **Bokeh**: Bokeh is a Python interactive visualization library that targets modern web browsers for presentation. It is used to create a line plot of real-time and predicted stock prices.

# Workflow
1. The tool first authenticates the user's Twitter account to interact with the Twitter API using the Tweepy's OAuthHandler.

2. A pre-trained DistilBERT model is loaded for sentiment analysis. This model is fine-tuned on the SST-2 task, which involves classifying text as expressing positive or negative sentiment.

3. The Flask application is initialized and a route corresponding to the home page is defined. The application checks if an HTTP request method is POST, indicating that the form on the page was submitted.

4. Upon form submission, the tool fetches recent tweets containing the stock label in English. It iterates through each tweet and uses the pre-loaded model to classify the sentiment of each tweet.

5. An average sentiment score is calculated, and a label (POSITIVE or NEGATIVE) is assigned based on this average score.

6. The tool then fetches historical stock data for the specified stock using yfinance. This data is preprocessed, combined with the average sentiment score, and then split into training and testing sets.
   
7. The training set is used to train a Linear Regression model, which is then used to predict closing prices on the test set.

8. A Bokeh plot is created to visualize the real-time and predicted closing prices of the stock. The plot is customized according to the theme and the real-time and predicted closing prices are plotted on the figure.

9. Finally, the Flask application renders an HTML page displaying the average sentiment, predicted closing price, relevant tweets, and the Bokeh plot.

Running the Application
To run the application, ensure that all the required libraries are installed. These can be installed using pip:

bash
Copy code
```
pip install flask tweepy transformers yfinance scikit-learn numpy bokeh
```
Also, ensure that you have the necessary API keys for the Twitter API in a config file.

Finally, run the application script:

bash
Copy code

```
python app.py
```
The application will start running on http://localhost:5005. You can open this URL in a web browser to interact with the application. You will be presented
