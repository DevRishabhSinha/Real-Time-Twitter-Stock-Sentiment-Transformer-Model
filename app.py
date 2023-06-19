"""
Import statements for tweepy, transformers, yfinance, scikit-learn
numpy, flask, and bokeh modules using pip and python interpreter
preferences
"""
import tweepy
from transformers import pipeline
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from flask import Flask, render_template, request
from datetime import datetime
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.io import curdoc
import config
from random import randrange

"""
API key credentials are read from the configuration file.
These are used to authenticate the user's Twitter account
to allow interaction with the Twitter API.
"""

consumer_key = config.consumer_key
consumer_secret = config.consumer_secret
access_token = config.access_token
access_token_secret = config.access_token_secret

"""
Authentication process for Twitter API using tweepy's OAuthHandler.
This uses the consumer keys and access tokens to authenticate
the application.
"""

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

"""
Loading the pretrained distilbert model for sentiment analysis.
This model has been fine-tuned on the SST-2 task, which
involves classifying text as expressing positive or negative sentiment.
"""

classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

"""
Initialize the Flask application. This sets up the framework
for a web application where the routes are defined to
dictate the behavior of the application.
"""

app = Flask(__name__)

"""
Definition of the application's route, which corresponds to
the home page. This function will be executed when this
route is accessed. The methods parameter indicates that
this route can respond to HTTP GET and POST requests.
"""


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Initializing a set of variables for average sentiment, average sentiment label,
    predicted close price, tweets text, predicted close price for today, list of
    predicted close prices, and components of the Bokeh plot.
    """
    avg_sentiment = None
    avg_sentiment_label = None
    predicted_close = None
    tweets_text = []
    predicted_close_today = None
    predicted_closes = []
    plot_script, plot_div = None, None

    """
    Checks if the HTTP request method is POST. This would 
    mean that the form on the page was submitted.
    """
    if request.method == 'POST':
        stock_label = request.form['stock_label']

        """
        Using the tweepy API, we can search for recent tweets containing the stock_label 
        in English. The tweets are then iterated through to analyze the sentiment of each 
        tweet using the previously loaded sentiment analysis model.
        """
        tweets = tweepy.Cursor(api.search_tweets, q=stock_label, lang="en").items(20)
        sentiments = []
        sentiments_label = []

        pos_sentiment_sum = 0
        neg_sentiment_sum = 0
        pos_count = 0
        neg_count = 0
        """
        Loop through the returned tweets, extract the text, and classify 
        sentiment using the distilbert model. Sentiment scores and labels are 
        collected for later use.
        """
        for tweet in tweets:
            """
            Utilizes the model to predict the sentiment of the tweet.
            """
            result = classifier(tweet.text)
            sentiment_score = result[0]['score']
            sentiment_label = result[0]['label']
            sentiments.append(sentiment_score)
            sentiments_label.append(sentiment_label)
            tweet_url = f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}"
            tweets_text.append((tweet.text, sentiment_score, sentiment_label, tweet_url))

            """
                Based on the sentiment label, update the respective sentiment sum 
                and count.
                """
            if sentiment_label == "POSITIVE":
                pos_sentiment_sum += sentiment_score
                pos_count += 1
            elif sentiment_label == "NEGATIVE":
                # Assigning a negative score for negative sentiments
                neg_sentiment_sum -= sentiment_score
                neg_count += 1
        """
        Calculate the average sentiment by dividing the sum of positive and 
        negative sentiment scores by the total number of tweets. 
        The avg_sentiment_label is determined based on the sign of avg_sentiment.
        """
        total_count = pos_count + neg_count
        avg_sentiment = (pos_sentiment_sum + neg_sentiment_sum) / total_count
        avg_sentiment_label = "POSITIVE" if avg_sentiment >= 0 else "NEGATIVE"

        """
        Using the yfinance library, historical stock data is downloaded 
        for the given stock label. The data is processed and reshaped 
        in order to be used as input for the linear regression model.
        """
        stock_data = yf.download(stock_label, start='2023-01-01', end=datetime.today().strftime('%Y-%m-%d'))

        """
        The downloaded stock data is then preprocessed by converting the index 
        into a column and converting the date to a specific string format.
        """
        stock_data['Date'] = stock_data.index
        stock_data['Date'] = stock_data['Date'].dt.strftime('%Y-%m-%d')

        """
        The relevant features for predicting the stock price are selected. 
        In this case, we use the adjusted close prices.
        """
        features = stock_data[['Adj Close']].values

        """
        The average sentiment is then reshaped and combined with the stock 
        data features in order to enrich the feature set with sentiment analysis.
        """
        avg_sentiment_reshaped = np.reshape(avg_sentiment, (-1, 1))
        avg_sentiment_repeated = np.repeat(avg_sentiment_reshaped, features.shape[0], axis=0)
        combined_features = np.hstack((features, avg_sentiment_repeated))

        """
        The enriched feature set is then split into training and testing 
        sets to allow for evaluation of the model's performance.
        """
        train_size = int(0.8 * len(combined_features))
        train_features = combined_features[:train_size]
        test_features = combined_features[train_size:]

        """
        Before training the model, the features are standardized using 
        Scikit-learn's StandardScaler. This makes the model less 
        sensitive to the scale of the features.
        """
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)

        """
        A Linear Regression model is trained on the scaled training 
        data. The model is then used to predict the closing prices on the 
        test set.
        """
        regression_model = LinearRegression()
        regression_model.fit(train_features_scaled, stock_data['Close'][:train_size])
        predicted_close = regression_model.predict(test_features_scaled)

        """
        The closing price for the current day is extracted from the 
        predicted close prices.
        """
        predicted_close_today = predicted_close[-1]

        """
        The predicted close prices along with their corresponding dates 
        are collected into a list for later use.
        """
        for i in range(len(predicted_close)):
            date = stock_data['Date'][train_size + i]
            predicted_closes.append((date, predicted_close[i]))

        """
        A Bokeh plot is created to visualize the real-time and predicted 
        close prices of the stock. The theme of the plot is set and 
        various elements of the plot are customized.
        """
        curdoc().theme = 'dark_minimal'
        p = figure(x_axis_type="datetime", title=f"Real-time and Predicted Close Prices: {stock_label}",
                   height=400, width=800)

        p.border_fill_color = "#292928"
        p.background_fill_color = "#292928"
        p.outline_line_color = None
        p.toolbar.logo = None
        p.toolbar_location = None
        p.xaxis.axis_label_text_color = "#FFA114"
        p.yaxis.axis_label_text_color = "#FFA114"
        p.title.text_color = "#FFA114"

        """
        The real-time and predicted close prices are plotted on the figure. 
        Components of the plot are then extracted for embedding in the web page.
        """
        p.line(stock_data.index[train_size:], predicted_close, legend_label="Predicted Close", color="orange")
        p.line(stock_data.index[train_size:], stock_data['Close'][train_size:], legend_label="Real-time Close",
               color="blue")

        # Extract the components
        plot_script, plot_div = components(p)
    """
    After all computations and the plot creation, the render_template
    method from Flask is used to render the 'index.html' page.
    The method also receives the calculated variables to be
    used and displayed on the rendered webpage.
    """
    return render_template('index.html', avg_sentiment=avg_sentiment, avg_sentiment_label=avg_sentiment_label,
                           predicted_close_today=predicted_close_today,
                           tweets_text=tweets_text, predicted_closes=predicted_closes,
                           script=plot_script, div=plot_div)


"""
Python's condition to check if this script is the main program
and not being imported as a module. If the condition is true,
the Flask application runs with specified parameters. 'Debug'
is set to True to provide detailed error messages in case of
an exception and 'port' is set to 5005 to specify the port
on which the application should run.
"""
if __name__ == '__main__':
    app.run(debug=True, port=5005)
