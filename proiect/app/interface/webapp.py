import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import yfinance as yf
from datetime import datetime
import torch

# Ensure the project root is in the sys.path
from app.algo.calculate_combine_risk_score import calculate_combined_risk_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.algo.calculate_MC_VaR_risk_score import calculate_mc_var_risk_score
from app.analysis.technical_preprocessor import preprocess_and_combine_data
from app.data_fetcher.news import fetch_news_rss, analyze_sentiments
from app.algo.calculate_technical_risk_score import calculate_technical_risk_score
from app.algo.calculate_financial_risk_score import calculate_financial_risk_score
from app.algo.calculate_sentiment_risk_score import calculate_sentiment_risk_score
from app.algo.calculate_technical_indicators_risk_score import calculate_technical_indicators_risk_score

app = Flask(__name__)

# Configure the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '12345678'

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

def fetch_market_data():
    # Fetch data for major indices and VIX
    sp500 = yf.Ticker('^GSPC')
    nasdaq = yf.Ticker('^IXIC')
    dowjones = yf.Ticker('^DJI')
    vix = yf.Ticker('^VIX')
    gold = yf.Ticker('GC=F')
    oil = yf.Ticker('CL=F')

    sp500_hist = sp500.history(period='1y')
    nasdaq_hist = nasdaq.history(period='1y')
    dowjones_hist = dowjones.history(period='1y')
    vix_hist = vix.history(period='1y')
    gold_hist = gold.history(period='1y')
    oil_hist = oil.history(period='1y')

    return sp500_hist, nasdaq_hist, dowjones_hist, vix_hist, gold_hist, oil_hist

def calculate_volatility(index_hist):
    log_returns = np.log(index_hist['Close'] / index_hist['Close'].shift(1))
    volatility = log_returns.std() * np.sqrt(252)  # Annualize the volatility
    return volatility

def calculate_vix_risk(vix_hist):
    avg_vix = vix_hist['Close'].mean()
    current_vix = vix_hist['Close'][-1]

    vix_risk = np.clip((current_vix - avg_vix) / avg_vix * 10, 1, 10)
    return vix_risk

def calculate_market_risk(sp500_hist, nasdaq_hist, dowjones_hist, vix_hist, gold_hist, oil_hist):
    sp500_volatility = calculate_volatility(sp500_hist)
    nasdaq_volatility = calculate_volatility(nasdaq_hist)
    dowjones_volatility = calculate_volatility(dowjones_hist)
    gold_volatility = calculate_volatility(gold_hist)
    oil_volatility = calculate_volatility(oil_hist)
    vix_risk = calculate_vix_risk(vix_hist)

    avg_volatility = (sp500_volatility + nasdaq_volatility + dowjones_volatility + gold_volatility + oil_volatility) / 5

    market_risk = np.clip(avg_volatility * 40, 1, 10)  # Scale to 1-10 range

    combined_risk = (market_risk + vix_risk) / 2
    return round(combined_risk, 1)  # Format to one decimal place

def get_market_risk_score():
    sp500_hist, nasdaq_hist, dowjones_hist, vix_hist, gold_hist, oil_hist = fetch_market_data()
    market_risk_score = calculate_market_risk(sp500_hist, nasdaq_hist, dowjones_hist, vix_hist, gold_hist, oil_hist)
    return f"{market_risk_score:.1f}"

def format_risk_scores(risk_scores):
    return {key: f"{value:.1f}" for key, value in risk_scores.items()}

def generate_candlestick_plot(price_data):
    fig = go.Figure(data=[go.Candlestick(
        x=price_data.index,
        open=price_data['Open'],
        high=price_data['High'],
        low=price_data['Low'],
        close=price_data['Close']
    )])
    fig.update_layout(title='Stock Price Over Time (Candlestick)', xaxis_title='Date', yaxis_title='Price')
    return fig.to_html(full_html=False)

def generate_macd_plot(macd_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=macd_data.index, y=macd_data['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=macd_data.index, y=macd_data['MACD_Signal'], mode='lines', name='MACD Signal'))
    fig.update_layout(title='MACD', xaxis_title='Date', yaxis_title='Value')
    return fig.to_html(full_html=False)

# Define models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    confirmed = db.Column(db.Boolean, default=False)
    portfolio = db.relationship('Portfolio', backref='user', lazy=True)

class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    investment = db.Column(db.Float, default=0.0)  # New investment column

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def generate_technical_plot(indicator_data, indicator_name):
    fig = go.Figure()

    if indicator_name == 'MACD':
        fig.add_trace(go.Scatter(x=indicator_data.index, y=indicator_data['MACD'], mode='lines', name='MACD'))
        fig.add_trace(go.Scatter(x=indicator_data.index, y=indicator_data['MACD_Signal'], mode='lines', name='MACD Signal'))
    else:
        if isinstance(indicator_data, pd.DataFrame):
            fig.add_trace(go.Scatter(x=indicator_data.index, y=indicator_data[indicator_name], mode='lines', name=indicator_name))
        else:  # Handle case where indicator_data is a Series
            fig.add_trace(go.Scatter(x=indicator_data.index, y=indicator_data, mode='lines', name=indicator_name))

    fig.update_layout(title=f'{indicator_name} Over Time', xaxis_title='Date', yaxis_title=indicator_name)
    return fig.to_html(full_html=False)

# Function to generate technical plots
def generate_technical_plots(technical_data):
    technical_plots = {}
    for column in technical_data.columns:
        if column not in ['Open', 'High', 'Low', 'Close', 'Volume']:
            technical_plot = go.Figure()
            technical_plot.add_trace(go.Scatter(x=technical_data.index, y=technical_data[column], mode='lines', name=column))
            technical_plot.update_layout(title=column, xaxis_title='Date', yaxis_title='Value')
            technical_plots[column] = technical_plot.to_html(full_html=False)
    return technical_plots

def generate_price_plot(price_data, show_sma_50=False, show_sma_200=False, show_bollinger_bands=False, chart_type='line'):
    fig = go.Figure()

    if chart_type == 'line':
        fig.add_trace(go.Scatter(x=price_data.index, y=price_data['Close'], mode='lines', name='Close Price'))
    else:  # Candlestick
        fig.add_trace(go.Candlestick(x=price_data.index,
                                     open=price_data['Open'],
                                     high=price_data['High'],
                                     low=price_data['Low'],
                                     close=price_data['Close'],
                                     name='Candlestick'))

    if show_sma_50:
        fig.add_trace(go.Scatter(x=price_data.index, y=price_data['SMA_50'], mode='lines', name='SMA 50'))

    if show_sma_200:
        fig.add_trace(go.Scatter(x=price_data.index, y=price_data['SMA_200'], mode='lines', name='SMA 200'))

    if show_bollinger_bands:
        if 'Bollinger_High' in price_data.columns and 'Bollinger_Low' in price_data.columns:
            fig.add_trace(go.Scatter(x=price_data.index, y=price_data['Bollinger_High'], mode='lines', name='Bollinger High'))
            fig.add_trace(go.Scatter(x=price_data.index, y=price_data['Bollinger_Low'], mode='lines', name='Bollinger Low'))

    fig.update_layout(title='Stock Price Over Time', xaxis_title='Date', yaxis_title='Price')
    return fig.to_html(full_html=False)




def generate_sentiment_plot(sentiment_data):
    titles = [entry['title'] for entry in sentiment_data]
    scores = [entry['sentiment'] for entry in sentiment_data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=titles, y=scores, mode='lines+markers', name='Sentiment Score'))

    fig.update_layout(
        title='Sentiment Analysis Over Time',
        xaxis=dict(
            showticklabels=False
        ),
        yaxis_title='Sentiment Score'
    )
    return fig.to_html(full_html=False)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    symbol = request.form.get('symbol') or request.args.get('symbol')
    if symbol:
        chart_type = request.form.get('chart_type', 'line')
        show_sma_50 = 'show_sma_50' in request.form
        show_sma_200 = 'show_sma_200' in request.form
        show_bollinger_bands = 'show_bollinger_bands' in request.form

        # Combine MACD and MACD_Signal into a single option
        show_macd = 'show_macd' in request.form

        # Filter out specific options
        active_indicators = [key.split('_', 1)[1] for key in request.form.keys() if
                             key.startswith('show_') and key not in ['show_sma_50', 'show_sma_200',
                                                                     'show_bollinger_bands', 'show_macd']]

        # Preprocess and combine data
        technical_data = preprocess_and_combine_data(symbol)

        # Filter out Bollinger_Upper and Bollinger_Lower from technical_indicators
        technical_indicators = technical_data.columns.difference(
            ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'Bollinger_High', 'Bollinger_Low', 'MACD', 'MACD_Signal'])

        price_plot = generate_price_plot(technical_data, show_sma_50=show_sma_50, show_sma_200=show_sma_200,
                                         show_bollinger_bands=show_bollinger_bands, chart_type=chart_type)

        technical_plots = {}
        for indicator in active_indicators:
            if indicator in technical_data.columns:
                technical_plots[indicator] = generate_technical_plot(technical_data[[indicator]], indicator)

        # Handle MACD separately
        if show_macd and 'MACD' in technical_data.columns and 'MACD_Signal' in technical_data.columns:
            technical_plots['MACD'] = generate_macd_plot(technical_data[['MACD', 'MACD_Signal']])

        articles = fetch_news_rss(symbol)
        analyzed_articles = analyze_sentiments(articles)
        sentiment_plot = generate_sentiment_plot(analyzed_articles)

        technical_risk_score = calculate_technical_risk_score(symbol)
        financial_risk_score = calculate_financial_risk_score(symbol)
        sentiment_risk_score = calculate_sentiment_risk_score(symbol)
        var_risk_score = calculate_mc_var_risk_score(symbol)
        technical_indicators_risk_score = calculate_technical_indicators_risk_score(symbol)

        risk_scores = {
            'Technical Risk Score': technical_risk_score,
            'Financial Risk Score': financial_risk_score,
            'Sentiment Risk Score': sentiment_risk_score,
            'VAR Risk Score': var_risk_score,
            'Technical Indicators Risk Score': technical_indicators_risk_score
        }

        formatted_risk_scores = format_risk_scores(risk_scores)
        combined_risk_score = f"{(sum(risk_scores.values()) / len(risk_scores)):.1f}"

        return render_template(
            'analysis.html',
            symbol=symbol,
            combined_risk_score=combined_risk_score,
            risk_scores=formatted_risk_scores,
            price_plot=price_plot,
            technical_plots=technical_plots,
            sentiment_plot=sentiment_plot,
            show_sma_50=show_sma_50,
            show_sma_200=show_sma_200,
            show_bollinger_bands=show_bollinger_bands,
            chart_type=chart_type,
            technical_indicators=technical_indicators,
            active_indicators=active_indicators,
            show_macd=show_macd
        )
    return render_template('analyze.html')



@app.route('/analyze/<symbol>')
@login_required
def analyze_symbol(symbol):
    return redirect(url_for('analyze', symbol=symbol))

@app.route('/graphs/<symbol>')
def graphs(symbol):
    technical_data = preprocess_and_combine_data(symbol)
    technical_plot = generate_technical_plot(technical_data)
    price_plot = generate_price_plot(technical_data)

    articles = fetch_news_rss(symbol)
    analyzed_articles = analyze_sentiments(articles)
    sentiment_plot = generate_sentiment_plot(analyzed_articles)

    return render_template(
        'graphs.html',
        symbol=symbol,
        technical_plot=technical_plot,
        price_plot=price_plot,
        sentiment_plot=sentiment_plot
    )

@app.route('/historical/<symbol>')
def historical(symbol):
    technical_data = preprocess_and_combine_data(symbol)
    technical_plot = generate_technical_plot(technical_data)
    price_plot = generate_price_plot(technical_data)
    return render_template('historical.html', symbol=symbol, technical_plot=technical_plot, price_plot=price_plot)

@app.route('/news/<symbol>')
def news(symbol):
    articles = fetch_news_rss(symbol)
    analyzed_articles = analyze_sentiments(articles)
    sentiment_plot = generate_sentiment_plot(analyzed_articles)
    return render_template('news.html', symbol=symbol, articles=analyzed_articles, sentiment_plot=sentiment_plot)

@app.route('/api/risk_score', methods=['POST'])
def api_risk_score():
    data = request.get_json()
    symbol = data['symbol']
    technical_risk_score = calculate_technical_risk_score(symbol)
    financial_risk_score = calculate_financial_risk_score(symbol)
    sentiment_risk_score = calculate_sentiment_risk_score(symbol)
    var_risk_score = calculate_mc_var_risk_score(symbol)
    technical_indicators_risk_score = calculate_technical_indicators_risk_score(symbol)

    risk_scores = {
        'Technical Risk Score': technical_risk_score,
        'Financial Risk Score': financial_risk_score,
        'Sentiment Risk Score': sentiment_risk_score,
        'VAR Risk Score': var_risk_score,
        'Technical Indicators Risk Score': technical_indicators_risk_score
    }

    combined_risk_score = (
        technical_risk_score + financial_risk_score + sentiment_risk_score + var_risk_score + technical_indicators_risk_score) / 5

    return jsonify({'symbol': symbol, 'risk_scores': risk_scores, 'combined_risk_score': combined_risk_score})

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(email=email).first():
            flash('Email address already exists', 'danger')
            return redirect(url_for('signup'))

        new_user = User(email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Your account has been created!', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and user.password == password:
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login failed. Check your email and password', 'danger')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/portfolio', methods=['GET', 'POST'])
@login_required
def portfolio():
    if request.method == 'POST':
        symbol = request.form['symbol']
        investment = float(request.form['investment'])  # Get the investment value
        if symbol:
            existing_stock = Portfolio.query.filter_by(user_id=current_user.id, symbol=symbol).first()
            if not existing_stock:
                new_stock = Portfolio(symbol=symbol, user_id=current_user.id, investment=investment)
                db.session.add(new_stock)
                db.session.commit()
            else:
                existing_stock.investment = investment
                db.session.commit()

    user_portfolio = Portfolio.query.filter_by(user_id=current_user.id).all()
    portfolio_data = []
    total_investment = sum(stock.investment for stock in user_portfolio)
    total_combined_risk = 0
    total_weighted_risk = 0

    for stock in user_portfolio:
        technical_risk_score = calculate_technical_risk_score(stock.symbol)
        financial_risk_score = calculate_financial_risk_score(stock.symbol)
        sentiment_risk_score = calculate_sentiment_risk_score(stock.symbol)
        var_risk_score = calculate_mc_var_risk_score(stock.symbol)
        technical_indicators_risk_score = calculate_technical_indicators_risk_score(stock.symbol)

        combined_risk_score = calculate_combined_risk_score(
            technical_risk_score,
            financial_risk_score,
            sentiment_risk_score,
            var_risk_score,
            technical_indicators_risk_score
        )

        total_combined_risk += combined_risk_score
        weighted_risk_score = (combined_risk_score * stock.investment) / total_investment if total_investment > 0 else combined_risk_score
        total_weighted_risk += weighted_risk_score

        risk_scores = {
            'combined_risk_score': combined_risk_score,
            'technical_risk_score': technical_risk_score,
            'financial_risk_score': financial_risk_score,
            'sentiment_risk_score': sentiment_risk_score,
            'var_risk_score': var_risk_score,
            'technical_indicators_risk_score': technical_indicators_risk_score
        }

        # Determine the risk category
        if combined_risk_score < 3:
            risk_category = 'Low'
        elif combined_risk_score < 5:
            risk_category = 'Medium'
        else:
            risk_category = 'High'

        portfolio_data.append({
            'symbol': stock.symbol,
            'investment': stock.investment,
            'risk_scores': {key: f"{value:.1f}" for key, value in risk_scores.items()},
            'risk_category': risk_category
        })

    average_combined_risk = round(total_combined_risk / len(user_portfolio), 1) if user_portfolio else 0
    weighted_average_risk = round(total_weighted_risk, 1)

    market_risk_score = get_market_risk_score()  # Get the market risk score

    return render_template('portfolio.html', portfolio=portfolio_data, average_combined_risk=average_combined_risk, weighted_average_risk=weighted_average_risk, market_risk_score=market_risk_score)


@app.route('/delete_symbol', methods=['POST'])
@login_required
def delete_symbol():
    symbol = request.form['symbol']
    stock = Portfolio.query.filter_by(user_id=current_user.id, symbol=symbol).first()
    if stock:
        db.session.delete(stock)
        db.session.commit()
    return redirect(url_for('portfolio'))

@app.route('/delete_stock/<symbol>', methods=['POST'])
@login_required
def delete_stock(symbol):
    stock_to_delete = Portfolio.query.filter_by(user_id=current_user.id, symbol=symbol).first()
    if stock_to_delete:
        db.session.delete(stock_to_delete)
        db.session.commit()
    return redirect(url_for('portfolio'))

@app.route('/add_symbol', methods=['POST'])
@login_required
def add_symbol():
    symbol = request.form['symbol']

    # Check if the symbol is already in the user's portfolio
    existing_symbol = Portfolio.query.filter_by(user_id=current_user.id, symbol=symbol).first()

    if existing_symbol is None:
        # Add the new symbol to the portfolio
        new_symbol = Portfolio(symbol=symbol, user_id=current_user.id)
        db.session.add(new_symbol)
        db.session.commit()
        flash(f'Symbol {symbol} added to your portfolio.', 'success')
    else:
        flash(f'Symbol {symbol} is already in your portfolio.', 'warning')

    return redirect(url_for('portfolio'))

@app.route('/explanation')
def explanation():
    return render_template('explanation.html')

if __name__ == "__main__":
    app.run(debug=True)
