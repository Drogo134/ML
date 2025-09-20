import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIIntegration:
    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HumanBehaviorPrediction/1.0',
            'Content-Type': 'application/json'
        })
        
    def fetch_external_behavior_data(self, api_endpoint: str, params: Dict = None) -> pd.DataFrame:
        try:
            response = self.session.get(api_endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
            
            logger.info(f"Fetched {len(df)} records from {api_endpoint}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from {api_endpoint}: {e}")
            return pd.DataFrame()
    
    def fetch_demographic_data(self, country: str = 'US') -> pd.DataFrame:
        api_endpoints = {
            'US': 'https://api.census.gov/data/2020/acs/acs5',
            'EU': 'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data',
            'UK': 'https://api.ons.gov.uk/dataset'
        }
        
        if country not in api_endpoints:
            logger.warning(f"No API endpoint for country: {country}")
            return pd.DataFrame()
        
        try:
            if country == 'US':
                params = {
                    'get': 'B01001_001E,B01001_002E,B01001_026E',
                    'for': 'state:*',
                    'key': 'YOUR_CENSUS_API_KEY'
                }
                df = self.fetch_external_behavior_data(api_endpoints[country], params)
                
                if not df.empty:
                    df.columns = ['total_population', 'male_population', 'female_population', 'state']
                    df['country'] = country
                    df['timestamp'] = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching demographic data for {country}: {e}")
            return pd.DataFrame()
    
    def fetch_economic_data(self, country: str = 'US') -> pd.DataFrame:
        try:
            if country == 'US':
                api_endpoint = 'https://api.stlouisfed.org/fred/series/observations'
                params = {
                    'series_id': 'UNRATE',
                    'api_key': 'YOUR_FRED_API_KEY',
                    'file_type': 'json',
                    'limit': 100
                }
                
                df = self.fetch_external_behavior_data(api_endpoint, params)
                
                if not df.empty:
                    df['country'] = country
                    df['timestamp'] = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching economic data for {country}: {e}")
            return pd.DataFrame()
    
    def fetch_weather_data(self, city: str, days: int = 30) -> pd.DataFrame:
        try:
            api_endpoint = 'https://api.openweathermap.org/data/2.5/forecast'
            params = {
                'q': city,
                'appid': 'YOUR_OPENWEATHER_API_KEY',
                'units': 'metric'
            }
            
            df = self.fetch_external_behavior_data(api_endpoint, params)
            
            if not df.empty and 'list' in df.columns:
                weather_data = []
                for item in df['list'].iloc[0]:
                    weather_data.append({
                        'datetime': item['dt_txt'],
                        'temperature': item['main']['temp'],
                        'humidity': item['main']['humidity'],
                        'pressure': item['main']['pressure'],
                        'weather': item['weather'][0]['main'],
                        'city': city
                    })
                
                df = pd.DataFrame(weather_data)
                df['timestamp'] = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching weather data for {city}: {e}")
            return pd.DataFrame()
    
    def fetch_social_media_sentiment(self, keyword: str, days: int = 7) -> pd.DataFrame:
        try:
            api_endpoint = 'https://api.twitter.com/2/tweets/search/recent'
            params = {
                'query': f'{keyword} -is:retweet lang:en',
                'max_results': 100,
                'tweet.fields': 'created_at,public_metrics,context_annotations'
            }
            
            headers = {
                'Authorization': 'Bearer YOUR_TWITTER_BEARER_TOKEN'
            }
            
            response = self.session.get(api_endpoint, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data:
                tweets = []
                for tweet in data['data']:
                    tweets.append({
                        'text': tweet['text'],
                        'created_at': tweet['created_at'],
                        'retweet_count': tweet['public_metrics']['retweet_count'],
                        'like_count': tweet['public_metrics']['like_count'],
                        'reply_count': tweet['public_metrics']['reply_count'],
                        'keyword': keyword
                    })
                
                df = pd.DataFrame(tweets)
                df['timestamp'] = datetime.now()
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching social media data for {keyword}: {e}")
            return pd.DataFrame()
    
    def fetch_news_sentiment(self, topic: str, days: int = 7) -> pd.DataFrame:
        try:
            api_endpoint = 'https://newsapi.org/v2/everything'
            params = {
                'q': topic,
                'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'apiKey': 'YOUR_NEWS_API_KEY',
                'pageSize': 100
            }
            
            df = self.fetch_external_behavior_data(api_endpoint, params)
            
            if not df.empty and 'articles' in df.columns:
                news_data = []
                for article in df['articles'].iloc[0]:
                    news_data.append({
                        'title': article['title'],
                        'description': article['description'],
                        'publishedAt': article['publishedAt'],
                        'source': article['source']['name'],
                        'url': article['url'],
                        'topic': topic
                    })
                
                df = pd.DataFrame(news_data)
                df['timestamp'] = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching news data for {topic}: {e}")
            return pd.DataFrame()
    
    def fetch_market_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        try:
            api_endpoint = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2023-01-01/2023-12-31'
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'apikey': 'YOUR_POLYGON_API_KEY'
            }
            
            df = self.fetch_external_behavior_data(api_endpoint, params)
            
            if not df.empty and 'results' in df.columns:
                market_data = []
                for result in df['results'].iloc[0]:
                    market_data.append({
                        'date': datetime.fromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d'),
                        'open': result['o'],
                        'high': result['h'],
                        'low': result['l'],
                        'close': result['c'],
                        'volume': result['v'],
                        'symbol': symbol
                    })
                
                df = pd.DataFrame(market_data)
                df['timestamp'] = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def enrich_behavior_data(self, df: pd.DataFrame, enrichment_types: List[str] = None) -> pd.DataFrame:
        if enrichment_types is None:
            enrichment_types = ['demographic', 'economic', 'weather']
        
        enriched_df = df.copy()
        
        for enrichment_type in enrichment_types:
            try:
                if enrichment_type == 'demographic':
                    demo_data = self.fetch_demographic_data()
                    if not demo_data.empty:
                        enriched_df = self._merge_data(enriched_df, demo_data, 'demographic')
                
                elif enrichment_type == 'economic':
                    econ_data = self.fetch_economic_data()
                    if not econ_data.empty:
                        enriched_df = self._merge_data(enriched_df, econ_data, 'economic')
                
                elif enrichment_type == 'weather':
                    weather_data = self.fetch_weather_data('New York')
                    if not weather_data.empty:
                        enriched_df = self._merge_data(enriched_df, weather_data, 'weather')
                
                elif enrichment_type == 'social_media':
                    social_data = self.fetch_social_media_sentiment('technology')
                    if not social_data.empty:
                        enriched_df = self._merge_data(enriched_df, social_data, 'social_media')
                
                elif enrichment_type == 'news':
                    news_data = self.fetch_news_sentiment('business')
                    if not news_data.empty:
                        enriched_df = self._merge_data(enriched_df, news_data, 'news')
                
                elif enrichment_type == 'market':
                    market_data = self.fetch_market_data('AAPL')
                    if not market_data.empty:
                        enriched_df = self._merge_data(enriched_df, market_data, 'market')
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error enriching data with {enrichment_type}: {e}")
                continue
        
        return enriched_df
    
    def _merge_data(self, main_df: pd.DataFrame, enrichment_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if enrichment_df.empty:
            return main_df
        
        for col in enrichment_df.columns:
            if col not in main_df.columns:
                main_df[f'{prefix}_{col}'] = np.random.choice(enrichment_df[col].dropna().values, len(main_df))
        
        return main_df
    
    def save_enriched_data(self, df: pd.DataFrame, filename: str = None) -> str:
        if filename is None:
            filename = f"enriched_behavior_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = self.config.DATA_DIR / filename
        df.to_csv(filepath, index=False)
        
        logger.info(f"Enriched data saved to {filepath}")
        return str(filepath)
    
    def create_api_config(self, config_file: str = 'api_config.json'):
        api_config = {
            'census_api_key': 'YOUR_CENSUS_API_KEY',
            'fred_api_key': 'YOUR_FRED_API_KEY',
            'openweather_api_key': 'YOUR_OPENWEATHER_API_KEY',
            'twitter_bearer_token': 'YOUR_TWITTER_BEARER_TOKEN',
            'news_api_key': 'YOUR_NEWS_API_KEY',
            'polygon_api_key': 'YOUR_POLYGON_API_KEY',
            'rate_limits': {
                'census': 500,
                'fred': 120,
                'openweather': 1000,
                'twitter': 300,
                'news': 1000,
                'polygon': 5
            }
        }
        
        config_path = self.config.BASE_DIR / config_file
        with open(config_path, 'w') as f:
            json.dump(api_config, f, indent=2)
        
        logger.info(f"API configuration saved to {config_path}")
        return str(config_path)
