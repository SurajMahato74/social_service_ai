import requests
import json
import pandas as pd
from datetime import datetime
import re
import os
from gap_analysis_system import ServiceGapAnalyzer

class SocialMediaScraper:
    def __init__(self):
        self.analyzer = ServiceGapAnalyzer()
        self.keywords = [
            'nepal government', 'social service', 'health service', 'education nepal',
            'employment nepal', 'infrastructure nepal', 'corruption nepal',
            'government office', 'public service', 'nepal administration'
        ]
        
    def scrape_reddit_nepal(self):
        """Scrape Nepal-related posts from Reddit"""
        posts = []
        try:
            # Reddit API (no auth needed for public posts)
            url = "https://www.reddit.com/r/Nepal/hot.json?limit=50"
            headers = {'User-Agent': 'Nepal Social Service Monitor 1.0'}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                for post in data['data']['children']:
                    post_data = post['data']
                    title = post_data.get('title', '')
                    text = post_data.get('selftext', '')
                    
                    # Check if related to government services
                    if self.is_service_related(title + ' ' + text):
                        posts.append({
                            'text': title + ' ' + text,
                            'source': 'reddit_nepal',
                            'timestamp': datetime.now().isoformat(),
                            'url': 'https://reddit.com' + post_data.get('permalink', ''),
                            'score': post_data.get('score', 0)
                        })
        except Exception as e:
            print(f"Reddit scraping error: {e}")
        
        return posts
    
    def scrape_twitter_nepal(self):
        """Scrape Nepal government service tweets"""
        # Twitter API requires authentication. 
        # Returning empty list to avoid fake data.
        return []
    
    def scrape_facebook_nepal(self):
        """Scrape Nepal government service Facebook posts"""
        # Facebook scraping requires complex auth/selenium.
        # Returning empty list to avoid fake data.
        return []
    
    def is_service_related(self, text):
        """Check if text is related to government services"""
        text_lower = text.lower()
        service_keywords = [
            'government', 'service', 'health', 'education', 'employment',
            'infrastructure', 'corruption', 'office', 'administration',
            'hospital', 'school', 'job', 'road', 'electricity', 'water'
        ]
        
        return any(keyword in text_lower for keyword in service_keywords)
    
    def analyze_social_sentiment(self, posts):
        """Analyze sentiment of social media posts"""
        results = []
        
        for post in posts:
            # Analyze sentiment using gap analyzer
            sentiment_result = self.analyzer.analyze_sentiment_advanced(post['text'])
            
            # Extract services and locations
            services, locations = self.analyzer.extract_service_mentions(post['text'])
            
            results.append({
                'text': post['text'],
                'source': post['source'],
                'timestamp': post['timestamp'],
                'sentiment_score': sentiment_result['sentiment_score'],
                'sentiment_label': sentiment_result['sentiment_label'],
                'services': services,
                'locations': locations,
                'urgency_level': sentiment_result['urgency_level'],
                'negative_indicators': sentiment_result['negative_indicators']
            })
        
        return results
    
    def calculate_real_satisfaction(self, analyzed_posts):
        """Calculate real satisfaction metrics from social media"""
        if not analyzed_posts:
            return self.get_fallback_data()
        
        # Overall satisfaction calculation
        sentiment_scores = [post['sentiment_score'] for post in analyzed_posts]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        overall_satisfaction = max(0, min(100, (avg_sentiment + 1) * 50))  # Convert -1,1 to 0,100
        
        # Provincial breakdown
        provinces = {}
        for post in analyzed_posts:
            for location in post['locations'] or ['unspecified']:
                if location not in provinces:
                    provinces[location] = []
                provinces[location].append(post['sentiment_score'])
        
        provincial_satisfaction = {}
        for province, scores in provinces.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                satisfaction = max(0, min(100, (avg_score + 1) * 50))
                provincial_satisfaction[province] = satisfaction
        
        # Service category breakdown
        services = {}
        for post in analyzed_posts:
            for service in post['services'] or ['general']:
                if service not in services:
                    services[service] = []
                services[service].append(post['sentiment_score'])
        
        service_satisfaction = {}
        for service, scores in services.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                satisfaction = max(0, min(100, (avg_score + 1) * 50))
                service_satisfaction[service] = satisfaction
        
        return {
            'overall_satisfaction': round(overall_satisfaction, 1),
            'total_feedback': len(analyzed_posts),
            'provinces': provincial_satisfaction,
            'services': service_satisfaction,
            'last_updated': datetime.now().isoformat()
        }
    
    def get_fallback_data(self):
        """Return empty structure if no data available"""
        return {
            'overall_satisfaction': 0,
            'total_feedback': 0,
            'provinces': {},
            'services': {},
            'last_updated': datetime.now().isoformat()
        }
    
    def run_social_scraping(self):
        """Main function to scrape and analyze social media"""
        print("Starting social media scraping...")
        
        # Scrape from multiple sources
        all_posts = []
        all_posts.extend(self.scrape_reddit_nepal())
        all_posts.extend(self.scrape_twitter_nepal())
        all_posts.extend(self.scrape_facebook_nepal())
        
        print(f"Scraped {len(all_posts)} posts from social media")
        
        # Analyze sentiment
        analyzed_posts = self.analyze_social_sentiment(all_posts)
        
        # Calculate satisfaction metrics
        satisfaction_data = self.calculate_real_satisfaction(analyzed_posts)
        
        # Save results
        os.makedirs('social_media_data', exist_ok=True)
        
        with open('social_media_data/latest_social_sentiment.json', 'w') as f:
            json.dump(satisfaction_data, f, indent=2)
        
        # Save raw posts for analysis
        df = pd.DataFrame(analyzed_posts)
        df.to_csv('social_media_data/social_posts_analyzed.csv', index=False)
        
        print(f"Social media analysis complete. Overall satisfaction: {satisfaction_data['overall_satisfaction']}%")
        return satisfaction_data

if __name__ == "__main__":
    scraper = SocialMediaScraper()
    scraper.run_social_scraping()