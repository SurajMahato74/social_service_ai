#!/usr/bin/env python3
"""
Comprehensive Multi-Source Web Scraper for Nepal Social Service AI
Rapid data collection from 10+ sources for training data generation
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import re
import json
from pathlib import Path
import asyncio
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveNepaliScraper:
    """
    Multi-source web scraper for Nepal social service data
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Target sources for rapid data collection
        self.sources = {
            'news_sites': {
                'kantipur': {
                    'url': 'https://ekantipur.com',
                    'search_endpoint': 'https://ekantipur.com/search',
                    'sections': ['news', 'society', 'politics', 'economy'],
                    'method': 'beautifulsoup',
                    'rate_limit': 2
                },
                'setopati': {
                    'url': 'https://www.setopati.com',
                    'search_endpoint': 'https://www.setopati.com/search',
                    'sections': ['news', 'social', 'politics'],
                    'method': 'beautifulsoup',
                    'rate_limit': 3
                },
                'onlinekhabar': {
                    'url': 'https://www.onlinekhabar.com',
                    'search_endpoint': 'https://www.onlinekhabar.com/',
                    'sections': ['news', 'business', 'society'],
                    'method': 'beautifulsoup',
                    'rate_limit': 3
                },
                'ratopati': {
                    'url': 'https://www.ratopati.com',
                    'search_endpoint': 'https://www.ratopati.com/search',
                    'sections': ['news', 'political', 'social'],
                    'method': 'beautifulsoup',
                    'rate_limit': 2
                },
                'nagariknews': {
                    'url': 'https://nagariknews.nagariknetwork.com',
                    'search_endpoint': 'https://nagariknews.nagariknetwork.com/search',
                    'sections': ['news', 'society', 'local'],
                    'method': 'beautifulsoup',
                    'rate_limit': 2
                }
            },
            
            'government_sources': {
                # Removed nepal.gov.np sources due to connection issues
            },
            
            'social_platforms': {
                'facebook': {
                    'method': 'facebook_scraper',
                    'pages': {
                        'government': ['mohpgovnp', 'swcnepal', 'dprnepal'],
                        'news': ['kantipurdaily', 'onlinekhabar', 'setopati'],
                        'ngos': ['redcrossnepal', 'oxfaminnepal', 'unicefnepal']
                    },
                    'rate_limit': 2
                },
                'twitter': {
                    'method': 'api',
                    'hashtags': ['#NepalHealth', '#NepalEducation', '#NepalDevelopment'],
                    'rate_limit': 10
                }
            }
        }
        
        # Service keywords for content filtering
        self.service_keywords = {
            'health': ['स्वास्थ्य', 'अस्पताल', 'डाक्टर', 'औषधि', 'खोप', 'चिकित्सा', 'नर्स'],
            'education': ['शिक्षा', 'विद्यालय', 'पाठशाला', 'छात्रवृत्ति', 'शिक्षक', 'परीक्षा'],
            'employment': ['रोजगार', 'काम', 'जागिर', 'व्यवसाय', 'श्रमिक', 'तलब'],
            'infrastructure': ['सडक', 'पुल', 'पानी', 'बिजुली', 'निर्माण', 'विकास'],
            'social_welfare': ['राहत', 'सहायता', 'आपतकाल', 'बाढी', 'भूकम्प', 'उद्धार'],
            'agriculture': ['कृषि', 'किसान', 'बाली', 'धान', 'मकै', 'तरकारी']
        }
        
        self.rate_limits = {}
        
    def scrape_news_site(self, site_config: Dict, keywords: List[str], max_articles: int = 100) -> List[Dict]:
        """
        Scrape articles from a news website
        """
        articles = []
        site_name = site_config.get('url', 'unknown')
        
        try:
            logger.info(f"Starting scrape of {site_name}")
            
            # Search for articles with different keywords
            for keyword in keywords:
                if len(articles) >= max_articles:
                    break
                
                # Handle different search parameters
                if 'onlinekhabar' in site_name:
                    search_params = {'s': keyword}
                else:
                    search_params = {'q': keyword}
                
                # Add limit if supported
                if 'limit' not in search_params and 'onlinekhabar' not in site_name:
                    search_params['limit'] = 50
                
                try:
                    response = self.session.get(
                        site_config['search_endpoint'],
                        params=search_params,
                        timeout=30
                    )
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract articles based on site structure
                    article_elements = self._extract_article_elements(soup, site_config)
                    
                    for element in article_elements:
                        if len(articles) >= max_articles:
                            break
                            
                        article_data = self._extract_article_data(element, site_config)
                        if article_data:
                            article_data['search_keyword'] = keyword
                            articles.append(article_data)
                            
                    # Rate limiting
                    time.sleep(1 / site_config.get('rate_limit', 2))
                    
                except Exception as e:
                    logger.error(f"Error searching {keyword} on {site_name}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to scrape {site_name}: {str(e)}")
            
        logger.info(f"Scraped {len(articles)} articles from {site_name}")
        return articles
    
    def scrape_government_source(self, gov_config: Dict, keywords: List[str], max_docs: int = 50) -> List[Dict]:
        """
        Scrape documents from government websites
        """
        documents = []
        site_name = gov_config.get('url', 'unknown')
        
        try:
            logger.info(f"Starting government scrape of {site_name}")
            
            for keyword in keywords:
                if len(documents) >= max_docs:
                    break
                    
                search_params = {
                    'search': keyword,
                    'category': 'all'
                }
                
                try:
                    response = self.session.get(
                        gov_config['search_endpoint'],
                        params=search_params,
                        timeout=30
                    )
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    doc_elements = self._extract_document_elements(soup, gov_config)
                    
                    for element in doc_elements:
                        if len(documents) >= max_docs:
                            break
                            
                        doc_data = self._extract_document_data(element, gov_config)
                        if doc_data:
                            doc_data['search_keyword'] = keyword
                            documents.append(doc_data)
                            
                    time.sleep(2)  # Government sites are more sensitive
                    
                except Exception as e:
                    logger.error(f"Error searching {keyword} on {site_name}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to scrape government site {site_name}: {str(e)}")
            
        logger.info(f"Scraped {len(documents)} documents from {site_name}")
        return documents
    
    def _extract_article_elements(self, soup: BeautifulSoup, site_config: Dict) -> List:
        """
        Extract article elements based on site structure
        """
        # Common selectors for news articles
        selectors = [
            'article', '.article', '.news-item', '.post', '.story',
            '.entry', '[class*="article"]', '[class*="news"]'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                return elements
                
        # Fallback to div-based extraction
        return soup.find_all('div', class_=re.compile(r'(article|news|story|post)'))
    
    def _extract_article_data(self, element, site_config: Dict) -> Optional[Dict]:
        """
        Extract article data from element
        """
        try:
            # Extract title
            title_elem = element.find(['h1', 'h2', 'h3', 'h4']) or element.find(class_=re.compile(r'(title|headline)'))
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            # Extract content
            content_elem = element.find(['p', 'div']) or element.find(class_=re.compile(r'(content|body|summary)'))
            content = content_elem.get_text(strip=True) if content_elem else ""
            
            # Extract date
            date_elem = element.find(['time', 'span', 'div'], class_=re.compile(r'(date|time|timestamp)'))
            date = date_elem.get('datetime') or date_elem.get_text(strip=True) if date_elem else ""
            
            # Extract URL
            url_elem = element.find('a')
            url = url_elem.get('href', '') if url_elem else ""
            if url and url.startswith('/'):
                url = site_config['url'] + url
                
            # Filter by service relevance
            full_text = f"{title} {content}".lower()
            if not any(keyword in full_text for category in self.service_keywords.values() for keyword in category):
                return None
                
            # Categorize content
            category = self._categorize_content(full_text)
            
            return {
                'title': title,
                'content': content,
                'date': date,
                'url': url,
                'source': site_config.get('url', ''),
                'category': category,
                'scraped_at': datetime.now().isoformat(),
                'text_length': len(full_text)
            }
            
        except Exception as e:
            logger.error(f"Error extracting article data: {str(e)}")
            return None
    
    def _extract_document_elements(self, soup: BeautifulSoup, gov_config: Dict) -> List:
        """
        Extract document elements from government sites
        """
        selectors = [
            '.document', '.notice', '.announcement', '.policy', '.report',
            '[class*="document"]', '[class*="notice"]', '[class*="policy"]'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                return elements
                
        return soup.find_all('tr', class_=re.compile(r'(row|data|item)'))
    
    def _extract_document_data(self, element, gov_config: Dict) -> Optional[Dict]:
        """
        Extract document data from government site
        """
        try:
            # Extract title
            title_elem = element.find(['a', 'td', 'div'], class_=re.compile(r'(title|name|subject)'))
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            # Extract description
            desc_elem = element.find(['td', 'div'], class_=re.compile(r'(description|summary|detail)'))
            description = desc_elem.get_text(strip=True) if desc_elem else ""
            
            # Extract date
            date_elem = element.find(['td', 'span', 'div'], class_=re.compile(r'(date|time)'))
            date = date_elem.get('datetime') or date_elem.get_text(strip=True) if date_elem else ""
            
            # Filter by service relevance
            full_text = f"{title} {description}".lower()
            if not any(keyword in full_text for category in self.service_keywords.values() for keyword in category):
                return None
                
            category = self._categorize_content(full_text)
            
            return {
                'title': title,
                'description': description,
                'date': date,
                'source': gov_config.get('url', ''),
                'category': category,
                'scraped_at': datetime.now().isoformat(),
                'text_length': len(full_text)
            }
            
        except Exception as e:
            logger.error(f"Error extracting document data: {str(e)}")
            return None
    
    def _categorize_content(self, text: str) -> str:
        """
        Categorize content based on service keywords
        text: text to categorize
        """
        text_lower = text.lower()
        
        for category, keywords in self.service_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
                
        return 'general'
    
    def run_parallel_collection(self, target_samples: int = 10000) -> pd.DataFrame:
        """
        Run parallel data collection from all sources
        """
        all_data = []
        samples_per_source = max(100, target_samples // len(self.sources['news_sites']))
        
        # Prepare keywords for searching
        search_keywords = []
        for category_keywords in self.service_keywords.values():
            search_keywords.extend(category_keywords[:3])  # Top 3 keywords per category
        
        logger.info(f"Starting parallel collection targeting {target_samples} samples")
        logger.info(f"Using {len(search_keywords)} search keywords")
        
        # Use ThreadPoolExecutor for parallel scraping
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_source = {}
            
            # Submit news site scraping tasks
            for site_name, site_config in self.sources['news_sites'].items():
                future = executor.submit(
                    self.scrape_news_site,
                    site_config,
                    search_keywords,
                    samples_per_source
                )
                future_to_source[future] = f"news:{site_name}"
                
            # Submit government source scraping tasks
            for gov_name, gov_config in self.sources['government_sources'].items():
                future = executor.submit(
                    self.scrape_government_source,
                    gov_config,
                    search_keywords,
                    samples_per_source // 3  # Fewer from gov sites
                )
                future_to_source[future] = f"gov:{gov_name}"
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_source):
                source_info = future_to_source[future]
                try:
                    data = future.result()
                    all_data.extend(data)
                    completed += 1
                    
                    logger.info(f"Completed {source_info}: {len(data)} samples")
                    
                    if len(all_data) >= target_samples:
                        logger.info(f"Target reached: {len(all_data)} samples")
                        break
                        
                except Exception as e:
                    logger.error(f"Failed to get results from {source_info}: {str(e)}")
        
        # Add sentiment scores using simple keyword matching
        for item in all_data:
            text = f"{item.get('title', '')} {item.get('content', '')} {item.get('description', '')}".lower()
            
            # Simple sentiment analysis
            positive_words = ['राम्रो', 'सुधार', 'सफल', 'प्रगति', 'उपलब्धि', 'सकारात्मक']
            negative_words = ['नराम्रो', 'समस्या', 'गुनासो', 'ढिलो', 'असफल', 'दुःख']
            
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            if pos_count > neg_count:
                item['sentiment'] = 'positive'
            elif neg_count > pos_count:
                item['sentiment'] = 'negative'
            else:
                item['sentiment'] = 'neutral'
        
        df = pd.DataFrame(all_data)
        logger.info(f"Final collection: {len(df)} total samples")
        
        return df
    
    def save_collection(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save collected data to file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/collected/nepali_social_service_training_data_{timestamp}.csv"
            
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filename, index=False, encoding='utf-8')
        
        # Save metadata
        metadata = {
            'collection_timestamp': datetime.now().isoformat(),
            'total_samples': len(df),
            'sources': df['source'].unique().tolist() if 'source' in df.columns else [],
            'categories': df['category'].value_counts().to_dict() if 'category' in df.columns else {},
            'collection_method': 'comprehensive_web_scraping'
        }
        
        metadata_file = filename.replace('.csv', '_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data saved to {filename}")
        logger.info(f"Metadata saved to {metadata_file}")
        
        return filename


def main():
    """
    Main function for rapid data collection
    """
    logger.info("Starting comprehensive data collection for Nepal Social Service AI")
    
    scraper = ComprehensiveNepaliScraper()
    
    # Target: collect 10,000 training samples rapidly
    target_samples = 10000
    
    print(f"🎯 TARGET: Collecting {target_samples:,} training samples")
    print("📊 SOURCES: 8+ news sites + 3 government sources")
    print("⚡ METHOD: Parallel web scraping with rate limiting")
    print("🚀 EXPECTED TIME: 30-60 minutes")
    
    # Run collection
    df = scraper.run_parallel_collection(target_samples)
    
    if not df.empty:
        # Save data
        filename = scraper.save_collection(df)
        
        # Display summary
        print("\n📈 COLLECTION SUMMARY:")
        print(f"Total samples: {len(df):,}")
        print(f"Data saved to: {filename}")
        
        if 'category' in df.columns:
            print("\n📂 CATEGORY DISTRIBUTION:")
            print(df['category'].value_counts())
        
        if 'sentiment' in df.columns:
            print("\n💭 SENTIMENT DISTRIBUTION:")
            print(df['sentiment'].value_counts())
        
        if 'source' in df.columns:
            print(f"\n🌐 SOURCES COVERED: {df['source'].nunique()}")
        
        print(f"\n✅ SUCCESS: Collected {len(df):,} training samples!")
        
    else:
        print("❌ FAILED: No data collected. Check network and sources.")


if __name__ == "__main__":
    main()