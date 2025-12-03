# Celery tasks for Social Service AI
# Note: Install celery and redis separately when ready

from django.utils import timezone
from datetime import datetime, timedelta
import logging
import os

try:
    from celery import shared_task
except ImportError:
    # Fallback decorator when celery is not installed
    def shared_task(func):
        return func

from .models import (
    District, SentimentData, ServiceLog, DailyScore, 
    MLModelMetadata, ScrapingLog, ServiceCategory
)

logger = logging.getLogger(__name__)

@shared_task
def scrape_all_sources():
    """Main task to scrape all social media sources"""
    logger.info("Starting daily social media scraping")
    
    # Facebook pages to scrape (Nepali government and NGO pages)
    facebook_pages = [
        'mohpgovnp',  # Ministry of Health and Population
        'swcnepal',   # Social Welfare Council
        'hamropatro', # Hamro Patro (popular Nepali platform)
        'kantipurdaily',
        'onlinekhabar',
        'setopati',
    ]
    
    total_scraped = 0
    
    for page in facebook_pages:
        try:
            scraped_count = scrape_facebook_page.delay(page)
            total_scraped += scraped_count if scraped_count else 0
        except Exception as e:
            logger.error(f"Error scraping {page}: {str(e)}")
    
    logger.info(f"Completed scraping. Total posts: {total_scraped}")
    return total_scraped

@shared_task
def scrape_facebook_page(page_name):
    """Scrape individual Facebook page"""
    from facebook_scraper import get_posts
    
    scraping_log = ScrapingLog.objects.create(
        source='facebook',
        url=f'https://facebook.com/{page_name}',
        start_time=timezone.now(),
        status='running'
    )
    
    try:
        posts_count = 0
        processed_count = 0
        
        # Scrape posts from last 7 days
        for post in get_posts(page_name, pages=5):
            try:
                # Check if post is recent (last 7 days)
                if post['time'] and (timezone.now().date() - post['time'].date()).days <= 7:
                    
                    # Extract Nepali keywords for service categorization
                    nepali_keywords = {
                        'health': ['स्वास्थ्य', 'अस्पताल', 'डाक्टर', 'औषधि', 'खोप'],
                        'education': ['शिक्षा', 'विद्यालय', 'पाठशाला', 'छात्रवृत्ति'],
                        'relief': ['राहत', 'सहायता', 'आपतकाल', 'बाढी', 'भूकम्प'],
                        'infrastructure': ['सडक', 'पुल', 'पानी', 'बिजुली', 'इन्टरनेट'],
                    }
                    
                    # Determine service category
                    service_category = None
                    text_content = post.get('text', '').lower()
                    
                    for category, keywords in nepali_keywords.items():
                        if any(keyword in text_content for keyword in keywords):
                            service_category = ServiceCategory.objects.filter(name=category).first()
                            break
                    
                    # Process with sentiment analysis
                    sentiment_task = analyze_sentiment.delay(
                        text_content=post.get('text', ''),
                        source='facebook',
                        source_url=post.get('post_url', ''),
                        post_date=post.get('time', timezone.now()),
                        service_category_id=service_category.id if service_category else None
                    )
                    
                    posts_count += 1
                    processed_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing post: {str(e)}")
                scraping_log.errors_count += 1
        
        # Update scraping log
        scraping_log.posts_scraped = posts_count
        scraping_log.posts_processed = processed_count
        scraping_log.end_time = timezone.now()
        scraping_log.status = 'completed'
        scraping_log.save()
        
        return posts_count
        
    except Exception as e:
        scraping_log.status = 'failed'
        scraping_log.error_details = str(e)
        scraping_log.end_time = timezone.now()
        scraping_log.save()
        logger.error(f"Facebook scraping failed for {page_name}: {str(e)}")
        return 0

@shared_task
def analyze_sentiment(text_content, source, source_url='', post_date=None, service_category_id=None, district_id=None):
    """Analyze sentiment of text using Nepali BERT model"""
    try:
        # Load sentiment model (placeholder - will be replaced with actual model)
        model_path = os.path.join(settings.ML_MODELS_PATH, 'sentiment_model.pkl')
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                sentiment_model = pickle.load(f)
            
            # Placeholder sentiment analysis (replace with actual BERT model)
            sentiment_score = 0.0  # Will be calculated by actual model
            confidence_score = 0.8
            
        else:
            # Fallback: simple keyword-based sentiment
            positive_keywords = ['राम्रो', 'सुधार', 'खुशी', 'सफल', 'उत्कृष्ट']
            negative_keywords = ['नराम्रो', 'समस्या', 'गुनासो', 'ढिलो', 'असफल']
            
            positive_count = sum(1 for word in positive_keywords if word in text_content)
            negative_count = sum(1 for word in negative_keywords if word in text_content)
            
            if positive_count > negative_count:
                sentiment_score = 0.5
            elif negative_count > positive_count:
                sentiment_score = -0.5
            else:
                sentiment_score = 0.0
                
            confidence_score = 0.6
        
        # Determine classifications
        is_positive = sentiment_score > 0.1
        is_complaint = sentiment_score < -0.3
        is_service_request = any(word in text_content.lower() for word in ['चाहिन्छ', 'आवश्यक', 'माग'])
        
        # Save sentiment data
        sentiment_data = SentimentData.objects.create(
            text_content=text_content,
            source=source,
            source_url=source_url,
            sentiment_score=sentiment_score,
            confidence_score=confidence_score,
            is_positive=is_positive,
            is_complaint=is_complaint,
            is_service_request=is_service_request,
            post_date=post_date or timezone.now(),
            processed_at=timezone.now(),
            service_category_id=service_category_id,
            district_id=district_id
        )
        
        logger.info(f"Sentiment analyzed: {sentiment_score:.2f} for text: {text_content[:50]}...")
        return sentiment_data.id
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        return None

@shared_task
def calculate_daily_ssei_all_districts():
    """Calculate SSEI for all districts for today"""
    today = timezone.now().date()
    districts = District.objects.all()
    
    for district in districts:
        calculate_district_ssei.delay(district.id, today.isoformat())
    
    logger.info(f"Triggered SSEI calculation for {districts.count()} districts")

@shared_task
def calculate_district_ssei(district_id, date_str):
    """Calculate SSEI for a specific district and date"""
    try:
        district = District.objects.get(id=district_id)
        date = datetime.fromisoformat(date_str).date()
        
        # Get or create daily score record
        daily_score, created = DailyScore.objects.get_or_create(
            district=district,
            date=date,
            defaults={
                'sentiment_score': 0,
                'service_delivery_score': 0,
                'demand_fulfillment_score': 0,
                'resource_efficiency_score': 0,
                'ssei_score': 0
            }
        )
        
        # Calculate sentiment score (0-100)
        sentiment_data = SentimentData.objects.filter(
            district=district,
            post_date__date=date
        )
        
        if sentiment_data.exists():
            from django.db.models import Avg
            avg_sentiment = sentiment_data.aggregate(
                avg_score=Avg('sentiment_score')
            )['avg_score']
            # Convert from (-1, 1) to (0, 100)
            daily_score.sentiment_score = (avg_sentiment + 1) * 50
            daily_score.total_posts_analyzed = sentiment_data.count()
        else:
            daily_score.sentiment_score = 50  # Neutral if no data
        
        # Calculate service delivery score
        services = ServiceLog.objects.filter(
            district=district,
            start_date__lte=date,
            end_date__gte=date
        )
        
        if services.exists():
            avg_efficiency = sum(service.efficiency_score for service in services) / services.count()
            daily_score.service_delivery_score = avg_efficiency
            daily_score.total_services_tracked = services.count()
        else:
            daily_score.service_delivery_score = 50  # Neutral if no data
        
        # Calculate demand fulfillment (placeholder)
        daily_score.demand_fulfillment_score = 60  # Will be calculated by forecasting model
        
        # Calculate resource efficiency (placeholder)
        daily_score.resource_efficiency_score = 65  # Will be calculated by optimization model
        
        # Calculate composite SSEI
        daily_score.calculate_ssei()
        
        # Apply bias mitigation (placeholder)
        daily_score.bias_adjustment = 0  # Will be calculated by AIF360 model
        
        daily_score.save()
        
        logger.info(f"SSEI calculated for {district.name} on {date}: {daily_score.ssei_score:.1f}")
        return daily_score.ssei_score
        
    except Exception as e:
        logger.error(f"SSEI calculation failed for district {district_id}: {str(e)}")
        return None

@shared_task
def retrain_all_models():
    """Retrain all ML models with new data"""
    logger.info("Starting weekly model retraining")
    
    models_to_retrain = [
        'sentiment',
        'forecasting', 
        'gap_detection',
        'bias_mitigation'
    ]
    
    for model_type in models_to_retrain:
        try:
            retrain_model.delay(model_type)
        except Exception as e:
            logger.error(f"Failed to trigger retraining for {model_type}: {str(e)}")

@shared_task
def retrain_model(model_type):
    """Retrain a specific ML model"""
    try:
        logger.info(f"Retraining {model_type} model")
        
        # This will be implemented with actual ML training code
        # For now, just update metadata
        
        new_version = f"v{timezone.now().strftime('%Y%m%d_%H%M')}"
        
        MLModelMetadata.objects.create(
            model_type=model_type,
            version=new_version,
            file_path=f"core/ml_models/{model_type}_model_{new_version}.pkl",
            training_date=timezone.now(),
            training_data_size=1000,  # Placeholder
            accuracy=0.85,  # Placeholder
            is_active=True
        )
        
        # Deactivate old models
        MLModelMetadata.objects.filter(
            model_type=model_type,
            is_active=True
        ).exclude(version=new_version).update(is_active=False)
        
        logger.info(f"Model {model_type} retrained successfully as {new_version}")
        
    except Exception as e:
        logger.error(f"Model retraining failed for {model_type}: {str(e)}")

@shared_task
def cleanup_old_data():
    """Clean up old data to maintain database performance"""
    try:
        # Delete sentiment data older than 6 months
        six_months_ago = timezone.now() - timedelta(days=180)
        old_sentiment_count = SentimentData.objects.filter(
            scraped_at__lt=six_months_ago
        ).count()
        
        SentimentData.objects.filter(scraped_at__lt=six_months_ago).delete()
        
        # Delete scraping logs older than 3 months
        three_months_ago = timezone.now() - timedelta(days=90)
        old_logs_count = ScrapingLog.objects.filter(
            start_time__lt=three_months_ago
        ).count()
        
        ScrapingLog.objects.filter(start_time__lt=three_months_ago).delete()
        
        logger.info(f"Cleanup completed: {old_sentiment_count} sentiment records, {old_logs_count} scraping logs deleted")
        
    except Exception as e:
        logger.error(f"Data cleanup failed: {str(e)}")

@shared_task
def send_daily_report():
    """Generate and send daily SSEI report"""
    try:
        today = timezone.now().date()
        
        # Get today's scores for all districts
        daily_scores = DailyScore.objects.filter(date=today).order_by('-ssei_score')
        
        if daily_scores.exists():
            from django.db.models import Avg
            avg_ssei = daily_scores.aggregate(avg_score=Avg('ssei_score'))['avg_score']
            
            report_data = {
                'date': today,
                'national_avg_ssei': avg_ssei,
                'top_districts': daily_scores[:5],
                'bottom_districts': daily_scores.reverse()[:5],
                'total_posts_analyzed': sum(score.total_posts_analyzed for score in daily_scores),
                'total_services_tracked': sum(score.total_services_tracked for score in daily_scores)
            }
            
            logger.info(f"Daily report generated: National SSEI = {avg_ssei:.1f}")
            return report_data
        
    except Exception as e:
        logger.error(f"Daily report generation failed: {str(e)}")
        return None