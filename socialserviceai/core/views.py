from django.shortcuts import render
from django.http import JsonResponse
from django.db.models import Avg, Count, Q, Sum
from django.utils import timezone
from datetime import datetime, timedelta
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.views.generic import TemplateView

from .models import (
    District, ServiceCategory, SentimentData, ServiceLog,
    DailyScore, MLModelMetadata, ScrapingLog
)
from .serializers import (
    DistrictSerializer, ServiceCategorySerializer, SentimentDataSerializer,
    ServiceLogSerializer, DailyScoreSerializer, DashboardSummarySerializer,
    MLModelMetadataSerializer, ScrapingLogSerializer
)
from .tasks import calculate_district_ssei, scrape_all_sources


class DashboardView(TemplateView):
    """Main dashboard view"""
    template_name = 'dashboard/index.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        today = timezone.now().date()
        
        # Get latest SSEI scores
        latest_scores = DailyScore.objects.filter(date=today)
        
        context.update({
            'total_districts': District.objects.count(),
            'national_avg_ssei': latest_scores.aggregate(avg=Avg('ssei_score'))['avg'] or 0,
            'total_posts_today': SentimentData.objects.filter(scraped_at__date=today).count(),
            'active_services': ServiceLog.objects.filter(status='ongoing').count(),
        })
        
        return context


class DistrictViewSet(viewsets.ModelViewSet):
    """API ViewSet for Districts"""
    queryset = District.objects.all()
    serializer_class = DistrictSerializer
    permission_classes = [IsAuthenticated]
    
    @action(detail=True, methods=['get'])
    def ssei_history(self, request, pk=None):
        """Get SSEI history for a district"""
        district = self.get_object()
        days = int(request.query_params.get('days', 30))
        
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=days)
        
        scores = DailyScore.objects.filter(
            district=district,
            date__range=[start_date, end_date]
        ).order_by('date')
        
        data = [{
            'date': score.date.isoformat(),
            'ssei_score': score.ssei_score,
            'sentiment_score': score.sentiment_score,
            'service_delivery_score': score.service_delivery_score
        } for score in scores]
        
        return Response(data)
    
    @action(detail=True, methods=['post'])
    def recalculate_ssei(self, request, pk=None):
        """Trigger SSEI recalculation for a district"""
        district = self.get_object()
        date = request.data.get('date', timezone.now().date().isoformat())
        
        task = calculate_district_ssei.delay(district.id, date)
        
        return Response({
            'message': 'SSEI recalculation triggered',
            'task_id': task.id
        })


class SentimentDataViewSet(viewsets.ModelViewSet):
    """API ViewSet for Sentiment Data"""
    queryset = SentimentData.objects.all()
    serializer_class = SentimentDataSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        queryset = SentimentData.objects.all()
        
        # Filter by district
        district_id = self.request.query_params.get('district')
        if district_id:
            queryset = queryset.filter(district_id=district_id)
        
        # Filter by date range
        start_date = self.request.query_params.get('start_date')
        end_date = self.request.query_params.get('end_date')
        if start_date and end_date:
            queryset = queryset.filter(
                post_date__date__range=[start_date, end_date]
            )
        
        # Filter by sentiment
        sentiment_filter = self.request.query_params.get('sentiment')
        if sentiment_filter == 'positive':
            queryset = queryset.filter(is_positive=True)
        elif sentiment_filter == 'negative':
            queryset = queryset.filter(sentiment_score__lt=0)
        elif sentiment_filter == 'complaints':
            queryset = queryset.filter(is_complaint=True)
        
        return queryset.order_by('-post_date')
    
    @action(detail=False, methods=['get'])
    def sentiment_trends(self, request):
        """Get sentiment trends over time"""
        days = int(request.query_params.get('days', 7))
        district_id = request.query_params.get('district')
        
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=days)
        
        queryset = SentimentData.objects.filter(
            post_date__date__range=[start_date, end_date]
        )
        
        if district_id:
            queryset = queryset.filter(district_id=district_id)
        
        # Group by date and calculate average sentiment
        trends = []
        current_date = start_date
        
        while current_date <= end_date:
            day_data = queryset.filter(post_date__date=current_date)
            avg_sentiment = day_data.aggregate(avg=Avg('sentiment_score'))['avg'] or 0
            
            trends.append({
                'date': current_date.isoformat(),
                'avg_sentiment': round(avg_sentiment, 3),
                'total_posts': day_data.count(),
                'positive_posts': day_data.filter(is_positive=True).count(),
                'complaints': day_data.filter(is_complaint=True).count()
            })
            
            current_date += timedelta(days=1)
        
        return Response(trends)


class ServiceLogViewSet(viewsets.ModelViewSet):
    """API ViewSet for Service Logs"""
    queryset = ServiceLog.objects.all()
    serializer_class = ServiceLogSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        queryset = ServiceLog.objects.all()
        
        # Filter by district
        district_id = self.request.query_params.get('district')
        if district_id:
            queryset = queryset.filter(district_id=district_id)
        
        # Filter by status
        status_filter = self.request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        # Filter by service category
        category_id = self.request.query_params.get('category')
        if category_id:
            queryset = queryset.filter(service_category_id=category_id)
        
        return queryset.order_by('-start_date')
    
    @action(detail=False, methods=['get'])
    def efficiency_report(self, request):
        """Get service efficiency report"""
        district_id = request.query_params.get('district')
        
        queryset = ServiceLog.objects.all()
        if district_id:
            queryset = queryset.filter(district_id=district_id)
        
        # Calculate efficiency metrics
        services = queryset.filter(status__in=['completed', 'ongoing'])
        
        report = {
            'total_services': services.count(),
            'avg_efficiency': sum(s.efficiency_score for s in services) / services.count() if services else 0,
            'total_beneficiaries_planned': services.aggregate(sum=Sum('beneficiaries_planned'))['sum'] or 0,
            'total_beneficiaries_actual': services.aggregate(sum=Sum('beneficiaries_actual'))['sum'] or 0,
            'total_budget_allocated': float(services.aggregate(sum=Sum('budget_allocated'))['sum'] or 0),
            'total_budget_utilized': float(services.aggregate(sum=Sum('budget_utilized'))['sum'] or 0),
            'by_category': []
        }
        
        # Group by service category
        for category in ServiceCategory.objects.all():
            category_services = services.filter(service_category=category)
            if category_services.exists():
                report['by_category'].append({
                    'category': category.get_name_display(),
                    'total_services': category_services.count(),
                    'avg_efficiency': sum(s.efficiency_score for s in category_services) / category_services.count(),
                    'total_budget': float(category_services.aggregate(sum=Sum('budget_allocated'))['sum'] or 0)
                })
        
        return Response(report)


class DailyScoreViewSet(viewsets.ModelViewSet):
    """API ViewSet for Daily SSEI Scores"""
    queryset = DailyScore.objects.all()
    serializer_class = DailyScoreSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        queryset = DailyScore.objects.all()
        
        # Filter by date
        date_filter = self.request.query_params.get('date')
        if date_filter:
            queryset = queryset.filter(date=date_filter)
        
        # Filter by district
        district_id = self.request.query_params.get('district')
        if district_id:
            queryset = queryset.filter(district_id=district_id)
        
        return queryset.order_by('-date', 'district__name')
    
    @action(detail=False, methods=['get'])
    def national_summary(self, request):
        """Get national SSEI summary"""
        date_filter = request.query_params.get('date', timezone.now().date())
        
        scores = DailyScore.objects.filter(date=date_filter)
        
        if not scores.exists():
            return Response({'message': 'No data available for the specified date'})
        
        summary = {
            'date': date_filter,
            'total_districts': scores.count(),
            'national_avg_ssei': scores.aggregate(avg=Avg('ssei_score'))['avg'],
            'avg_sentiment': scores.aggregate(avg=Avg('sentiment_score'))['avg'],
            'avg_service_delivery': scores.aggregate(avg=Avg('service_delivery_score'))['avg'],
            'top_districts': DailyScoreSerializer(
                scores.order_by('-ssei_score')[:5], many=True
            ).data,
            'bottom_districts': DailyScoreSerializer(
                scores.order_by('ssei_score')[:5], many=True
            ).data,
            'province_breakdown': []
        }
        
        # Group by province
        provinces = scores.values('district__province').distinct()
        for province in provinces:
            province_name = province['district__province']
            province_scores = scores.filter(district__province=province_name)
            
            summary['province_breakdown'].append({
                'province': province_name,
                'districts_count': province_scores.count(),
                'avg_ssei': province_scores.aggregate(avg=Avg('ssei_score'))['avg']
            })
        
        return Response(summary)
    
    @action(detail=False, methods=['get'])
    def map_data(self, request):
        """Get data for Nepal map visualization"""
        date_filter = request.query_params.get('date', timezone.now().date())
        
        scores = DailyScore.objects.filter(date=date_filter).select_related('district')
        
        map_data = []
        for score in scores:
            if score.district.latitude and score.district.longitude:
                map_data.append({
                    'district_id': score.district.id,
                    'district_name': score.district.name,
                    'district_name_nepali': score.district.name_nepali,
                    'province': score.district.province,
                    'latitude': score.district.latitude,
                    'longitude': score.district.longitude,
                    'ssei_score': score.ssei_score,
                    'sentiment_score': score.sentiment_score,
                    'service_delivery_score': score.service_delivery_score,
                    'total_posts': score.total_posts_analyzed,
                    'total_services': score.total_services_tracked
                })
        
        return Response(map_data)


class DashboardAPIView(viewsets.ViewSet):
    """API endpoints for dashboard data"""
    permission_classes = [IsAuthenticated]
    
    @action(detail=False, methods=['get'])
    def summary(self, request):
        """Get dashboard summary data"""
        today = timezone.now().date()
        
        # Get today's scores
        today_scores = DailyScore.objects.filter(date=today)
        
        # Calculate summary metrics
        summary_data = {
            'national_avg_ssei': today_scores.aggregate(avg=Avg('ssei_score'))['avg'] or 0,
            'total_districts': District.objects.count(),
            'total_posts_today': SentimentData.objects.filter(scraped_at__date=today).count(),
            'total_services_active': ServiceLog.objects.filter(status='ongoing').count(),
            'top_performing_districts': DailyScoreSerializer(
                today_scores.order_by('-ssei_score')[:5], many=True
            ).data,
            'bottom_performing_districts': DailyScoreSerializer(
                today_scores.order_by('ssei_score')[:5], many=True
            ).data,
        }
        
        # Get sentiment trend for last 7 days
        sentiment_trend = []
        for i in range(7):
            date = today - timedelta(days=i)
            day_sentiment = SentimentData.objects.filter(
                post_date__date=date
            ).aggregate(avg=Avg('sentiment_score'))['avg'] or 0
            
            sentiment_trend.append({
                'date': date.isoformat(),
                'avg_sentiment': round(day_sentiment, 3)
            })
        
        summary_data['sentiment_trend'] = list(reversed(sentiment_trend))
        
        # Service category breakdown
        category_breakdown = {}
        for category in ServiceCategory.objects.all():
            active_services = ServiceLog.objects.filter(
                service_category=category,
                status='ongoing'
            ).count()
            category_breakdown[category.get_name_display()] = active_services
        
        summary_data['service_category_breakdown'] = category_breakdown
        
        return Response(summary_data)
    
    @action(detail=False, methods=['post'])
    def trigger_scraping(self, request):
        """Manually trigger social media scraping"""
        task = scrape_all_sources.delay()
        
        return Response({
            'message': 'Scraping task triggered successfully',
            'task_id': task.id
        })
    
    @action(detail=False, methods=['get'])
    def system_status(self, request):
        """Get system status and health metrics"""
        today = timezone.now().date()
        
        # Get latest scraping logs
        latest_scraping = ScrapingLog.objects.filter(
            start_time__date=today
        ).order_by('-start_time').first()
        
        # Get model metadata
        active_models = MLModelMetadata.objects.filter(is_active=True)
        
        status_data = {
            'last_scraping': {
                'time': latest_scraping.start_time if latest_scraping else None,
                'status': latest_scraping.status if latest_scraping else 'No data',
                'posts_scraped': latest_scraping.posts_scraped if latest_scraping else 0
            },
            'active_models': MLModelMetadataSerializer(active_models, many=True).data,
            'data_freshness': {
                'latest_sentiment_data': SentimentData.objects.order_by('-scraped_at').first().scraped_at if SentimentData.objects.exists() else None,
                'latest_ssei_calculation': DailyScore.objects.order_by('-calculation_timestamp').first().calculation_timestamp if DailyScore.objects.exists() else None
            },
            'database_stats': {
                'total_sentiment_records': SentimentData.objects.count(),
                'total_service_records': ServiceLog.objects.count(),
                'total_daily_scores': DailyScore.objects.count()
            }
        }
        
        return Response(status_data)


# Traditional Django views for templates
def nepal_map_view(request):
    """Nepal map visualization page"""
    return render(request, 'dashboard/nepal_map.html')

def sentiment_analysis_view(request):
    """Sentiment analysis dashboard page"""
    return render(request, 'dashboard/sentiment_analysis.html')

def service_tracking_view(request):
    """Service tracking dashboard page"""
    return render(request, 'dashboard/service_tracking.html')

def reports_view(request):
    """Reports and analytics page"""
    return render(request, 'dashboard/reports.html')
