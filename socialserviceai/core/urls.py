from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# API Router
router = DefaultRouter()
router.register(r'districts', views.DistrictViewSet)
router.register(r'sentiment-data', views.SentimentDataViewSet)
router.register(r'service-logs', views.ServiceLogViewSet)
router.register(r'daily-scores', views.DailyScoreViewSet)
router.register(r'dashboard', views.DashboardAPIView, basename='dashboard')

app_name = 'core'

urlpatterns = [
    # API URLs
    path('api/', include(router.urls)),
    
    # Dashboard Views
    path('', views.DashboardView.as_view(), name='dashboard'),
    path('nepal-map/', views.nepal_map_view, name='nepal_map'),
    path('sentiment-analysis/', views.sentiment_analysis_view, name='sentiment_analysis'),
    path('service-tracking/', views.service_tracking_view, name='service_tracking'),
    path('reports/', views.reports_view, name='reports'),
]