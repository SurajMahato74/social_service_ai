"""
URL configuration for socialserviceai project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views
from . import sentiment_views

urlpatterns = [
    path('', views.home, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('public/', sentiment_views.public_sentiment_dashboard, name='public_sentiment'),
    path('gaps/', sentiment_views.gap_analysis_dashboard, name='gap_analysis'),
    path('live-gaps/', sentiment_views.live_gaps_dashboard, name='live_gaps'),
    path('api/status/', views.api_status, name='api_status'),
    path('api/sentiment/', sentiment_views.sentiment_api, name='sentiment_api'),
    path('api/gaps/', sentiment_views.gap_analysis_api, name='gap_analysis_api'),
    path('api/metadata/', views.api_metadata, name='api_metadata'),
    path('api/history/', views.api_history, name='api_history'),
    path('admin/', admin.site.urls),
    path('api-auth/', include('rest_framework.urls')),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    # Serve production files
    urlpatterns += static('/static/production_logs/', document_root=settings.BASE_DIR / 'production_logs')
    urlpatterns += static('/static/production_models/', document_root=settings.BASE_DIR / 'production_models')
