import os
from celery import Celery
from django.conf import settings

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'socialserviceai.settings')

app = Celery('social_service_ai')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

# Celery Beat Schedule for periodic tasks
app.conf.beat_schedule = {
    'scrape-social-media-daily': {
        'task': 'core.tasks.scrape_all_sources',
        'schedule': 21600.0,  # Every 6 hours (6 * 60 * 60)
    },
    'calculate-daily-ssei': {
        'task': 'core.tasks.calculate_daily_ssei_all_districts',
        'schedule': 3600.0,  # Every hour
    },
    'retrain-models-weekly': {
        'task': 'core.tasks.retrain_all_models',
        'schedule': 604800.0,  # Every week (7 * 24 * 60 * 60)
    },
    'cleanup-old-data': {
        'task': 'core.tasks.cleanup_old_data',
        'schedule': 86400.0,  # Daily cleanup
    },
}

app.conf.timezone = 'Asia/Kathmandu'

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')