from .views import download,version
from django.urls import path
urlpatterns=[
path('version/', version, name='version'),
path('download/', download, name='download'),]