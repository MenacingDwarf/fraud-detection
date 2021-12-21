from django.urls import path
from .views import *

urlpatterns = [
    path('', hello),
    path('customer/', Customer.as_view()),
    path('transaction/', Operation.as_view()),
    path('model/', MLModel.as_view())
]