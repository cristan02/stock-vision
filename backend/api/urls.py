from django.urls import path
from . import views

urlpatterns = [
    # path('hello/', views.say_hello),
    path('predict-stock-rf/', views.random_forest_stock_prediction),
    path('predict-stock-svm/', views.svm_stock_prediction),
    path('predict-stock-lstm/', views.lstm_stock_prediction),
    path('portfolio-suggestion/', views.optimize_portfolio_api),
    path('validate-tickers/', views.validate_tickers),
]