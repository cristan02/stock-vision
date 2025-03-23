from django.urls import path
from . import views

urlpatterns = [
    # Base LSTM Model
    path('tanh/', views.lstm_tanh_stock_prediction, name='lstm_tanh_stock_prediction'),

    # LSTM with ReLU Activation
    path('relu/', views.lstm_relu_stock_prediction, name='lstm_relu_stock_prediction'),

    # LSTM with Bidirectional Layers
    path('bidirectional/', views.lstm_bidirectional_stock_prediction, name='lstm_bidirectional_stock_prediction'),

    # LSTM with Convolutional Layers
    path('conv/', views.lstm_conv_stock_prediction, name='lstm_conv_stock_prediction'),
]