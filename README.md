# AIRush2

# Project Outline
2019 Naver AI hackathon - CTR Prediction

# Requirements
- Python 3.x
- Pytorch 1.0
- NSML 

# Model architecture

- The code is implemented based on baseline model (FCN)
- Binary Classification Model
- CNN + Embedding -> FCN 
- Apply Resnet 18 to extract the image features and target encoding method to embed the sequence data (user reading history)
- Large batch size to increase the training efficiency by using shallow FCN model structure 
- The final model is an ensemble model that uses hard voting method for N different models
