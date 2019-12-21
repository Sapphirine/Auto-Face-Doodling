# Auto Face Doodling

This project aims at exploring, recognizing and further automatically generate the patterns of facial doodles, including eye, nose, ear, mouth and face, focusing on the aspects of the word category and country based on the Quick, Draw! dataset from Google. 

# Dataset:
* https://github.com/googlecreativelab/quickdraw-dataset


# Structure of files:
```
Auto-Face-Doodling

├── analysis_modeling  
│   ├── Data_Prep_EDA.ipynb            # Data preprocessing and exploratory data analysis
│   ├── Categories.ipynb               # Word Category Classification
│   ├── Countries_CNN.ipynb            # Country Classification using CNN
│   ├── Countries_CNN+LSTM.ipynb       # Country Classification using CNN + LSTM
│   └── GAN.ipynb                      # Auto Doodling using GAN
├── demo
│   ├── data
│   ├── model
│   ├── country_model
│   ├── static
│   ├── template
│   ├── app.py
│   ├── model.json
│   └── model.h5                                               
└── Readme.md
```

