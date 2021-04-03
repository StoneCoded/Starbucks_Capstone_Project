# Udacity/Starbucks_Capstone_Project

## Starbucks

Final project for Udacity's Data Scientist Nanodegree using data supplied by
Starbucks that was originally a take home assignment for prospective Starbucks
employees.

## Contents
1. Files and Implementation
2. Project Overview
3. Results and Observations
4. Licensing, Authors, and Acknowledgements

## Files and Implementation

###Files
The Two main files are offer_predictor.py that contains the final product but to
help visualise things I created starbucks_offer_predictor.ipynb that directly
inherits everything from the class itself.

starbucks_offer_predictor.ipynb
  - Main Jupyter Notebook for using Predictor Class
  offer_predictor.py
    - Offer Predictor Class Object

clean_data.py
  - Data Cleaning
process_data.py
  - Suplimentary functions for Data Cleaning
offer_pred_functions.py
  - Functions for Offer Prediction Class
offer_pred_plots.py
  - Plot/Graph functions for Prediction Class

redundant_functions.py
  - Functions no longer needed but may come into the fold with later work.

In order to run the Jupyter notebook you will need to have python 3 installed and the following libraries:

- [pandas](https://pandas.pydata.org/)
- [Sklearn](https://scikit-learn.org/stable/)
- [Numpy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

## Project Motivation

Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). I wanted to determine which demographic groups respond best to which offer type. To challenge myself further, I wanted to predict how successful offers would be for a given person based solely on their demographic.


## File Descriptions

The Jupyter Notebook showcases the object class that builds 11 Success and 11 Transaction Amount models. These
predict the best offer response from someone if only given demographic data.

The data given to us was stored in 3 json files:

- `portfolio.json` - containing offer ids and data regarding the offer: duration, difficulty and type.
- `Profile.json`  data containing demographic information for each customer.
- `transcript.json` - data containing every record for transaction, offers received, offers reviewed and offers completed.

Here are the schemas for each file:

`portfolio.json`
id (string) - offer id
offer_type (string) - type of offer ie BOGO, discount, informational
difficulty (int) - minimum required spend to complete an offer
reward (int) - reward given for completing an offer
duration (int) - time for offer to be open, in days
channels (list of strings)

`profile.json`
age (int) - age of the customer
became_member_on (int) - date when customer created an app account
gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
id (str) - customer id
income (float) - customer's income

`transcript.json`
event (str) - record description (ie transaction, offer received, offer viewed, etc.)
person (str) - customer id
time (int) - time in hours since start of test. The data begins at time t=0
value - (dict of strings) - either an offer id or transaction amount depending on the record

## Results

The main findings of this project can be found in my medium post about the project [here](https://medium.com/@ben-stone/starbucks-offer-prediction-f16a75f64024).

I used Logistic Regression and Linear Regression to predict whether an offer would be successful and how much in theory a person would spend.

## 4.Licensing, Authors, and Acknowledgements
All data was provided by Udacity and Starbuck for the purpose of this project.

Udacity:[Here](https://www.Udacity.com)
Starbucks:[Here](https://www.Starbucks.com)

Copyright 2021 Ben Stone

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
