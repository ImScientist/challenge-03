# Challenge travel data

## Goal

In this challenge, you are tasked to build a classifier to 
predict the conversion likelihood of a user based on previous 
search events, with emphasis on the feature engineering and 
evaluation part.

## Setup 

```shell script
conda create -n ta python=3.7.7 -y
conda activate ta
pip install -r requirements.txt
pip install -e . 
# python setup.py install 
```

## Training
- Create a file `.env` with the same environment variables as `.env.sample`:
    - `$GOOGLE_API_KEY`: needed only for the generation  of `data/iata_countries.csv`. 
    You can leave it empty.
    - `$DATA_DIR`: absolute path of the `data` folder in this repository
    - `$RESULTS_DIR`: absolute path of the directory where the results and model
    artifacts will be stored. If it does not exist it will be created during 
    model training.  

- Data preprocessing & features generation   
    This is the most time consuming step (20 min). We have applied the function to 
    the entire dataset and the pickled features are stored in `data/data.pkl`. 
    ```shell script
    source .env
    # use `--save_data` to save the generated features in `$DATA_DIR/data.pkl` 
    python exec/features_generation.py --nrows 200
    ```

- Model training    
    We have used the `LightGBM` library to train our model: 
    ```shell script
    source .env 
    python exec/train.py
    ```
    The model and a summary of the training is stored in  `$RESULTS_DIR`.
    The same training procedure can be executed within a Jupyter notebook, as
    shown [here](notebooks/Train.ipynb)

- Model calibration   
  Our task is not only to predict the class label, but also obtain a probability of the 
  respective label. We use the `sklearn.calibration` library to correct the probabilities 
  obtained from the model 
  [[source]](https://scikit-learn.org/stable/modules/calibration.html). 
  At the moment, this procedure can be executed only within a Jupyter notebook, as
  shown [here](notebooks/Calibrate.ipynb) 


## Discussion (feature generation)
  We have generated the following features for every user:
    Continuous variables:  
    - `date_from_month`: obtained from `date_from`     
    - `date_from_weekday`: obtained from `date_from`  
    - `date_to_month`: obtained from `date_to`   
    - `date_to_weekday`: obtained from `date_to`  
    - `ts_weekday`: obtained from `ts`     
    - `ts_hour`: obtained from `ts`  
    - `num_adults`: ..     
    - `num_children`: ..  
    - `n_bookings`: number of book events of a user before this event   
    - `n_od_pairs`: number of checked unique origin-destination paris after the last book event
    - `attempt_n`: number of attempts (webpage visits) after the last book event     
    - `distnace`: distance (km) between origin and destination  
    - `date_range`: trip duration = `date_to` - `date_from`    
    - `time_to_trip`: `date_from` - `ts`       
    - `dt_ts`: time difference btw two consecutive webpage visits   
    - `d_origin_dist`: distance btw origin locations of two consecutive webpage visits   
    - `d_destination_dist`: distance btw destination locations of two consecutive webpage visits
    
  Categorical variables:  
    - `origin`: `origin` encoded as integer     
    - `destination`: `destination` encoded as integer   
    - `od_pair`: an ordered `origin` - `destination` pair. This variable does not change
    if you swap the position of `origin` and `destination` in two consecutive events.  
    - `d_origin`: change of `origin` in two consecutive events (`0` - no change; `1`- change)   
    - `d_destination`: change of `destination` in two consecutive events (`0` - no change; `1`- change)
    - `d_od_pair`: change of `od_pair` in two consecutive events (`0` - no change; `1`- change)
    - `d_num_adults`: change of `num_adults` in two consecutive events (`0` - no change; `1`- change)
    - `d_num_children`: change of `num_children` in two consecutive events (`0` - no change; `1`- change) 

  Some of the unused features:
    - `countries`: The name of the country associated with the (lat, lon) pair of every 
    origin or destination. Because of the high cardinality of this variable we have not used it.


## Discussion (training)

  - We use a LightGBM
  
  - The data is chronologically split in train, validation and test dataset.  
    The validation is used for early stopping of the model training. The size 
    of the three datasets: 29783, 11696 4419


  - We have used the binary cross entropy as a loss function: 
    - The weights assigned to a booking event in all three sets 
      (train, validation and test) is `N/N_pos` (N - total number of elements; 
      N_pos number of booking events).
    - The weight assigned to a negative event is `1`.
       
    As a consequence, the model prefers to classify negative elements as positive 
    to doing the opposite, since in the latter case the penalty added to the loss 
    function is proportional to `N/N_pos`. It follows that the recall tends to 1
    in the training and validation data sets.

## Discussion (model calibration)
  - We face several problems:
    - From the high recall achieved during the model training we could assume that
      the model is willing to assign high scores to elements of the negative class.
      (look at the calibration curve for 0.9<p<=1; the fraction of positive is
      approximately 0.65)
    - The number of positive elements is very small
    - The original calibration curve is not monotonically increasing
  - With the use of an isotonic regression we managed only to obtain a 
    monotonically increasing calibration curve for the train and test data set.
  - Before starting calibrating the model, we have to improve the gradient boosting 
    model. The model should at least have a monotonically increasing calibration curve
    before calibrating it.
    

## Discussion (possible model updates)
  - Elimination of features:
    - use the feature importance obtained from the lgb model (for example, use the 
    feature importance derived from the information gain) 
    - use the permutation feature importance

  - Tune the model hyperparameters with GridsearchCV  
    - tree depth, max leaves, max elements in a leaf
    - loss function: reduce the weight assigned to the elements of the positive class
