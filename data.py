#!/usr/bin/env python
# coding: utf-8

# ## Download and prepare train, test, validation and forecast dataset

import requests
import pytz
from datetime import datetime, date, timedelta
import os
from glob import glob
from numpy import isnan, save
from pandas import read_csv, to_datetime, concat, merge
from matplotlib import pyplot
from numpy import array, split

#Function to get date range for train or forecast
def get_data_start_and_end_date( type='train' ):
    if type == 'train':
        #incase of development of new model
        start_date, end_date = '01-Jan-2019','25-Dec-2020'
    else:
        # need to set timezone incase we working outside from London time zone
        tz = pytz.timezone('Europe/London')
        # for forecast we only need current day data and api returns works on logic date >= start_date and end date < end_date
        start_date, end_date = datetime.now().astimezone(tz).date().strftime("%d-%b-%Y"), (datetime.now().astimezone(tz)+timedelta(days=1)).date().strftime("%d-%b-%Y")
    return start_date, end_date

def download_site_data( site_id, start_date, end_date, download_path = './data/train/' ):
    if not os.path.exists( download_path ):
        os.makedirs( download_path )
    # working url curl "http://www.londonair.org.uk/london/asp/downloadsite.asp?site=WM0&species1=COm&species2=NOm&species3=NOXm&species4=PM10m&species5=PM25m&species6=&start=31-Dec-2020&end=01-Jan-2021&res=6&period=15min&units=ugm3"
    data_url = 'http://www.londonair.org.uk/london/asp/downloadsite.asp?site=site_id&species1=COm&species2=NOm&species3=NOXm&species4=PM10m&species5=PM25m&species6=&start=start_date&end=end_date&res=6&period=15min&units=ugm3'
    data_url = data_url.replace( 'site_id', site_id )
    data_url = data_url.replace( 'start_date', start_date )
    data_url = data_url.replace( 'end_date', end_date )
    file_name = download_path+site_id+'_'+start_date+'.csv'
    try:
        r = requests.get( data_url, allow_redirects=True)
        open( file_name, 'wb' ).write(r.content)
    except:
        print('File download error.')
        return False
    print('Data successfuly downloaded for Site ID:{} for period:{},{}.'.format( site_id, start_date, end_date ))
    return True


# fill missing values with the data available at last timestamp
def fill_missing(df):
    for row in range( df.shape[0] ):
        if isnan(df.Value[row]):
            if row == 0:
                df.Value[row] = 0
            else:
                df.Value[row] = df.Value[row - 1]
    return df

# All data files contains data from 01-Jan-2020 to 24-Jan-2020 on 15 minutes interval.
# Different data files from different sites and each file contains site location in file name
def prepare_dataset( data_path, species=['CO', 'NO', 'PM10', 'PM2.5'] ):
    # All data files contains data from 01-Jan-2020 to 24-Jan-2020 on 15 minutes interval.
    # Different data files from different sites and each file contains site location in file name
    # Final dataframe
    final_df = None
    for file in glob(data_path+"*.csv"):
        df = read_csv(file)
        df = df.drop({'Units','Provisional or Ratified'}, axis=1)
        for specie in species:
            df_specie = df.loc[df['Species'] == specie]
            if not df_specie.empty:
                if df_specie.shape[0] > df_specie.Value.isnull().sum():
                    df_specie.reset_index(drop=True, inplace=True)
                    df_specie = fill_missing( df_specie )
                    df_specie = df_specie.drop({'Site','Species'}, axis=1 )
                    df_specie.rename({'Value': specie}, inplace=True, axis=1)
                    df_specie['ReadingDateTime'] = to_datetime(df_specie.ReadingDateTime)
                    df_specie.set_index('ReadingDateTime',inplace=True)
                    df_specie = df_specie.resample( '30Min' )
                    df_specie = df_specie.mean()
                    if final_df is None:
                        final_df = df_specie
                    elif specie in final_df.columns:
                        final_df = concat([final_df, df_specie])
                        final_df = final_df.resample( '30Min' )
                        final_df = final_df.mean()
                    else:
        #                 print(specie)
                        final_df = merge(final_df, df_specie, on='ReadingDateTime')
    return final_df

# split a multivariate dataset into train/test sets
def split_train_test_dataset(data, lag=2):
    # split into 11 month, 23 days and 1 day for train, test and validation set correspondingly
#     train, test, validate = data[:-1152], data[-1152:-48], data[-48:]
    train, test, validate = data[:-6940], data[-6940:-48], data[-48:]
#     print( "Train shape: {}, Test shape: {} and Validate shape: {}".format( train.shape, test.shape, validate.shape ) )
    # restructure into windows of data
    train = array(split(train, len(train)/lag))
    test = array(split(test, len(test)/lag))
    validate = array(split(validate, len(validate)/lag))
    print( "Train shape: {}, Test shape: {} and Validate shape: {}".format( train.shape, test.shape, validate.shape ) )
    return train, test, validate

# prepare multivariate dataset for next 30 minutes forecast
def split_forecast_dataset(data, lag=2):
    # Read last 1 hour data
    forecast_data = data[-2:]
    # restructure into windows of data
    forecast_data = array(split(forecast_data, len(forecast_data)/lag))
    print( "Forecast dataset shape: {}".format( forecast_data.shape ) )
    return forecast_data


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
    # flatten data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end, :])
            # move along one time step
            in_start += 1
    return array(X), array(y)

# Clean files from directory
def clean_directory( path ):
    files = glob( path+'/*.csv' )
    for file in files:
        os.remove( file )
