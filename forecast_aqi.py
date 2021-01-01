#!/usr/bin/env python
# coding: utf-8

# ## Make next 30 minutes CO, NO, PM10 and PM2.5 forecast for Central London

import data
import argparse
import numpy as np
from datetime import datetime, timedelta
from keras.models import load_model


# ## Download current day data and prepare dataset

class AQI_Forecast():
    def __init__( self ):
        # All sites list for Central London
        self.site_ids = ['BL0', 'CT3', 'HG1', 'IS2', 'KC1', 'LB4', 'SK8', 'TH4', 'WA9', 'WM0']
        self.n_input = 2
        self.verbose = 0
    def prepare_forecast_data( self, data_path='./data/forecast/' ):
        # Download and store all sites data
        start_date, end_date = data.get_data_start_and_end_date('forecast')
        #Clean the old files from firectory
        data.clean_directory( data_path )
        #Download for training model
        for site in self.site_ids:
            data.download_site_data( site, start_date, end_date, data_path )
        forecast_df = data.prepare_dataset( data_path )
#         print(forecast_df.head())
        # FORECAST DATASET FIELDS NEED TO REARRANGE
        forecast_df = forecast_df[['NO', 'PM10', 'PM2.5', 'CO']]
#         print(forecast_df.head())
#         print( forecast_df.shape )
        forecast_period = (datetime.strptime(np.datetime_as_string(forecast_df.index[-1:].values[0],unit='s'),
                                             '%Y-%m-%dT%H:%M:%S')+timedelta(minutes=30)).strftime("%d-%b-%Y %H:%M:%S")
#         print('Forecast period: {}'.format(forecast_period))
        forecast_dataset = data.split_forecast_dataset( forecast_df.values )
        return forecast_period, forecast_dataset
    
    def forecast( self, forecast_dataset, model='./models/cnn_lstm.h5' ):
        model = load_model( model )
        forecast = model.predict( forecast_dataset, verbose=self.verbose )
        return forecast

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/forecast/', help='Enter data path')
    parser.add_argument('--model', type=str, default='./models/cnn_lstm.h5', help='Enter model path')
    arg = parser.parse_args()
    aqi_forecast = AQI_Forecast()
    forecast_period, forecast_dataset = aqi_forecast.prepare_forecast_data( arg.data )
#     forecast_period, forecast_dataset = aqi_forecast.prepare_forecast_data( )
#     print(forecast_dataset)
    forecast = aqi_forecast.forecast( np.array(forecast_dataset), model=arg.model )
#     forecast = aqi_forecast.forecast( np.array(forecast_dataset) )
    print( 'CO:{}, NO:{}, PM10:{} and PM2.5:{} forecast for next 30 minutes: {}'.format( forecast[:,3], forecast[:,0], forecast[:,1], forecast[:,2], forecast_period ) )
