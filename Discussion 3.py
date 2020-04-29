# Import libraries
import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics import tsaplots
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt


class ARMA:

    def __init__(self):
        # Download data
        self.__df = pd.read_csv('MyData.csv')
        self.__df.set_index('Month', inplace=True)
        self.__index_diff_values = None
        self.__in_sample_predicted_values = None

    def __check_stat(self):
        # Plot original data
        self.__df.plot(legend=False, title='Original Data')
        pyplot.show()

        # Apply Augmented Dickey-Fuller Test to original data
        adf_result_before_differencing = adfuller(self.__df['Value'])
        print 'Results of ADF test for original data:'
        print 'ADF Statistic: %f', adf_result_before_differencing[0]
        print 'p-value: %f', adf_result_before_differencing[1]
        if adf_result_before_differencing[1] >= 0.05:
            print 'Fail to reject the null hypothesis (H0) at 5 % level of \
            significance. The data has a unit root and is non-stationary'
        else:
            print 'Reject the null hypothesis (H0) at 5 % level of \
            significance. The data does not have a unit root and is stationary'

    def __make_stat(self):
        #  Differencing to make the series stationary
        self.__index_diff_values = self.__df['Value'] - \
            self.__df['Value'].shift()
        self.__index_diff_values.dropna(inplace=True)

        # Check stationarity
        # Plot differenced data
        self.__index_diff_values.plot(legend=False, title='Differenced data',
                                      color='green')
        pyplot.show()

        # Apply Augmented Dickey-Fuller Test after differencing
        adf_result_after_differencing = adfuller(self.__index_diff_values)
        print 'Results of ADF test after differencing:'
        print 'ADF Statistic: %f', adf_result_after_differencing[0]
        print 'p-value: %f', adf_result_after_differencing[1]
        if adf_result_after_differencing[1] >= 0.05:
            print 'Fail to reject the null hypothesis (H0) at 5 % level of \
            significance. The data has a unit root and is non-stationary'
        else:
            print 'Reject the null hypothesis (H0) at 5 % level of \
            significance. The data does not have a unit root and is stationary'

    def __plot_acf_pacf(self):
        # Plot ACF
        tsaplots.plot_acf(self.__index_diff_values, lags=50)
        pyplot.show()

        # Plot PACF
        tsaplots.plot_pacf(self.__index_diff_values, lags=50)
        pyplot.show()

    def __chekc_forcast(self):
        # Implement ARIMA(3,0,2)
        model = ARIMA(self.__index_diff_values.values, order=(1, 0, 1))
        model_fit = model.fit()
        print model_fit.summary()

        # In-sampele forecast
        self.__in_sample_predicted_values = model_fit.fittedvalues.cumsum() +\
            self.__df['Value'][0]

        self.__in_sample_predicted_values = [self.__df['Value'][0]] + list(
                self.__in_sample_predicted_values)

        self.__df['predicted'] = self.__in_sample_predicted_values

        self.__df.columns = ['actual', 'predicted']

        self.__df.plot()

    def get_differencies(self):
        return self.__index_diff_values

    def get_predicted(self):
        return self.__in_sample_predicted_values

    def get_actual(self):
        return self.__df

    def main(self):
        self.__check_stat()
        self.__make_stat()
        self.__plot_acf_pacf()
        self.__chekc_forcast()


arma = ARMA()
arma.main()

a = arma.get_actual()

b = arma.get_predicted()
