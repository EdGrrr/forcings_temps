import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import statsmodels.formula.api as smf
import statsmodels.api as sm


#File paths
forcing_file = 'Forcing_2500.csv'
enso_file = 'ENSO_Kaplan.txt'
temp_file = 'berkeley_HadSST_merged.txt'
#Options/Parameters
d1 = 4 #Fast decay time (years) of thermal response
d2 = 209.5 #Slow decay time two (years) of thermal response
c1 = 0.404 #Contribution to total equilibrium warming of fast response
c2 = 0.351 #Contribution to total equilibrium warming of slow response
vd1 = 1 #Fast decay time (years) of thermal response for volcanoes
vd2 = 209.5 #Slow decay time (years) of thermal response for volcanoes
vc1 = c1
vc2 = c2

def import_forcings(forcing_file, aero_scaling=1):
    '''
    Import forcing data into a pandas dataframe
    '''
    rf = pd.read_csv(forcing_file)
    rf['year'] = rf['date'].astype(int)
    rf['month'] = ((rf['date'] - rf['year']) * 12 + 1).round(0).astype(int)

    aero = rf['aero']
    rf['total'] += (aero_scaling-1)*aero
    rf['anthro'] += (aero_scaling-1)*aero
    rf['aero'] = aero_scaling*aero
    return rf

def remove_enso(temp_file, enso_file):
    '''
    Remove the influence of ENSO from observations using a 3-month lagged
    index per Foster and Rahmstorf. Return both temperatures with and without
    ENSO effects included.
    '''
    temps = pd.read_csv(temp_file, sep='\s+', skiprows=86, index_col=False,
                        usecols=[0,1,2,3], names=['year', 'month', 'anomaly', 'uncertainty'],
                        nrows=2148-96+9)

    temps['temp'] = temps['anomaly']
    enso = pd.read_csv(enso_file, sep='\s+', skiprows=46, index_col=False, names=['date', 'enso'])
    enso['year'] = enso['date'].astype(int)
    enso['month'] = ((enso['date'] - enso['year']) * 12 + 1).round(0).astype(int)
    temps = pd.merge(
        temps,
        enso,
        left_on=['year', 'month'],
        right_on=['year', 'month'],
        how='outer'
    )
    for n in range(1,10):
        temps['enso_'+str(n)] = temps['enso'].shift(n)
    temps_subset = temps.dropna(subset=['enso_3'])
    smresults = smf.ols('temp ~ enso_3', temps_subset).fit()
    temps['enso_3'].fillna(0, inplace=True)
    temps['pred'] = temps['enso_3'] * smresults.params.enso_3 + smresults.params.Intercept
    temps['temps_enso'] = temps['temp']
    temps['temps_no_enso'] = temps['temps_enso'] - temps['pred']
    temps['temps_no_enso'] = calc_anomaly(temps, 'temps_no_enso')
    temps['temps_enso'] = calc_anomaly(temps, 'temps_enso')
    return temps[['year', 'month', 'temps_enso', 'temps_no_enso']]

def calc_anomaly(temps, name):
    '''
    Calculate anomalies relative to the first 50 years of the temperature field provided.
    '''
    start = temps['year'][0]
    end = temps['year'][50]
    baseline = temps.loc[(temps['year'] >= start) & (temps['year'] <= end), name].mean()
    temps[name] = temps[name] - baseline
    return temps[name]

def temp_response(forc, d1, d2, c1, c2):
    '''
    Calculate the forcing responses from provided forcings.
    '''
    f1 = 1 - np.exp(-1./d1/12.)
    f2 = 1 - np.exp(-1./d2/12.)
    temp1 = [0]
    temp2 = [0]
    for i in range(1, len(forc.index)):
        temp1.append(temp1[i-1] + (forc.values[i-1] + forc.values[i]) * c1 * f1 / 2. - temp1[i-1] * f1)
        temp2.append(temp2[i-1] + (forc.values[i-1] + forc.values[i]) * c2 * f2 / 2. - temp2[i-1] * f2)
    return np.add(temp1, temp2)


def calc_anthro_natural_forcings(rf, temps, d1, d2, c1, c2, vd1, vd2, vc1, vc2):
    '''
    Calculate the total, natural only, and anthropogenic only temperature response
    based on a multivariate regression. Return the temperature timeseries and coefficients.
    '''
    forcings = ['total', 'anthro']
    for forcing in forcings:
        rf[forcing+'_warming'] = temp_response(rf[forcing], d1, d2, c1, c2)
    forcings = ['nat']
    for forcing in forcings:
        rf[forcing+'_warming'] = temp_response(rf[forcing], vd1, vd2, vc1, vc2)
    data = pd.merge(
    rf,
    temps,
    left_on=['year', 'month'],
    right_on=['year', 'month'],
    how='outer'
    )
    data_subset = data.dropna(subset=['total', 'temps_enso', 'temps_no_enso'])
    smresults = smf.ols('temps_enso ~ nat_warming + anthro_warming', data_subset).fit()
    intercept_anthro = smresults.params.Intercept
    #print(smresults.summary())
    anthro_lower = smresults.conf_int(alpha = 0.05)[0][2]
    anthro_upper = smresults.conf_int(alpha = 0.05)[1][2]
    #print anthro_lower, anthro_upper
    data['anthro_temp'] = (smresults.params.anthro_warming * data['anthro_warming'])

    smresults = smf.ols('temps_no_enso ~ nat_warming + anthro_warming', data_subset).fit()
    intercept_nat = smresults.params.Intercept
    nat_lower = smresults.conf_int(alpha = 0.05)[0][1]
    nat_upper = smresults.conf_int(alpha = 0.05)[1][1]
    #print nat_lower, nat_upper
    data['nat_temp'] = (smresults.params.nat_warming * data['nat_warming'])
    data['total_temp'] =  data['nat_temp'] + data['anthro_temp']
    data['temps_enso'] = data['temps_enso'] - intercept_anthro
    return {
        'results' : data,
        'anthro_coef' : smresults.params.anthro_warming,
        'nat_coef' : smresults.params.nat_warming,
        'anthro_lower': anthro_lower,
        'anthro_upper': anthro_upper,
        'nat_lower': nat_lower,
        'nat_upper': nat_upper
    }


def calc_other_forcings(data, anthro_coef, nat_coef, d1, d2, c1, c2, vd1, vd2, vc1, vc2, save_name):
    '''
    Use the natural only and anthropogenic only coefficients to calculate the temperature response
    of each different individual forcing series.
    '''
    forcings = ['total', 'anthro', 'co2', 'wmghg', 'o3Tr', 'o3St', 'luc', 'aero']
    for forcing in forcings:
        data[forcing+'_warming'] = temp_response(data[forcing], d1, d2, c1, c2)
    forcings = ['volc', 'solar', 'nat']
    for forcing in forcings:
        data[forcing+'_warming'] = temp_response(data[forcing], vd1, vd2, vc1, vc2)
    nat_forcings = ['volc', 'solar']
    for forcing in nat_forcings:
        data[forcing+'_temp'] = nat_coef * data[forcing+'_warming']
    anthro_forcings = ['co2', 'wmghg', 'o3Tr', 'o3St', 'luc', 'aero']
    for forcing in anthro_forcings:
        data[forcing+'_temp'] = anthro_coef * data[forcing+'_warming']
    data['ghg_temp'] = data['co2_temp'] + data['wmghg_temp']
    if save_name == 'global':
        data.to_csv('forcing_analysis.csv')
    if save_name == 'land':
        data.to_csv('forcing_analysis_land.csv')
    return data


def forcing_run(series, forcing_file, temp_file, enso_file, d1, d2, c1, c2, vd1, vd2, vc1, vc2, num):
    '''
    Calculate uncertainties in the total, natural only, and anthropogenic only temperature response
    using 200 different forcing series and the regression coefficient uncertainty for each via
    a Monte Carlo approach. Note: generates 4000 resulting series.
    '''
    temps = remove_enso(series, temp_file, enso_file)
    results = pd.DataFrame()
    o = 0
    for n in range(200):
        n_s = str(n + 1).zfill(3)
        print(n_s)
        rf = pd.read_csv('forcings/GWI_piers_forcing_mem'+n_s+'.csv', sep=' ')
        rf.columns = ['date', 'anthro', 'nat', 'total']
        rf['date'] = (rf['date'] - 0.042).round(2)
        rf['year'] = rf['date'].astype(int)
        rf['month'] = ((rf['date'] - rf['year']) * 12 + 1).round(0).astype(int)
        forcings = calc_anthro_natural_forcings(rf, temps, d1, d2, c1, c2, vd1, vd2, vc1, vc2)
        anthro_sigma = forcings['anthro_upper'] - forcings['anthro_coef']
        nat_sigma = forcings['nat_upper'] - forcings['nat_coef']
        df = forcings['results'][['month', 'year', 'nat_warming', 'anthro_warming']]
        anthro_coefs = sample_normal(forcings['anthro_coef'], anthro_sigma, num)
        nat_coefs = sample_normal(forcings['nat_coef'], nat_sigma, num)
        for m in range(num):
            res = df[['month', 'year']]
            res['anthro_temp'] = anthro_coefs[m] * df['anthro_warming']
            res['nat_temp'] = nat_coefs[m] * df['nat_warming']
            res['total_temp'] = res['anthro_temp'] + res['nat_temp']
            o += 1
            res.columns = ['month', 'year', 'anthro_temp_'+str(o), 'nat_temp_'+str(o), 'total_temp_'+str(o)]
            if (n+1) % 20 == 0 and (n+1) != 0 and m == 0:
                print(n_s)
                results = pd.merge(results, res, how='outer', right_on=['month', 'year'], left_on=['month', 'year'])
                results.to_csv('forcing_analysis_uncertainty_'+str(n+1)+'.csv')
                results = res[['month', 'year']]
                o = 0
            else:
                try:
                    results = pd.merge(results, res, how='outer', right_on=['month', 'year'], left_on=['month', 'year'])
                except:
                    results = res

def sample_normal(mean, stdev, num=10000):
    '''
    Sample a normal distribution and return a specified number of random values from
    the distribution.
    '''
    mu, sigma = mean, stdev # mean and standard deviation
    s = np.random.normal(mu, sigma, num)
    return s


#Code to run the various parts manually
save_name = None
temps = remove_enso(temp_file, enso_file)
rf = import_forcings(forcing_file)
df = calc_anthro_natural_forcings(rf, temps, d1, d2, c1, c2, vd1, vd2, vc1, vc2)
vals = calc_other_forcings(df['results'], df['anthro_coef'], df['nat_coef'], d1, d2, c1, c2, vd1, vd2, vc1, vc2, save_name)


plt.plot(vals['date'], vals['temps_enso'], c='lightgrey')
plt.plot(vals['date'], vals['total_temp'], c='k')
plt.plot(vals['date'], vals['ghg_temp'], c='r')
plt.plot(vals['date'], vals['aero_temp'], c='b')
plt.plot(vals['date'], vals['nat_temp'], c='g')
plt.plot([1850, 2022], [0, 0], lw=0.5, c='k')

plt.text(1857, 1.21, 'Observed temperature - Berkeley', c='grey')
plt.text(1985, 1.5, 'GHGs', c='r')
plt.text(1945, -0.5, 'Aerosols', c='b')
plt.text(1998, -0.3, 'Natural', c='g')
plt.text(1998, 0.3, 'Total', c='k')


plt.xlim(1850, 2022)
plt.xticks([1850, 1900, 1950, 2000, 2020])
plt.ylim(-0.8, 1.8)
plt.ylabel('Global mean\nsurface temperature change'+ r'($^{\circ}$C)')

fig = plt.gcf()
fig.set_size_inches(6, 3)
fig.savefig('temperature_by_forcer.png', bbox_inches='tight')
plt.show()




rf = import_forcings(forcing_file, aero_scaling=1)
df = calc_anthro_natural_forcings(rf, temps, d1, d2, c1, c2, vd1, vd2, vc1, vc2)
vals = calc_other_forcings(df['results'], df['anthro_coef'], df['nat_coef'], d1, d2, c1, c2, vd1, vd2, vc1, vc2, save_name)

plt.plot(vals['date'], vals['temps_enso'], c='lightgrey')
plt.plot(vals['date'], vals['total_temp'], c='k')

rf = import_forcings(forcing_file, aero_scaling=2)
df = calc_anthro_natural_forcings(rf, temps, d1, d2, c1, c2, vd1, vd2, vc1, vc2)
vals = calc_other_forcings(df['results'], df['anthro_coef'], df['nat_coef'], d1, d2, c1, c2, vd1, vd2, vc1, vc2, save_name)
plt.plot(vals['date'], vals['total_temp'], c='r')

rf = import_forcings(forcing_file, aero_scaling=0.2)
df = calc_anthro_natural_forcings(rf, temps, d1, d2, c1, c2, vd1, vd2, vc1, vc2)
vals = calc_other_forcings(df['results'], df['anthro_coef'], df['nat_coef'], d1, d2, c1, c2, vd1, vd2, vc1, vc2, save_name)
plt.plot(vals['date'], vals['total_temp'], c='b')

plt.plot([1850, 2100], [0, 0], lw=0.5, c='k')

plt.text(1857, 1.21, 'Observed temperature - Berkeley', c='grey')
plt.text(1985, 3.5, 'Strong aerosol cooling', c='red')
plt.text(2030, 0.5, 'Weak aerosol\ncooling', c='b')


plt.xlim(1850, 2100)
plt.xticks([1850, 1900, 1950, 2000, 2050, 2100])
plt.ylim(-0.8, 4.5)
plt.ylabel('Global mean\nsurface temperature change'+ r'($^{\circ}$C)')

fig = plt.gcf()
fig.set_size_inches(6, 3)
fig.savefig('aerosol_uncertainty_warming.png', bbox_inches='tight')
plt.show()

