# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 18:19:32 2022

@author: Gwenael Münker
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import QuantLib as ql
import datetime
from scipy.stats import norm
import researchpy as rp
from matplotlib.gridspec import GridSpec

np.random.seed(1)
 
#############################################################################################
######################################  Input  ##############################################
#############################################################################################
Zeitintervall = 'm'

MSCI_input_loc = r'F:\Wintersemester 21 22\Bachelorarbeit\Daten\MSCI\MSCI.xlsx'
EONIA_input_loc = r'F:\Wintersemester 21 22\Bachelorarbeit\Daten\EONIA\EONIA.xlsx'

MSCI_World = pd.read_excel(MSCI_input_loc)
EONIA = pd.read_excel(EONIA_input_loc)

#############################################################################################
###################################  Erstelle TimeSeries  ###################################
#############################################################################################

def Create_TimeSeries():
    #Erstelle eine TimerSeries aus MSCI World und EONIA Zeitreihen

    EONIA['EONIA']              = EONIA['EONIA'] / 100 #Eonia in Prozent umwandeln
    MSCI_World['MSCI_World']    = MSCI_World['Value']  
    #Wir benutzen die beobachteten Tage des MSCI World
    TimeSeries                  = pd.merge(MSCI_World[['Date','MSCI_World']],EONIA,on='Date',how='left')
        
    return(TimeSeries)

#Ausführung Funktion zur Erstellung der TimeSeries
TimeSeries = Create_TimeSeries()

#############################################################################################
###################################  Transformation  ########################################
#############################################################################################


def EONIA_Conversion():
    #Die EONIA Rate ist annualisiert, um sie mit unseren Renditen vergleichbar zumachen, müssen wir EONIA wieder auf
    #tägliche bzw. monatliche Werte umwandeln (EONIA wird auf Basis von ACT/360 berechet)
    
    TimeSeries['Jahr']              = pd.DatetimeIndex(TimeSeries['Date']).year
    DaysPerYear                     = []
    
    for date in TimeSeries['Jahr']: #Berechnung der Anzahl der Tage in entsprechenden Jahr
        ActThreeSixty = ql.Actual360().yearFraction(ql.Date(1,1,date),ql.Date(1,1,date+1))
        ActThreeSixty = ActThreeSixty*360
        DaysPerYear.append(ActThreeSixty)
        
    TimeSeries['Dif']               = DaysPerYear
    i                               = 0
            
    #Transformation EONIA in monatliche Werte
    for row in TimeSeries['EONIA']:
        TimeSeries['EONIA'][0] = 0
        TimeSeries.loc[i,'EONIA'] = pow(TimeSeries.loc[i,'EONIA']+1,1/(TimeSeries.loc[i,'Dif']/30))-1 #jährlich -> monatlich
        i = i + 1   
               
    TimeSeries.drop(columns=['Jahr','Dif'],inplace = True)
    TimeSeries.set_index('Date',inplace=True)
        
    TimeSeries['MSCI_World'] = TimeSeries['MSCI_World'].pct_change()
        
    return (TimeSeries)

#Ausführung Transformationen
EONIA_Conversion() 
TimeSeries = TimeSeries[1:]

#############################################################################################
######################################  CPPI & TIPP  ########################################
#############################################################################################

def CPPI(risky_r, safe_r, m=2, start=1000000, floor=0.9, TIPP=False): 

    dates = risky_r.index
    n_schritte = len(dates)
    S = MSCI_World['Value'][1+(276-n_schritte):] ####Für Backtest
    #S= 439.249*(1+risky_r).cumprod()       ####Für MC
    I = start
    floor_wert = start*floor
    peak = floor_wert

    # erstelle DFs
    V_C = pd.DataFrame().reindex_like(risky_r)
    E_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    Floor_history = pd.DataFrame().reindex_like(risky_r)
    Rf_history = pd.DataFrame().reindex_like(risky_r)
    E_w_history = pd.DataFrame().reindex_like(risky_r)
    Rf_w_history = pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_schritte):
        if TIPP == True:
            floor_prime = I*floor
            floor_wert = np.maximum(floor_prime,floor_wert)
        cushion = (I - floor_wert)
        E = m*cushion
        E = np.minimum(E,I)
        E = np.maximum(E,0)
        n_e = E//S.iloc[step]
        E = n_e*S.iloc[step]
        Rf = I - E
        E_w = E/I
        Rf_w = 1 - E_w
        I = E*(1+risky_r.iloc[step]) + Rf*(1+safe_r.iloc[step])
        cushion_history.iloc[step] = cushion
        E_history.iloc[step] = E
        V_C.iloc[step] = I
        Floor_history.iloc[step] = floor_wert
        Rf_history.iloc[step] = Rf
        Rf_w_history.iloc[step] = Rf_w
        E_w_history.iloc[step] = E_w
    result = {
        "V_C": V_C, 
        "Cushion": cushion_history,
        "E": E_history,
        "MSCI_World_r":risky_r,
        "EONIA_r": safe_r,
        "Floor": Floor_history,
        "S": S,
        "Rf":Rf_history,
        "E_w":E_w_history,
        "Rf_w":Rf_w_history
    }
    return result

def CPPI_TIPP(TimeSeries, m=2,floor=0.9,start=1000000):  ######Damit in der Auswertung CPPI und TIPP nicht einzeln ausgeführt werden müssen
    
    start = start
    m = m
    floor = floor
    risky_r = TimeSeries[['MSCI_World']]
    safe_r = TimeSeries[['EONIA']]
    risky_r[0] = risky_r['MSCI_World']
    safe_r[0] = safe_r['EONIA']
    risky_r = risky_r.reset_index().drop(["Date",'MSCI_World'], axis=1)
    safe_r = safe_r.reset_index().drop(["Date",'EONIA'], axis=1)

    
    Backtest_CPPI = CPPI(risky_r = risky_r,safe_r=safe_r,TIPP=False,m=m,floor=floor,start=start)
    Backtest_TIPP = CPPI(risky_r = risky_r,safe_r=safe_r,TIPP=True,m=m,floor=floor,start=start)
    
    return (Backtest_CPPI,Backtest_TIPP)

#############################################################################################
#######################################   Simulation ########################################
#############################################################################################

def MC_Simulation(asset):

    asset = asset.astype(float)
    #Parameter der diskreten Rendite
    mu_asset = np.mean(asset)
    sigma_asset = np.std(asset, ddof=1)
    #tramsformiere in LOG-Normalverteilte ZV
    v = np.sqrt(np.log(1+(pow(sigma_asset,2)/pow(1+mu_asset,2))))
    m = np.log(1+mu_asset) - 0.5*pow(v,2)

    n = 200000 #Anzahl Pfade

    MC = list(range(1,n+1)) #DataFrame in Abh. von Anzahl der Pfade
    MC = pd.DataFrame(MC)       
    
    MC_Zeit = 24 #Zeitspanne in Monaten
    
    for i in range(MC_Zeit): #generiere Standardnormalverteilte Zufallsvariablen
        MC[i] = np.random.normal(0, 1, n)
    
    MC = np.exp(m + v*MC) #Errechne lognormalverteilte Renditen aus Zufallszahlen

    MC_cum = MC.transpose()
    MC_cum = MC_cum-1
    
    return (MC_cum)

MC_R_MSCI_World = MC_Simulation(TimeSeries['MSCI_World']) ####Simulierte Renditen MSCI World
MC_EONIA = MC_Simulation(TimeSeries["EONIA"]) #Simulierte EONIA Werte

asset = TimeSeries['MSCI_World'].astype(float)
mu_asset = np.mean(asset)
sigma_asset = np.std(asset, ddof=1)
v = np.sqrt(np.log(1+(pow(sigma_asset,2)/pow(1+mu_asset,2))))
m = np.log(1+mu_asset) - 0.5*pow(v,2)

#############################################################################################
#####################################  BS Funktionen ########################################
#############################################################################################

N = norm.cdf

def BS_CALL(S, X, T, r, sigma): #BS Formel für europäische Call Option
    d1 = (np.log(S/X) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - X * np.exp(-r*T)* N(d2)

def BS_PUT(S, X, T, r, sigma): #BS Formel für europäische Put Option
    d1 = (np.log(S/X) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return X*np.exp(-r*T)*N(-d2) - S*N(-d1)

#############################################################################################
###################################  Covered Short Call #####################################
#############################################################################################

sigma_asset = np.std(np.log(1+TimeSeries['MSCI_World']),ddof=1)*np.sqrt(12) ###Für MC
log_rf = np.log(1+np.mean(TimeSeries['EONIA'][1:]))*12 ####Für MC


def CVSC(risky_r,safe_r,I=1000000,A=8000,K=1):
        
    dates = risky_r.index
    n_schritte = len(dates)
    SP= (I-A)/(len(risky_r)-1)
    #S = 439.249*(1+risky_r).cumprod() ###Für MC
    S = MSCI_World['Value'][1+(276-n_schritte):] ####Für Backtest
    #P_start = 439.249 ###Für MC
    P_start = MSCI_World['Value'].iloc[276-n_schritte]###Für Backtest
    B_Prime_history = pd.DataFrame().reindex_like(risky_r)
    V_C = pd.DataFrame().reindex_like(risky_r)
    n_history = pd.DataFrame().reindex_like(risky_r)
    Call_Price_history = pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_schritte):
        sigma_asset = np.std(np.log(1+TimeSeries['MSCI_World'][:12+step]),ddof=1)*np.sqrt(12) ##Für Backtest
        log_rf = np.log(1+np.mean(TimeSeries['EONIA'][:12+step]))*12 ###Backtest
        if step== 0:
            Call_Price = BS_CALL(P_start,K*P_start,1/12,log_rf,sigma_asset)
            n = A//(P_start - Call_Price)
            B_Prime = A - n*(P_start - Call_Price)
            I = n*(np.minimum(S.iloc[step],K*P_start)) + B_Prime*(1+safe_r.iloc[step])
        
        else:
            Call_Price = BS_CALL(S.iloc[step-1],K*S.iloc[step-1],1/12,log_rf,sigma_asset)
            n = (I+SP)//(S.iloc[step-1] - Call_Price)
            B_Prime = (I+SP) - n*(S.iloc[step-1] - Call_Price)
            I = n*(np.minimum(S.iloc[step],K*S.iloc[step-1])) + B_Prime*(1+safe_r.iloc[step])
        
        B_Prime_history.iloc[step] = B_Prime
        V_C.iloc[step] = I+SP*(len(risky_r)-1-step)
        n_history.iloc[step] = n
        Call_Price_history.iloc[step] = Call_Price
    
    result = {
        "V_C": V_C, 
        "B_Prime": B_Prime_history,
        "n": n_history,
        "Call_Price": Call_Price_history,
        }
    return result      

def PP(risky_r,safe_r,I=1000000,A=8000,K=1): 
    
    dates = risky_r.index
    n_schritte = len(dates)
    SP= (I-A)/(len(risky_r)-1)
    #S = 439.249*(1+risky_r).cumprod() ###Für MC
    S = MSCI_World['Value'][1+(276-n_schritte):] ####Für Backtest
    #P_start = 439.249 ###Für MC
    P_start = MSCI_World['Value'].iloc[276-n_schritte]###Für Backtest
    B_Prime_history = pd.DataFrame().reindex_like(risky_r)
    V_C = pd.DataFrame().reindex_like(risky_r)
    n_history = pd.DataFrame().reindex_like(risky_r)
    Put_Price_history = pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_schritte):
        sigma_asset = np.std(np.log(1+TimeSeries['MSCI_World'][:12+step]),ddof=1)*np.sqrt(12) ##Für Backtest
        log_rf = np.log(1+np.mean(TimeSeries['EONIA'][:12+step]))*12 ### Für Backtest
        if step== 0:
            Put_Price = BS_PUT(P_start,K*P_start,1/12,log_rf,sigma_asset)
            n = A//(P_start + Put_Price)
            B_Prime = A - n*(P_start + Put_Price)
            I = n*(np.maximum(S.iloc[step],K*P_start)) + B_Prime*(1+safe_r.iloc[step])
        
        else:
            Put_Price = BS_PUT(S.iloc[step-1],K*S.iloc[step-1],1/12,log_rf,sigma_asset)
            n = (I+SP)//(S.iloc[step-1] + Put_Price)
            B_Prime = (I+SP) - n*(S.iloc[step-1] + Put_Price)
            I = n*(np.maximum(S.iloc[step],K*S.iloc[step-1])) + B_Prime*(1+safe_r.iloc[step])
        
        B_Prime_history.iloc[step] = B_Prime
        V_C.iloc[step] = I + SP*(len(risky_r)-1-step)
        n_history.iloc[step] = n
        Put_Price_history.iloc[step] = Put_Price
    
    result = {
        "V_C": V_C, 
        "B_Prime": B_Prime_history,
        "n": n_history,
        "Put_Price": Put_Price_history,
        }
    return result       

#############################################################################################
################################  Backtest Auswertung #######################################
#############################################################################################

####### allgemeines Basisszenario
CPPI_Backtest_m3_f8_00, TIPP_Backtest_m3_f8_00 = CPPI_TIPP(TimeSeries=TimeSeries[:][12:], m=3, floor=0.8)

risky_r = TimeSeries[['MSCI_World']][12:]
safe_r = TimeSeries[['EONIA']][12:]
risky_r[0] = risky_r['MSCI_World']
safe_r[0] = safe_r['EONIA']
risky_r = risky_r.reset_index().drop(["Date",'MSCI_World'], axis=1)
safe_r = safe_r.reset_index().drop(["Date",'EONIA'], axis=1)
PP_Backtest = PP(risky_r=pd.DataFrame(risky_r),safe_r=pd.DataFrame(safe_r)) 
CVSC_Backtest = CVSC(risky_r=pd.DataFrame(risky_r),safe_r=pd.DataFrame(safe_r)) 

Auswertung = pd.concat([PP_Backtest['V_C'],CVSC_Backtest['V_C'],CPPI_Backtest_m3_f8_00['V_C'],TIPP_Backtest_m3_f8_00['V_C'],(1000000)*(1+CPPI_Backtest_m3_f8_00['MSCI_World_r']).cumprod(),CPPI_Backtest_m3_f8_00['Floor'],TIPP_Backtest_m3_f8_00['V_C']],axis=1)
Auswertung.columns = ['PP','CVSC','CPPI','TIPP','MSCI_World','CPPI Floor','TIPP Floor']
Auswertung['Date'] = TimeSeries[12:].index
Auswertung.set_index('Date',inplace=True)
Auswertung = Auswertung/1000000

ax = (Auswertung).plot(figsize=(24, 12),legend=False)
ax.legend(['PP','CVSC','CPPI','TIPP','MSCI World','CPPI Floor','TIPP Floor'],loc='upper left',prop={'size': 18})
ax.set_ylabel('Vermögem in M€',fontsize=24)
ax.set_xlabel('',fontsize=1)
ax.tick_params(axis='both', which='major', labelsize=14)

Auswertung.drop(columns=['CPPI Floor','TIPP Floor'],inplace = True)


previous_peaks = Auswertung.cummax()
drawdowns = (Auswertung - previous_peaks)/previous_peaks
dd = drawdowns.min()
returns = Auswertung.pct_change()
returns = returns[1:]
std=returns.std(ddof=1)
std_an = std*np.sqrt(12)
total_return = (1+returns).prod()-1
total_return = total_return.transpose()
monthly_return = pow(1+total_return,1/263)-1
monthly_rf = pow((1+TimeSeries['EONIA'][13:]).prod(),1/263)-1
annualized_return = pow(1+monthly_return,12)-1
annualized_rf = pow(1+monthly_rf,12)-1
Sharpe = (monthly_return-monthly_rf)/std
Sharpe_annu = (annualized_return-annualized_rf)/std_an

summary_S= pd.concat([total_return, monthly_return, std,Sharpe,annualized_return,std_an,Sharpe_annu,dd],axis=1)
summary_S.columns=['Gesamtrendite','Rendite_m','Std_m','Sharpe_m','Rendite_j','Std_j','Sharpe_j','Maximum Drawdown']

#########Variations Basispreis PP und CVSC
PP_Backtest_0_95 = PP(risky_r=pd.DataFrame(risky_r),safe_r=pd.DataFrame(safe_r),K=0.95) 
CVSC_Backtest_0_99 = CVSC(risky_r=pd.DataFrame(risky_r),safe_r=pd.DataFrame(safe_r),K=0.99) 
PP_Backtest_1_01 = PP(risky_r=pd.DataFrame(risky_r),safe_r=pd.DataFrame(safe_r),K=1.01) 
CVSC_Backtest_1_05 = CVSC(risky_r=pd.DataFrame(risky_r),safe_r=pd.DataFrame(safe_r),K=1.05) 
PP_Backtest_1_0 = PP(risky_r=pd.DataFrame(risky_r),safe_r=pd.DataFrame(safe_r)) 
CVSC_Backtest_1_0 = CVSC(risky_r=pd.DataFrame(risky_r),safe_r=pd.DataFrame(safe_r)) 

PP_Ausübungspreis_K = pd.concat([PP_Backtest_0_95['V_C'],PP_Backtest_1_0['V_C'],PP_Backtest_1_01['V_C'],(1000000)*(1+CPPI_Backtest_m3_f8_00['MSCI_World_r']).cumprod()],axis=1)
PP_Ausübungspreis_K.columns = ['PP 0.98','PP 1.0','PP 1.02','MSCI World']
PP_Ausübungspreis_K['Date'] = TimeSeries[12:].index
PP_Ausübungspreis_K.set_index('Date',inplace=True)

CVSC_Ausübungspreis_K = pd.concat([CVSC_Backtest_0_99['V_C'],CVSC_Backtest_1_0['V_C'],CVSC_Backtest_1_05['V_C'],(1000000)*(1+CPPI_Backtest_m3_f8_00['MSCI_World_r']).cumprod()],axis=1)
CVSC_Ausübungspreis_K.columns = ['CVSC 0.98','CVSC 1.0','CVSC 1.02','MSCI World']
CVSC_Ausübungspreis_K['Date'] = TimeSeries[12:].index
CVSC_Ausübungspreis_K.set_index('Date',inplace=True)

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True,figsize=(26, 10))
ax1.plot(PP_Ausübungspreis_K/1000000)
ax2.plot(CVSC_Ausübungspreis_K/1000000)
ax1.set_title('PP',fontsize=20)
ax2.set_title('CVSC',fontsize=20)
ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax1.set_ylabel('Vermögem in M€',fontsize=24)
ax1.legend(['PP K0,95','PP K1,0','PP K1,01','MSCI World'],loc='upper left',prop={'size': 18})
ax2.legend(['CVSC K0,99','CVSC K1,0','CVSC K1,05','MSCI World'],loc='upper left',prop={'size': 18})

PP_Ausübungspreis_K.drop(columns=['MSCI World'],inplace = True)

Auswertung_K = pd.concat([PP_Ausübungspreis_K,CVSC_Ausübungspreis_K],axis=1)
Auswertung_K['Date'] = TimeSeries[12:].index
Auswertung_K.set_index('Date',inplace=True)
Auswertung_K = Auswertung_K/1000000

previous_peaks = Auswertung_K.cummax()
drawdowns = (Auswertung_K - previous_peaks)/previous_peaks
dd = drawdowns.min()
returns = Auswertung_K.pct_change()
returns = returns[1:]
std=returns.std(ddof=1)
std_an = std*np.sqrt(12)
total_return = (1+returns).prod()-1
total_return = total_return.transpose()
monthly_return = pow(1+total_return,1/263)-1
monthly_rf = pow((1+TimeSeries['EONIA'][13:]).prod(),1/263)-1
annualized_return = pow(1+monthly_return,12)-1
annualized_rf = pow(1+monthly_rf,12)-1
Sharpe = (monthly_return-monthly_rf)/std
Sharpe_annu = (annualized_return-annualized_rf)/std_an

summary_K= pd.concat([total_return, monthly_return, std,Sharpe,annualized_return,std_an,Sharpe_annu,dd],axis=1)
summary_K.columns=['Gesamtrendite','Rendite_m','Std_m','Sharpe_m','Rendite_j','Std_j','Sharpe_j','Maximum Drawdown']


##########Base Case
CPPI_Backtest_B, TIPP_Backtest_B = CPPI_TIPP(TimeSeries=TimeSeries, m=3, floor=0.8)

CPPI_TIPP_B = pd.concat([CPPI_Backtest_B['V_C'],TIPP_Backtest_B['V_C'],(1000000)*(1+CPPI_Backtest_B['MSCI_World_r']).cumprod()],axis=1)
CPPI_TIPP_B.columns = ['CPPI','TIPP','MSCI World']
CPPI_TIPP_B['Date'] = TimeSeries.index
CPPI_TIPP_B.set_index('Date',inplace=True)

previous_peaks = CPPI_TIPP_B.cummax()
drawdowns = (CPPI_TIPP_B - previous_peaks)/previous_peaks
dd = drawdowns.min()
returns = CPPI_TIPP_B.pct_change()
returns = returns[1:]
std=returns.std(ddof=1)
std_an = std*np.sqrt(12)
total_return = (1+returns).prod()-1
total_return = total_return.transpose()
monthly_return = pow(1+total_return,1/275)-1
monthly_rf = pow((1+TimeSeries['EONIA'][1:]).prod(),1/275)-1
annualized_return = pow(1+monthly_return,12)-1
annualized_rf = pow(1+monthly_rf,12)-1
Sharpe = (monthly_return-monthly_rf)/std
Sharpe_annu = (annualized_return-annualized_rf)/std_an

summary_B= pd.concat([total_return, monthly_return, std,Sharpe,annualized_return,std_an,Sharpe_annu,dd],axis=1)
summary_B.columns=['Gesamtrendite','Rendite_m','Std_m','Sharpe_m','Rendite_j','Std_j','Sharpe_j','Maximum Drawdown']

Weights_B = pd.concat([CPPI_Backtest_B['E_w'],TIPP_Backtest_B['E_w']],axis=1)
Weights_B.columns = ['CPPI_E_w','TIPP_E_w']
Weights_B['Date'] = TimeSeries.index
Weights_B.set_index('Date',inplace=True)
y_1 = Weights_B.index
x11 = Weights_B['CPPI_E_w'] #m3 f9
x12 = Weights_B['TIPP_E_w']

fig = plt.figure(constrained_layout=True,figsize=(50, 35))
gs = GridSpec(2, 2, figure=fig) 
ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(x11,label='_nolegend_')
ax1.fill_between(y_1, x11, step="mid", alpha=0.4,label='Gewicht MSCI World')
ax1.fill_between(y_1, x11,1, step="mid", alpha=0.4,label='Gewicht EONIA 1M')
ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
ax1.set_title('Gewicht CPPI',fontsize=45)
ax1.legend(['Gewicht MSCI World','Gewicht EONIA 1M'],loc='best',prop={'size': 45})
ax1.tick_params(axis='both', which='major', labelsize=35) 
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(x12)
ax2.fill_between(y_1, x12, step="mid", alpha=0.4,label='Gewicht MSCI World')
ax2.fill_between(y_1, x12,1, step="mid", alpha=0.4,label='Gewicht EONIA 1M')
ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
ax2.set_title('Gewicht TIPP',fontsize=45)
ax2.tick_params(axis='both', which='major', labelsize=35) 
ax3 = fig.add_subplot(gs[0, :])
y = pd.concat([CPPI_Backtest_B['V_C']/1000000,TIPP_Backtest_B['V_C']/1000000,(1)*(1+CPPI_Backtest_B['MSCI_World_r']).cumprod(),CPPI_Backtest_B['Floor']/1000000,TIPP_Backtest_B['Floor']/1000000],axis=1)
y['Date'] = TimeSeries.index
y.set_index('Date',inplace=True)
ax3.plot(y,linewidth=2.5)
ax3.legend(['CPPI','TIPP','MSCI World','CPPI Floor','TIPP Floor'],prop={'size': 45})
ax3.set_ylabel('Vermögem in M€',fontsize=45)
ax3.set_xlabel('',fontsize=1)
ax3.xaxis.set_major_locator(plt.MaxNLocator(4))
ax3.tick_params(axis='both', which='major', labelsize=35) 

##################Variation m und floor

CPPI_Backtest_m3_f7, TIPP_Backtest_m3_f7 = CPPI_TIPP(TimeSeries=TimeSeries, m=3, floor=0.7)
CPPI_Backtest_m3_f9, TIPP_Backtest_m3_f9 = CPPI_TIPP(TimeSeries=TimeSeries, m=3, floor=0.9)
CPPI_Backtest_m2_f8, TIPP_Backtest_m2_f8 = CPPI_TIPP(TimeSeries=TimeSeries, m=2, floor=0.8)
CPPI_Backtest_m4_f8, TIPP_Backtest_m4_f8 = CPPI_TIPP(TimeSeries=TimeSeries, m=4, floor=0.8)
CPPI_Backtest_m2_f9, TIPP_Backtest_m2_f9 = CPPI_TIPP(TimeSeries=TimeSeries, m=2, floor=0.9)
CPPI_Backtest_m4_f7, TIPP_Backtest_m4_f7 = CPPI_TIPP(TimeSeries=TimeSeries, m=4, floor=0.7)
#Weights CPPI vs TIPP
 
Weights_m = pd.concat([CPPI_Backtest_m4_f7['E_w'],TIPP_Backtest_m4_f7['E_w']],axis=1)
Weights_m.columns = ['CPPI_E_w','TIPP_E_w']
Weights_m['Date'] = TimeSeries.index
Weights_m.set_index('Date',inplace=True)

y = Weights_m.index
x11 = Weights_m['CPPI_E_w'] #m3 f7
x12 = Weights_m['CPPI_E_w'] #m3 f9
x21 = Weights_m['CPPI_E_w'] #m2 f8
x22 = Weights_m['CPPI_E_w'] #m4 f8
x31 = Weights_m['CPPI_E_w'] #m2 f9
x32 = Weights_m['CPPI_E_w'] #m4 f7
x13 = Weights_m['TIPP_E_w'] #m3 f7
x14 = Weights_m['TIPP_E_w'] #m3 f9
x23 = Weights_m['TIPP_E_w'] #m2 f8
x24 = Weights_m['TIPP_E_w'] #m4 f8
x33 = Weights_m['TIPP_E_w'] #m2 f9
x34 = Weights_m['TIPP_E_w'] #m4 f7


fig, ax = plt.subplots(nrows=3,ncols=4,sharey=True,figsize=(26, 14))
ax[0,0].plot(x11)
ax[0,1].plot(x12)
ax[0,2].plot(x13)
ax[0,3].plot(x14)
ax[1,0].plot(x21)
ax[1,1].plot(x22)
ax[1,2].plot(x23)
ax[1,3].plot(x24)
ax[2,0].plot(x31)
ax[2,1].plot(x32)
ax[2,2].plot(x33)
ax[2,3].plot(x34)

ax[0,0].fill_between(y, x11, step="mid", alpha=0.4,label='Gewicht MSCI World')
ax[0,0].fill_between(y, x11,1, step="mid", alpha=0.4,label='Gewicht EONIA 1M')
ax[0,1].fill_between(y, x12, step="mid", alpha=0.4,label='Gewicht MSCI World')
ax[0,1].fill_between(y, x12,1, step="mid", alpha=0.4,label='Gewicht EONIA 1M')
ax[0,2].fill_between(y, x13, step="mid", alpha=0.4,label='Gewicht MSCI World')
ax[0,2].fill_between(y, x13,1, step="mid", alpha=0.4,label='Gewicht EONIA 1M')
ax[0,3].fill_between(y, x14, step="mid", alpha=0.4,label='Gewicht MSCI World')
ax[0,3].fill_between(y, x14,1, step="mid", alpha=0.4,label='Gewicht EONIA 1M')

ax[1,0].fill_between(y, x21, step="mid", alpha=0.4,label='Gewicht MSCI World')
ax[1,0].fill_between(y, x21,1, step="mid", alpha=0.4,label='Gewicht EONIA 1M')
ax[1,1].fill_between(y, x22, step="mid", alpha=0.4,label='Gewicht MSCI World')
ax[1,1].fill_between(y, x22,1, step="mid", alpha=0.4,label='Gewicht EONIA 1M')
ax[1,2].fill_between(y, x23, step="mid", alpha=0.4,label='Gewicht MSCI World')
ax[1,2].fill_between(y, x23,1, step="mid", alpha=0.4,label='Gewicht EONIA 1M')
ax[1,3].fill_between(y, x24, step="mid", alpha=0.4,label='Gewicht MSCI World')
ax[1,3].fill_between(y, x24,1, step="mid", alpha=0.4,label='Gewicht EONIA 1M')

ax[2,0].fill_between(y, x31, step="mid", alpha=0.4,label='Gewicht MSCI World')
ax[2,0].fill_between(y, x31,1, step="mid", alpha=0.4,label='Gewicht EONIA 1M')
ax[2,1].fill_between(y, x32, step="mid", alpha=0.4,label='Gewicht MSCI World')
ax[2,1].fill_between(y, x32,1, step="mid", alpha=0.4,label='Gewicht EONIA 1M')
ax[2,2].fill_between(y, x33, step="mid", alpha=0.4,label='Gewicht MSCI World')
ax[2,2].fill_between(y, x33,1, step="mid", alpha=0.4,label='Gewicht EONIA 1M')
ax[2,3].fill_between(y, x34, step="mid", alpha=0.4,label='Gewicht MSCI World')
ax[2,3].fill_between(y, x34,1, step="mid", alpha=0.4,label='Gewicht EONIA 1M')

ax[0,0].set_title('CPPI (m=3, f=0.7)',fontsize=20)
ax[0,1].set_title('CPPI (m=3, f=0.9)',fontsize=20)
ax[0,2].set_title('TIPP (m=3, f=0.7)',fontsize=20)
ax[0,3].set_title('TIPP (m=3, f=0.9)',fontsize=20)
ax[1,0].set_title('CPPI (m=2, f=0.8)',fontsize=20)
ax[1,1].set_title('CPPI (m=4, f=0.8)',fontsize=20)
ax[1,2].set_title('TIPP (m=2, f=0.8)',fontsize=20)
ax[1,3].set_title('TIPP (m=4, f=0.8)',fontsize=20)
ax[2,0].set_title('CPPI (m=2, f=0.9)',fontsize=20)
ax[2,1].set_title('CPPI (m=4, f=0.7)',fontsize=20)
ax[2,2].set_title('TIPP (m=2, f=0.9)',fontsize=20)
ax[2,3].set_title('TIPP (m=4, f=0.7)',fontsize=20)
plt.subplots_adjust(wspace=0.0)
ax[0,0].xaxis.set_major_locator(plt.MaxNLocator(4))
ax[0,1].xaxis.set_major_locator(plt.MaxNLocator(4))
ax[0,2].xaxis.set_major_locator(plt.MaxNLocator(4))
ax[0,3].xaxis.set_major_locator(plt.MaxNLocator(4))
ax[1,0].xaxis.set_major_locator(plt.MaxNLocator(4))
ax[1,1].xaxis.set_major_locator(plt.MaxNLocator(4))
ax[1,2].xaxis.set_major_locator(plt.MaxNLocator(4))
ax[1,3].xaxis.set_major_locator(plt.MaxNLocator(4))
ax[2,0].xaxis.set_major_locator(plt.MaxNLocator(4))
ax[2,1].xaxis.set_major_locator(plt.MaxNLocator(4))
ax[2,2].xaxis.set_major_locator(plt.MaxNLocator(4))
ax[2,3].xaxis.set_major_locator(plt.MaxNLocator(4))
#ax[0].legend( loc='upper center',prop={'size': 12})
handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',prop={'size': 16})

####Verlauf Backtest m und f
CPPI_N = pd.concat([CPPI_Backtest_m3_f7['V_C'],CPPI_Backtest_m3_f9['V_C'],CPPI_Backtest_m2_f8['V_C'],CPPI_Backtest_m4_f8['V_C'],CPPI_Backtest_m2_f9['V_C'],CPPI_Backtest_m4_f7['V_C'],CPPI_Backtest_m3_f7['Floor'],CPPI_Backtest_m2_f8['Floor'],CPPI_Backtest_m3_f9['Floor'],(1000000)*(1+CPPI_Backtest_m2_f9['MSCI_World_r']).cumprod()],axis=1)
CPPI_N.columns = ['CPPI m3 f7','CPPI m3 f9','CPPI m2 f8','CPPI m4 f8','CPPI m2 f9','CPPI m4 f7','f7 Floor','f8 Floor','f9 Floor','MSCI World']
CPPI_N['Date'] = TimeSeries.index
CPPI_N.set_index('Date',inplace=True)
CPPI_N=CPPI_N/1000000

TIPP_N = pd.concat([TIPP_Backtest_m3_f7['V_C'],TIPP_Backtest_m3_f9['V_C'],TIPP_Backtest_m2_f8['V_C'],TIPP_Backtest_m4_f8['V_C'],TIPP_Backtest_m2_f9['V_C'],TIPP_Backtest_m4_f7['V_C'],TIPP_Backtest_m3_f7['Floor'],TIPP_Backtest_m3_f9['Floor'],TIPP_Backtest_m2_f8['Floor'],TIPP_Backtest_m4_f8['Floor'],TIPP_Backtest_m2_f9['Floor'],TIPP_Backtest_m4_f7['Floor'],(1000000)*(1+CPPI_Backtest_m2_f9['MSCI_World_r']).cumprod()],axis=1)
TIPP_N.columns = ['TIPP m3 f7','TIPP m3 f9','TIPP m2 f8','TIPP m4 f8','TIPP m2 f9','TIPP m4 f7','m3 f7 Floor','m3 f9 Floor','m2 f8 Floor','m4 f8 Floor','m2 f9 Floor','m4 f7 Floor','MSCI World']
TIPP_N['Date'] = TimeSeries.index
TIPP_N.set_index('Date',inplace=True)
TIPP_N=TIPP_N/1000000

y = CPPI_N.index
fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3,ncols=2,sharey=True,figsize=(24, 22))
ax1.plot(CPPI_N[['CPPI m3 f7','CPPI m3 f9','f7 Floor','f9 Floor','MSCI World']])
ax2.plot(TIPP_N[['TIPP m3 f7','TIPP m3 f9','m3 f7 Floor','m3 f9 Floor','MSCI World']])
ax3.plot(CPPI_N[['CPPI m2 f8','CPPI m4 f8','f8 Floor','MSCI World']])
ax4.plot(TIPP_N[['TIPP m2 f8','TIPP m4 f8','m2 f8 Floor','m4 f8 Floor','MSCI World']])
ax5.plot(CPPI_N[['CPPI m2 f9','CPPI m4 f7','f7 Floor','f9 Floor','MSCI World']])
ax6.plot(TIPP_N[['TIPP m2 f9','TIPP m4 f7','m2 f9 Floor','m4 f7 Floor','MSCI World']])
plt.subplots_adjust(wspace=0.0)
ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
ax2.xaxis.set_major_locator(plt.MaxNLocator(6))
ax3.xaxis.set_major_locator(plt.MaxNLocator(6))
ax4.xaxis.set_major_locator(plt.MaxNLocator(6))
ax5.xaxis.set_major_locator(plt.MaxNLocator(6))
ax6.xaxis.set_major_locator(plt.MaxNLocator(6))
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)
ax4.tick_params(axis='both', which='major', labelsize=14)
ax5.tick_params(axis='both', which='major', labelsize=14)
ax6.tick_params(axis='both', which='major', labelsize=14)
ax1.set_title('CPPI Variation f',fontsize=20)
ax2.set_title('TIPP Variation f',fontsize=20)
ax3.set_title('CPPI Variation m',fontsize=20)
ax4.set_title('TIPP Variation m',fontsize=20)
ax5.set_title('CPPI Variation f und m',fontsize=20)
ax6.set_title('TIPP Variation f und m',fontsize=20)
ax1.set_ylabel('Vermögen in M€',fontsize=16)
ax3.set_ylabel('Vermögen in M€',fontsize=16)
ax5.set_ylabel('Vermögen in M€',fontsize=16)
ax1.legend(['CPPI m3 f7','CPPI m3 f9','f7 Floor','f9 Floor','MSCI World'],loc='upper left',prop={'size': 12})
ax2.legend(['TIPP m3 f7','TIPP m3 f9','m3 f7 Floor','m3 f9 Floor','MSCI World'],loc='upper left',prop={'size': 12})
ax3.legend(['CPPI m2 f8','CPPI m4 f8','f8 Floor','MSCI World'],loc='upper left',prop={'size': 12})
ax4.legend(['TIPP m2 f8','TIPP m4 f8','m2 f8 Floor','m4 f8 Floor','MSCI World'],loc='upper left',prop={'size': 12})
ax5.legend(['CPPI m2 f9','CPPI m4 f7','f7 Floor','f9 Floor','MSCI World'],loc='upper left',prop={'size': 12})
ax6.legend(['TIPP m2 f9','TIPP m4 f7','m2 f9 Floor','m4 f7 Floor','MSCI World'],loc='upper left',prop={'size': 12})

###summary statistics

CPPI_N.drop(columns=['f7 Floor','f8 Floor','f9 Floor'],inplace = True)
TIPP_N.drop(columns=['m3 f7 Floor','m3 f9 Floor','m2 f8 Floor','m4 f8 Floor','m2 f9 Floor','m4 f7 Floor','MSCI World'],inplace = True)
CPPI_TIPP_N = pd.concat([CPPI_N,TIPP_N],axis=1)

corr_mat_N = CPPI_TIPP_N.corr()

previous_peaks = CPPI_TIPP_N.cummax()
drawdowns = (CPPI_TIPP_N - previous_peaks)/previous_peaks
dd = drawdowns.min()
returns = CPPI_TIPP_N.pct_change()
returns = returns[1:]
std=returns.std(ddof=1)
std_an = std*np.sqrt(12)
total_return = (1+returns).prod()-1
total_return = total_return.transpose()
monthly_return = pow(1+total_return,1/275)-1
monthly_rf = pow((1+TimeSeries['EONIA'][1:]).prod(),1/275)-1
annualized_return = pow(1+monthly_return,12)-1
annualized_rf = pow(1+monthly_rf,12)-1
Sharpe = (monthly_return-monthly_rf)/std
Sharpe_annu = (annualized_return-annualized_rf)/std_an

summary_N= pd.concat([total_return, monthly_return, std,Sharpe,annualized_return,std_an,Sharpe_annu,dd],axis=1)
summary_N.columns=['Gesamtrendite','Rendite_m','Std_m','Sharpe_m','Rendite_j','Std_j','Sharpe_j','Maximum Drawdown']


##### Unterschiedliche Startpunkte

CPPI_Backtest_m3_f8_99, TIPP_Backtest_m3_f8_99 = CPPI_TIPP(TimeSeries=TimeSeries, m=3, floor=0.8)
CPPI_Backtest_m3_f8_00, TIPP_Backtest_m3_f8_00 = CPPI_TIPP(TimeSeries=TimeSeries[:][24:], m=3, floor=0.8)
CPPI_Backtest_m3_f8_01, TIPP_Backtest_m3_f8_01 = CPPI_TIPP(TimeSeries=TimeSeries[:][48:], m=3, floor=0.8)

CPPI_T = pd.concat([CPPI_Backtest_m3_f8_99['V_C'],CPPI_Backtest_m3_f8_00['V_C'],CPPI_Backtest_m3_f8_01['V_C'],CPPI_Backtest_m3_f8_99['Floor']],axis=1)
CPPI_T.columns = ['CPPI 1999','CPPI 2001','CPPI 2003','Floor 1999']
CPPI_T = CPPI_T.iloc[::-1]
CPPI_T = CPPI_T.apply(lambda x: pd.Series(x.dropna().values))
CPPI_T = CPPI_T.iloc[::-1]
CPPI_T['Date'] = TimeSeries.index
CPPI_T.set_index('Date',inplace=True)
CPPI_T = CPPI_T/1000000

TIPP_T = pd.concat([TIPP_Backtest_m3_f8_99['V_C'],TIPP_Backtest_m3_f8_00['V_C'],TIPP_Backtest_m3_f8_01['V_C'],TIPP_Backtest_m3_f8_99['Floor'],TIPP_Backtest_m3_f8_00['Floor'],TIPP_Backtest_m3_f8_01['Floor']],axis=1)
TIPP_T.columns = ['TIPP 1999','TIPP 2001','TIPP 2003','Floor 1999','Floor 2001','Floor 2003']
TIPP_T = TIPP_T.iloc[::-1]
TIPP_T = TIPP_T.apply(lambda x: pd.Series(x.dropna().values))
TIPP_T = TIPP_T.iloc[::-1]
TIPP_T['Date'] = TimeSeries.index
TIPP_T.set_index('Date',inplace=True)
TIPP_T = TIPP_T/1000000

y = CPPI_T.index
fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True,figsize=(24, 9))
ax1.plot(CPPI_T[['CPPI 1999','CPPI 2001','CPPI 2003','Floor 1999']])
ax2.plot(TIPP_T[['TIPP 1999','TIPP 2001','TIPP 2003','Floor 1999','Floor 2001','Floor 2003']])
ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
ax2.xaxis.set_major_locator(plt.MaxNLocator(6))
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax1.set_title('CPPI',fontsize=20)
ax2.set_title('TIPP',fontsize=20)
ax1.set_ylabel('Vermögen in M€',fontsize=16)
ax1.legend(['CPPI 1999','CPPI 2001','CPPI 2003','Floor 1999'],loc='upper left',prop={'size': 12})
ax2.legend(['TIPP 1999','TIPP 2001','TIPP 2003','Floor 1999','Floor 2001','Floor 2003'],loc='upper left',prop={'size': 12})
##summary statistics unterschiedloche Startzeitpunkte

CPPI_T.drop(columns=['Floor 1999'],inplace = True)
TIPP_T.drop(columns=['Floor 1999','Floor 2001','Floor 2003'],inplace = True)
CPPI_TIPP_T = pd.concat([CPPI_T,TIPP_T],axis=1)

previous_peaks = CPPI_TIPP_T.cummax()
drawdowns = (CPPI_TIPP_T - previous_peaks)/previous_peaks
dd = drawdowns.min()
returns = CPPI_TIPP_T.pct_change()
returns = returns[1:]
std=returns.std(ddof=1)
std_an = std*np.sqrt(12)
total_return = (1+returns).prod()-1
total_return = total_return.transpose()
periods_potenz = [1/275,1/(275-24),1/(275-48),1/275,1/(275-24),1/(275-48)]
monthly_return = pow(1+total_return,periods_potenz)-1 
monthly_rf = [pow((1+TimeSeries['EONIA'][1:]).prod(),1/275)-1, pow((1+TimeSeries['EONIA'][25:]).prod(),1/(275-25))-1,pow((1+TimeSeries['EONIA'][49:]).prod(),1/(275-49))-1]
monthly_rf =  pd.DataFrame(monthly_rf + monthly_rf)
monthly_rf['index'] = monthly_return.index
monthly_rf.set_index('index',inplace=True)
annualized_return = pow(1+monthly_return,12)-1
annualized_rf = pow(1+monthly_rf[0],12)-1
Sharpe = (monthly_return-monthly_rf[0])/std
Sharpe_annu = (annualized_return-annualized_rf)/std_an

summary_T= pd.concat([total_return, monthly_return, std,Sharpe,annualized_return,std_an,Sharpe_annu,dd],axis=1)
summary_T.columns=['Gesamtrendite','Rendite_m','Std_m','Sharpe_m','Rendite_j','Std_j','Sharpe_j','Maximum Drawdown']


####Unterschiedliche Vermögen
CPPI_Backtest_Mio, TIPP_Backtest_Mio = CPPI_TIPP(TimeSeries=TimeSeries, m=3, floor=0.8,start=1000000)
CPPI_Backtest_ttau, TIPP_Backtest_ttau = CPPI_TIPP(TimeSeries=TimeSeries, m=3, floor=0.8,start=10000)
CPPI_Backtest_tau, TIPP_Backtest_tau = CPPI_TIPP(TimeSeries=TimeSeries, m=3, floor=0.8,start=1000)

CPPI_V = pd.concat([CPPI_Backtest_Mio['V_C'],CPPI_Backtest_ttau['V_C'],CPPI_Backtest_tau['V_C'],(1000000)*(1+CPPI_Backtest_Mio['MSCI_World_r']).cumprod()],axis=1)
CPPI_V.columns = ['CPPI 1M','CPPI 10K','CPPI 1K','MSCI World']
CPPI_V['Date'] = TimeSeries.index
CPPI_V.set_index('Date',inplace=True)

TIPP_V = pd.concat([TIPP_Backtest_Mio['V_C'],TIPP_Backtest_ttau['V_C'],TIPP_Backtest_tau['V_C']],axis=1)
TIPP_V.columns=['TIPP 1M','TIPP 10K','TIPP 1K']
TIPP_V['Date'] = TimeSeries.index
TIPP_V.set_index('Date',inplace=True)

CPPI_TIPP_V = pd.concat([CPPI_V,TIPP_V],axis=1)

previous_peaks = CPPI_TIPP_V.cummax()
drawdowns = (CPPI_TIPP_V - previous_peaks)/previous_peaks
dd = drawdowns.min()
returns = CPPI_TIPP_V.pct_change()
returns = returns[1:]
std=returns.std(ddof=1)
std_an = std*np.sqrt(12)
total_return = (1+returns).prod()-1
total_return = total_return.transpose()
monthly_return = pow(1+total_return,1/275)-1
monthly_rf = pow((1+TimeSeries['EONIA'][1:]).prod(),1/275)-1
annualized_return = pow(1+monthly_return,12)-1
annualized_rf = pow(1+monthly_rf,12)-1
Sharpe = (monthly_return-monthly_rf)/std
Sharpe_annu = (annualized_return-annualized_rf)/std_an

summary_V= pd.concat([total_return, monthly_return, std,Sharpe,annualized_return,std_an,Sharpe_annu,dd],axis=1)
summary_V.columns=['Gesamtrendite','Rendite_m','Std_m','Sharpe_m','Rendite_j','Std_j','Sharpe_j','Maximum Drawdown']

####################################################################################
################################  Simulation Auswertung #####################################
#############################################################################################

PP_Simulation = PP(risky_r=pd.DataFrame(MC_R_MSCI_World),safe_r=pd.DataFrame(MC_EONIA))
CVSC_Simulation = CVSC(risky_r=pd.DataFrame(MC_R_MSCI_World),safe_r=pd.DataFrame(MC_EONIA)) 
CPPI_Simulation = CPPI(risky_r=pd.DataFrame(MC_R_MSCI_World),safe_r=pd.DataFrame(MC_EONIA),floor=0.8,m=3,TIPP=False)
TIPP_Simulation = CPPI(risky_r=pd.DataFrame(MC_R_MSCI_World),safe_r=pd.DataFrame(MC_EONIA),floor=0.8,m=3,TIPP=True)

### Darstellung EONIA und MSCI World
MC_R_MSCI_World_calc = MC_R_MSCI_World.copy()
MC_R_MSCI_World_calc.set_index(MC_R_MSCI_World_calc.index +1,inplace=True)
MC_EONIA_calc = MC_EONIA.copy()
MC_EONIA_calc.set_index(MC_EONIA_calc.index +1,inplace=True)
MC_MSCI_World_cum = 100*((1+MC_R_MSCI_World_calc).cumprod()-1)
EONIA_cum = 100*((1+MC_EONIA_calc).cumprod()-1)
x1= np.mean(MC_MSCI_World_cum,axis=1)
x1max = MC_MSCI_World_cum.max(axis=1)
x1min = MC_MSCI_World_cum.min(axis=1)
x2 = np.mean(EONIA_cum,axis=1)
x2max = EONIA_cum.max(axis=1)
x2min = EONIA_cum.min(axis=1)
y = x1.index
ax = x1.plot(figsize=(24, 11))
ax.plot(x2)
plt.axhline(y=0.0, color='k', linestyle='-')
ax.fill_between(y,x1min, x1max.transpose(), step="pre", alpha=0.4)
ax.fill_between(y,x2min, x2max.transpose(), step="pre", alpha=0.4)
ax.legend(['MSCI World','EONIA 1M','x=0'],loc='upper left',prop={'size': 30})
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('Rendite in %',fontsize=30)
ax.set_xlabel('Monate',fontsize=30)
ax.set_xlim(1)
ax.autoscale(axis='x',tight=True)




#Prob Distribution
PP_S = PP_Simulation['V_C'].transpose()
CVSC_S = CVSC_Simulation['V_C'].transpose()
CPPI_S = CPPI_Simulation['V_C'].transpose()
TIPP_S = TIPP_Simulation['V_C'].transpose()
minimum = (CVSC_S[23].min())/1000000
maximum = (CPPI_S[23].max())/1000000
number_bins = 250
bins = np.linspace(minimum,maximum,number_bins)
fig, axs = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True,tight_layout=True,figsize=(32, 18))
axs[0,0].hist(PP_S[23]/1000000,alpha=0.7, bins=bins)
axs[1,0].hist(CVSC_S[23]/1000000,alpha=0.7,bins=bins)
axs[0,1].hist(CPPI_S[23]/1000000,alpha=0.7, bins=bins)
axs[1,1].hist(TIPP_S[23]/1000000,alpha=0.7,bins=bins)
axs[0,0].set_ylabel('Anzahl der Pfade',fontsize=35)
axs[1,0].set_xlabel('Endvermögen in M€',fontsize=35)
axs[1,0].set_ylabel('Anzahl der Pfade',fontsize=35)
axs[1,1].set_xlabel('Endvermögen in M€',fontsize=35)
axs[0,0].set_title('PP',fontsize=45)
axs[1,0].set_title('CVSC',fontsize=45)
axs[0,1].set_title('CPPI',fontsize=45)
axs[1,1].set_title('TIPP',fontsize=45)
axs[0,0].tick_params(axis='both', which='major', labelsize=30)  
axs[1,0].tick_params(axis='both', which='major', labelsize=30) 
axs[0,1].tick_params(axis='both', which='major', labelsize=30)  
axs[1,1].tick_params(axis='both', which='major', labelsize=30)      
plt.xlim(minimum, maximum)

Endvermögen = pd.concat([PP_S[23],CVSC_S[23],CPPI_S[23],TIPP_S[23]],axis=1)
Endvermögen.columns=['PP','CVSC','CPPI','TIPP']
summary1 = Endvermögen.describe()
summary2 = rp.summary_cont(Endvermögen)
summary2.set_index('Variable',inplace=True)
summary2['skew'] = Endvermögen.skew()
summary2['kurt'] = Endvermögen.kurtosis()
summary = pd.concat([summary1,summary2.transpose()],axis=0)

TIPP_S_Floor = TIPP_Simulation['Floor'].transpose()
unterschreitung = TIPP_S[23] - TIPP_S_Floor[23] < 0



##### Shortfall
def Shortfallmaße(asset):
    
    Shortfall_bol = asset < 1000000
    Shortfall = (1000000 - asset)*Shortfall_bol 
    SW = sum(Shortfall_bol)/len(asset)
    SE = sum(Shortfall)/len(asset)
    if sum(Shortfall_bol) == 0:
        MEL = 0
    else:
        MEL = sum(Shortfall)/sum(Shortfall_bol)
    
    return(SW,SE,MEL)

SW_PP, SE_PP, MEL_PP = Shortfallmaße(asset=PP_S[23])
SW_CVSC, SE_CVSC, MEL_CVSC = Shortfallmaße(CVSC_S[23])
SW_CPPI, SE_CPPI, MEL_CPPI = Shortfallmaße(CPPI_S[23])
SW_TIPP, SE_TIPP, MEL_TIPP = Shortfallmaße(TIPP_S[23])

SW_PP = []
SW_CVSC = []
MEL_PP = []
MEL_CVSC = []
SW_CPPI = []
SW_TIPP = []
MEL_CPPI = []
MEL_TIPP = []
for t in range(len(CPPI_S.transpose())):
    SW1, SE1, MEL1 = Shortfallmaße(asset=PP_S[t])
    SW2, SE2, MEL2 = Shortfallmaße(asset=CVSC_S[t])
    SW3, SE3, MEL3 = Shortfallmaße(asset=CPPI_S[t])
    SW4, SE4, MEL4 = Shortfallmaße(asset=TIPP_S[t])
    SW_PP.append(SW1)
    SW_CVSC.append(SW2)
    MEL_PP.append(MEL1)
    MEL_CVSC.append(MEL2)
    SW_CPPI.append(SW3)
    SW_TIPP.append(SW4)
    MEL_CPPI.append(MEL3)
    MEL_TIPP.append(MEL4)
    
    
####Shortfälle über die Zeit
SW_PP = [SW_PP * 100 for SW_PP in SW_PP]
SW_CVSC = [SW_CVSC * 100 for SW_CVSC in SW_CVSC]
SW_CPPI = [SW_CPPI * 100 for SW_CPPI in SW_CPPI]
SW_TIPP = [SW_TIPP * 100 for SW_TIPP in SW_TIPP]

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharex=True,tight_layout=True,figsize=(24, 10))
ax1.plot(SW_PP)
ax1.plot(SW_CVSC)
ax1.plot(SW_CPPI)
ax1.plot(SW_TIPP)
ax2.plot(MEL_PP)
ax2.plot(MEL_CVSC)
ax2.plot(MEL_CPPI)
ax2.plot(MEL_TIPP)
ax1.set_title('SW',fontsize=35)
ax2.set_title('MEL',fontsize=35)
ax1.tick_params(axis='both', which='major', labelsize=20)  
ax2.tick_params(axis='both', which='major', labelsize=20)  
ax2.set_ylabel('MEL in €',fontsize=30)
ax1.set_ylabel('SW in %',fontsize=30)
ax1.set_xlabel('Monate',fontsize=30)
ax2.set_xlabel('Monate',fontsize=30)
ax1.legend(['PP','CVSC','CPPI','TIPP'],loc='upper right',prop={'size': 25})
ax2.legend(['PP','CVSC','CPPI','TIPP'],loc='upper left',prop={'size': 25})


#####stochastische dominanz
plt.figure(figsize=(21, 14))
PP_count, PP_bins_count = np.histogram(PP_S[23]/1000000, bins=75)
PP_pdf = PP_count / sum(PP_count)
PP_cdf = np.cumsum(PP_pdf)
plt.plot(PP_bins_count[1:], PP_cdf, label="PP")
CVSC_count, CVSC_bins_count = np.histogram(CVSC_S[23]/1000000, bins=75)
CVSC_pdf = CVSC_count / sum(CVSC_count)
CVSC_cdf = np.cumsum(CVSC_pdf)
plt.plot(CVSC_bins_count[1:], CVSC_cdf, label="CVSC")
CPPI_count, CPPI_bins_count = np.histogram(CPPI_S[23]/1000000, bins=75)
CPPI_pdf = CPPI_count / sum(CPPI_count)
CPPI_cdf = np.cumsum(CPPI_pdf)
plt.plot(CPPI_bins_count[1:], CPPI_cdf, label="CPPI")
TIPP_count, TIPP_bins_count = np.histogram(TIPP_S[23]/1000000, bins=75)
TIPP_pdf = TIPP_count / sum(TIPP_count)
TIPP_cdf = np.cumsum(TIPP_pdf)
plt.plot(TIPP_bins_count[1:], TIPP_cdf, label="TIPP")
plt.legend(fontsize=30)
plt.xlabel('Endvermögen in €', fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=20)   
plt.axhline(y = 0, color = 'k')
plt.axhline(y = 1, color = 'k')



fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,sharex=True,tight_layout=True,figsize=(48,30))
ax1.plot(PP_bins_count[1:], PP_cdf, label="PP",linewidth=2.5)
ax1.plot(CPPI_bins_count[1:], CPPI_cdf, label="CPPI",linewidth=2.5)
ax2.plot(CVSC_bins_count[1:], CVSC_cdf, label="CVSC",linewidth=2.5)
ax2.plot(CPPI_bins_count[1:], CPPI_cdf, label="CPPI",linewidth=2.5)
ax3.plot(PP_bins_count[1:], PP_cdf, label="PP",linewidth=2.5)
ax3.plot(TIPP_bins_count[1:], TIPP_cdf, label="TIPP",linewidth=2.5)
ax4.plot(CVSC_bins_count[1:], CVSC_cdf, label="CVSC",linewidth=2.5)
ax4.plot(TIPP_bins_count[1:], TIPP_cdf, label="TIPP",linewidth=2.5)
ax1.legend(fontsize=45)
ax2.legend(fontsize=45)
ax3.legend(fontsize=45)
ax4.legend(fontsize=45)
ax1.axhline(y = 0, color = 'k')
ax1.axhline(y = 1, color = 'k')
ax2.axhline(y = 0, color = 'k')
ax2.axhline(y = 1, color = 'k')
ax3.axhline(y = 0, color = 'k')
ax3.axhline(y = 1, color = 'k')
ax4.axhline(y = 0, color = 'k')
ax4.axhline(y = 1, color = 'k')
ax3.set_xlabel('Endvermögen in €', fontsize=40)
ax4.set_xlabel('Endvermögen in €', fontsize=40)
ax1.tick_params(axis='both', which='major', labelsize=30) 
ax2.tick_params(axis='both', which='major', labelsize=30) 
ax3.tick_params(axis='both', which='major', labelsize=30) 
ax4.tick_params(axis='both', which='major', labelsize=30) 


#####Unterschiedliche Laufzeiten Dichtefunktionen
PP_Simulation36 = PP(risky_r=pd.DataFrame(MC_R_MSCI_World),safe_r=pd.DataFrame(MC_EONIA))
CVSC_Simulation36 = CVSC(risky_r=pd.DataFrame(MC_R_MSCI_World),safe_r=pd.DataFrame(MC_EONIA)) 
CPPI_Simulation36 = CPPI(risky_r=pd.DataFrame(MC_R_MSCI_World),safe_r=pd.DataFrame(MC_EONIA),floor=0.8,m=3,TIPP=False)
TIPP_Simulation36 = CPPI(risky_r=pd.DataFrame(MC_R_MSCI_World),safe_r=pd.DataFrame(MC_EONIA),floor=0.8,m=3,TIPP=True)

PP_Simulation12 = PP(risky_r=pd.DataFrame(MC_R_MSCI_World),safe_r=pd.DataFrame(MC_EONIA))
CVSC_Simulation12 = CVSC(risky_r=pd.DataFrame(MC_R_MSCI_World),safe_r=pd.DataFrame(MC_EONIA)) 
CPPI_Simulation12 = CPPI(risky_r=pd.DataFrame(MC_R_MSCI_World),safe_r=pd.DataFrame(MC_EONIA),floor=0.8,m=3,TIPP=False)
TIPP_Simulation12 = CPPI(risky_r=pd.DataFrame(MC_R_MSCI_World),safe_r=pd.DataFrame(MC_EONIA),floor=0.8,m=3,TIPP=True)

PP_S_36 = PP_Simulation36['V_C'].transpose()
CVSC_S_36 = CVSC_Simulation36['V_C'].transpose()
CPPI_S_36 = CPPI_Simulation36['V_C'].transpose()
TIPP_S_36 = TIPP_Simulation36['V_C'].transpose()
PP_S_12 = PP_Simulation36['V_C'].transpose()
CVSC_S_12 = CVSC_Simulation36['V_C'].transpose()
CPPI_S_12 = CPPI_Simulation36['V_C'].transpose()
TIPP_S_12 = TIPP_Simulation36['V_C'].transpose()

minimum = (CVSC_S_36[35].min())/1000000
maximum = (CPPI_S_36[35].max())/1000000
number_bins = 250
bins = np.linspace(minimum,maximum,number_bins)
fig, axs = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True,tight_layout=True,figsize=(32, 20))
axs[0,0].hist(PP_S_36[35]/1000000,alpha=0.7, bins=bins,label='36 Monate')
axs[1,0].hist(CVSC_S_36[35]/1000000,alpha=0.7,bins=bins)
axs[0,1].hist(CPPI_S_36[35]/1000000,alpha=0.7, bins=bins)
axs[1,1].hist(TIPP_S_36[35]/1000000,alpha=0.7,bins=bins)
axs[0,0].hist(PP_S_12[11]/1000000,alpha=0.7, bins=bins,label='12 Monate')
axs[1,0].hist(CVSC_S_12[11]/1000000,alpha=0.7,bins=bins)
axs[0,1].hist(CPPI_S_12[11]/1000000,alpha=0.7, bins=bins)
axs[1,1].hist(TIPP_S_12[11]/1000000,alpha=0.7,bins=bins)
axs[0,0].set_ylabel('Anzahl der Pfade',fontsize=35)
axs[1,0].set_xlabel('Endvermögen in M€',fontsize=35)
axs[1,0].set_ylabel('Anzahl der Pfade',fontsize=35)
axs[1,1].set_xlabel('Endvermögen in M€',fontsize=35)
axs[0,0].set_title('PP',fontsize=45)
axs[1,0].set_title('CVSC',fontsize=45)
axs[0,1].set_title('CPPI',fontsize=45)
axs[1,1].set_title('TIPP',fontsize=45)
axs[0,0].tick_params(axis='both', which='major', labelsize=30)  
axs[1,0].tick_params(axis='both', which='major', labelsize=30) 
axs[0,1].tick_params(axis='both', which='major', labelsize=30)  
axs[1,1].tick_params(axis='both', which='major', labelsize=30)      
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',prop={'size': 45})

