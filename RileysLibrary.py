# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:08:42 2021

@author: conlon
"""





# ================================ LIBRARIES ================================ #

# Meta data handling
import os
import threading
import functools
import json
from pathlib import Path

# Data & Math
import math
import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant

# APIs
from fredapi import Fred

# Date & time handling
from time import sleep
from datetime import timedelta, datetime

# URL & HTML handling
from requests import post# get, post
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as soup

# Stats & Modeling
from sklearn.linear_model import LinearRegression
from scipy import stats

# Polygon handling
from shapely import wkt
from shapely.ops import cascaded_union#,nearest_points
from shapely.affinity import translate, scale

# Charting
import matplotlib.pyplot as plt
from scipy.stats import norm


try : 
    with open("fred_api_key.txt", "r") as f: 
        key = f.read().strip()
except :
    key = input('FRED api key?: ')
fred = Fred(key)


dictColor = {'VGood'    :'#115F4F',
             'Good'     :'#42B09D',
             'Neutral'  :'#F0C84E',
             'Bad'      :'#B34040',
             'VBad'     :'#5F1A1A',
             'Water'    :'#A5DBF7',
             'NoFill'   :'#D0D0D3',
             'Other'    :'#9A9A9A',
             'Border'   :'#FFFFFF'
             }

# V Good    / #115F4F
# Good      / #42B09D
# Neutral   / #F0C84E
# Bad       / #B34040
# V Bad     / #5F1A1A
# Water     / #A5DBF7
# No Fill   / #D0D0D3
# Other     / #9A9A9A
# Borders   / #FFFFFF

# =========================================================================== #





# =============================== DECORATORS ================================ #
def scheduleRun(
        hour=22, 
        minute=0, 
        weekdays_only=True,
        runNow=True,
        log = None
        ):
    def decorator(func):
        # Set escape parameter
        stopEvent = threading.Event()
        
        # Define scheduling wrapper
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if runNow :
                if not log :
                    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Running immediately (runNow=True)...")
                else :
                    log.logging.info(
                        f'[{datetime.now():%Y-%m-%d %H:%M:%S}] '+
                        'Running immediately (runNow=True)...'
                        )
                    
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    if not log :
                        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Error: {e}")
                    else :
                        log.logging.info(
                            f'[{datetime.now():%Y-%m-%d %H:%M:%S}] '+
                            f'Error: {e}'
                            )
                    #   end if
                #   end try/except
            #   end if runNow
            
            while not stopEvent.isSet() :
                dtCurrent = datetime.now()
                target = dtCurrent.replace(hour=hour, minute=minute, second=0, microsecond=0)

                # If current time is past today's target, move to next day
                if dtCurrent >= target:
                    target += timedelta(days=1)
                #   end if

                # If weekdays_only, skip weekends
                if weekdays_only:
                    while target.weekday() >= 5:
                        target += timedelta(days=1)
                #   end if
                
                # Get sleep duration
                sleep_seconds = (target - dtCurrent).total_seconds()
                if not log :
                    print(f"[{dtCurrent:%Y-%m-%d %H:%M:%S}] Next run at {target:%Y-%m-%d %H:%M:%S}"
                          )
                else :
                    log.logging.info(
                        f'[{datetime.now():%Y-%m-%d %H:%M:%S}] Next run at {target:%Y-%m-%d %H:%M:%S}'
                        )
                #   end if

                # Sleep in interval to allow for interrupt
                interval = 60
                while sleep_seconds > 0 and not stopEvent.is_set():
                    sleep(min(interval, sleep_seconds))
                    sleep_seconds -= interval

                # Check for interrupt to sleep
                if stopEvent.is_set():
                    if not log :
                        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Scheduler stopped.")
                    else :
                        log.logging.info(
                            f'[{datetime.now():%Y-%m-%d %H:%M:%S}] '+
                            'Scheduler stopped.'
                            )
                    #   end if
                    break # keyboard interrupt
                #   end if
                
                # Run function; Restart loop
                if not log :
                    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Running function...")
                else :
                    log.logging.info(
                        f'[{datetime.now():%Y-%m-%d %H:%M:%S}] '+
                        'Scheduler stopped.'
                        )
                #   end if
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    if not log :
                        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Error: {e}")
                    else :
                        log.logging.info(
                            f'[{datetime.now():%Y-%m-%d %H:%M:%S}] '+
                            f'Error: {e}"'
                            )
                #   end try/except
            #   end while
        return wrapper
    #   end decorator-wrapper
    return decorator
####
def retryOnError(
        error_types, 
        max_retries=3, 
        delay=30,
        log=None
        ):
    """
    Decorator that retries a function if it raises a specified error.
    
    Parameters
    ----------
    error_types : Exception or tuple of Exception
        The exception(s) that should trigger a retry.
    max_retries : int, optional
        Maximum number of retries before giving up (default: 3).
    delay : int or float, optional
        Delay in seconds between retries (default: 5).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except error_types as e:
                    if attempt < max_retries:
                        if not log :
                            print(f"Attempt {attempt} failed with {e}. "+
                                  f"Retrying in {delay}s..."
                                  )
                        else :
                            log.logging.info(
                                f'[{datetime.now():%Y-%m-%d %H:%M:%S}] '+
                                f"Attempt {attempt} failed with {e}. "+
                                f"Retrying in {delay}s..."
                                )
                        #   end if
                        sleep(delay)
                    else:
                        if not log :
                            print(f"Attempt {attempt} failed with {e}. "+
                                  "No retries left."
                                  )
                        else :
                            log.logging.info(
                                f'[{datetime.now():%Y-%m-%d %H:%M:%S}] '+
                                f"Attempt {attempt} failed with {e}. "+
                                "No retries left."
                                )
                        #   end if
                        raise
                    #   end if attempt < max_retries
                except Exception as e:
                    # Any other error should stop the script
                    if not log :
                        print(f"Unexpected error: {e}")
                    else :
                        log.logging.info(
                            f'[{datetime.now():%Y-%m-%d %H:%M:%S}] '+
                            f"Unexpected error: {e}"
                            )
                    #   end if
                    raise
                #   end try/except
            #   end for
        #   end def
        return wrapper
    #   end def
    return decorator
####
def persistCache(filePath=None):
    '''
    Decorator that caches function calls and returns cached result when
    arguments are recognized.
    Decorator checks for existing cache

    Parameters
    ----------
    filePath : str, optional
        Define directory and file name for import and exporting cache. 
        The default is None.
    '''
    def decorator(func):
        # Create cache in memory
        cache = {}
        
        if filePath is not None:
            resolvedPath = Path(filePath)
            resolvedPath.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(resolvedPath, 'r') as f: cache = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                cache = {}
            #   end try/except
        #   end if

        def saveCache():
            if filePath is not None:
                with open(resolvedPath, 'w') as f: json.dump(cache, f)
            #   end if
        #   end def

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str((func.__name__, args, tuple(sorted(kwargs.items()))))
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            return result
        #   end def

        wrapper.saveCache = saveCache  # expose for manual call
        return wrapper

    return decorator
####
# =========================================================================== #





# ================================ CLASS-OBJ ================================ #
class logger :
    def __init__ (
            self,
            writeTo = 'log/'
            ) :
        # Import library
        import logging
        
        # Make logging dir if it doesn't already exist
        Path(writeTo).mkdir(parents=True,exist_ok=True)
        
        # Establish basic log config regular memory dumps
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(
                    f'{writeTo}{pd.Timestamp("today"):%Y-%m-%d %H_%M}.txt'),
                logging.StreamHandler()  # also prints to console
            ]
        )
        
        # Create logging object
        self.log = logging.getLogger()
        
    #   end __init__
    #
    def print(self, *args, sep=' '):
        # Print text to user, save to log, dump memory
        message = sep.join(str(a) for a in args)
        self.log.info(
            f'[{pd.Timestamp("today"):%Y-%m-%d %H:%M:%S}] ' + message
        )
    #   end print
    #
#   end logger
####
class CannedSoup :
    
    def __init__ (self, **kwargs) :
        # Import dependents
        
        
        # Attach arguments as methods
        self.__attach_args(kwargs)
        
    #	end init
    
    
    
    def __attach_args (self, kwargs) :
        if 'url' in kwargs: self.url = kwargs['url']
        elif not hasattr(self, 'url'):
            raise ValueError("url must be provided")
        #   end if
        
        self.timeout = kwargs.get(
            'timeout', 
            getattr(self, 'timeout', 10)
            ) # seconds
        
        self.verbose = kwargs.get(
            'verbose', 
            getattr(self, 'verbose', True)
            ) # message to user
        
        
        self.header = kwargs.get(
            'header',
            getattr(self, 'header',
                    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                     "Accept-Language": "en-US,en;q=0.9",
                     "Accept": "text/html,application/xhtml+xml",
                     "Referer": "https://www.google.com/"
                    }
                )
            ) # header info for attempt 2
        
        
        self.retries = kwargs.get(
            'retries',
            getattr(self, 'retries', 3)
            ) # number of retries on failed request
        
        self.baseDelay = kwargs.get(
            'baseDelay',
            getattr(self, 'baseDelay', 5)
            ) # number of retries on failed request
        
    #	end __attach_args
    
    
    def _attempt_request (
        self,
        getFn, # function to be attempted
        ) :
        '''
        getFn : Callable that returns a requests.Response
        '''
        # Import dependents
        import time
        import requests
        
        # Attempt functions
        for attempt in range(self.retries + 1) : 
            try :
                response = getFn()
                status   = self.__classify_response(response)
                
                if status == 'ok' : return response.text
                
                elif status == 'blocked' : 
                    raise PermissionError(f'Blocked ({response.status_code})')
                    
                
                elif status in ['rate_limited', 'server_error'] :
                    if attempt < self.retries :
                        delay = self.baseDelay * (2 ** attempt)
                        if self.verbose :
                            print(f"Retryable error ({response.status_code}), sleeping {delay}s")
                        #   end if
                        time.sleep(delay)
                        continue
                    else : 
                        raise Exception(f'Retries exhausted ({response.status_code})')
                    #   end if
                
                else : 
                    raise Exception(f'Unhandled status: {response.status_code}')
            
            except requests.exceptions.RequestException as e :
                # Network issues => Treat as transient
                if attempt < self.retries :
                    delay = self.baseDelay * (2 ** attempt)
                    if self.verbose :
                        print(f'Network error, retrying in {delay}s: {e}')
                    time.sleep(delay)
                #   end if
                
            #   end try/except
            
        #   end for attempt
        # raise Exception('Unreachable')
        
    #	end _attempt_request
    

    def __classify_response (
            self,
            response
            ) :
        code = response.status_code
	
        if code==200 : return 'ok'
        elif code in [401, 403] : return 'blocked'
        elif code == 429 : return 'rate_limited'
        elif 500 <= code < 600 : return 'server_error'
        elif 400 <= code < 500 : return 'client_error'
        else : return 'unknown'
	
	#	end classift response

    
    def simple_request (self) :
        import requests
        
        return self._attempt_request(
            lambda : requests.get(self.url, timeout=self.timeout)
            )
    #   end simple_request
    
    
    def session_request (self) :
        import requests
        
        session = requests.Session()
        session.headers.update(self.header)
        
        return self._attempt_request(
            lambda : session.get(
                self.url, timeout=self.timeout)
            )
    #   end session_request
    
    
    def cloud_request (self) :
        import cloudscraper
        
        scraper = cloudscraper.create_scraper()
        
        return self._attempt_request(
            lambda : scraper.get(
                self.url, timeout=self.timeout)
            )
    #   end cloud_request
    
    
    def selenium_request (self) :
        #import time
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        options = Options()
        options.add_argument('--headless=new')
        
        driver = webdriver.Chrome(options=options)
        driver.get(self.url)
        time.sleep(5)
        
        html = driver.page_source
        driver.quit()
        
        return html
    #   end selenium_request


    def Can (self, **kwargs) :
        
        # 1st Attempt: Simple Request
        try :
            if self.verbose : print('1st Attempt: Simple Request')
            self.html = self.simple_request()
            
        except PermissionError :
            if self.verbose : print('Attempt failed, escalating...')
        # end
        
        

		# 2nd Attempt: With Headers + Session
        try :
            if self.verbose : print('2nd Attempt: Session & Headers')
            self.html = self.session_request()
            
        except PermissionError :
            if self.verbose : print('Escalating from Stage 2')
        # end
        
        
        
        # 3rd Attempt: cloudscraper
        try :
            if self.verbose : print('3rd Attempt: cloudscraper')
            self.html = self.cloud_request()
            
        except PermissionError :
            if self.verbose : print('Escalating to Stage 4')
        # end
        

        
        # 4th Attempt: Selenium
        try :
            if self.verbose : print('4th Attempt: Selenium')
            self.html = self._attempt_request(
                self.selenium_request()
                )
            
        except Exception as e :
            raise Exception(f'Selenium failed: {e}')
        

        # All Attempts failed
        raise Exception('All stages failed :(')

	#	end def
    can = CAN = RUN = Run = run = Can
	#	end alias

####
# =========================================================================== #





# ================================ FUNCTIONS ================================ #
def MichelTimestamp () :
    ''' Return today's date as Month Day Year str with spaces between entries
    '''
    return ('{:02d}'.format(pd.Timestamp.today().month)+' '+
            '{:02d}'.format(pd.Timestamp.today().day)+' '+
            '{:02d}'.format(pd.Timestamp.today().year)
            )
####
def bls_get (dictData={'Name':'seriesID'}) :
    # dictData assumes {'Name','seriesID'}
    strURL = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'
    page = post(strURL,
                data=json.dumps({'seriesid':list(dictData.values()),
                                 'startyear':'2016','endyear':'2022'
                                 }),
                headers={'Content-type':'application/json'}
                )
    dataJSON = json.loads(page.text)
    dfOut  = pd.DataFrame()
    for dictI in dataJSON['Results']['series'] :
        df = pd.DataFrame()
        strName = list(dictData.keys()
                       )[list(dictData.values()).index(dictI['seriesID'])]
        
        for I in dictI['data'] :
            df = df.append(I,ignore_index=True)
        #   endfor
        
        df['month'] = df['period'].apply(lambda x : int(x[1:]))
        df = df[['year','month','value']]
        df.rename(columns={'value':strName},inplace=True)
        
        if len(dfOut)==0 : dfOut = df.copy()
        else : dfOut = dfOut.merge(df,on=['year','month'],how='outer')
    #   endfor
    dfOut['ts']    = dfOut.apply(lambda x : pd.Timestamp(str(x['year'])+'-'+
                                                      str(x['month'])+'-15'
                                                      ),axis=1)
    dfOut = dfOut.set_index('ts')
    
    return dfOut
####
def get_data (fred,listFred,try_limit=5) :
    # listFred assumes list of tuples, listFred = [col name, seriesID]
    # Copy-paste below:
    # dfData = get_data(fred,
    #                   [tuple(x) for x in 
    #                    dfSource[['Name','SeriesID']][(~dfSource['Name'].isna())&
    #                                                  (~dfSource['SeriesID'].isna())
    #                                                  ].to_numpy()
    #                    ])
    dfData = pd.DataFrame()
    for tpl in listFred :
        bContinue       = 0
        intErrorCount   = 0
        
        if isna(tpl[1]) : continue
        
        while (bContinue==0)&(intErrorCount<try_limit) : 
            try :
                data = pd.DataFrame(fred.get_series(tpl[1])
                                 ).rename(columns={0:tpl[0]})
                data.index.name = 'date'
            except : 
                try: 
                    htmlPage = dlURL('https://fred.stlouisfed.org/data/'+
                                     tpl[1]+'.txt')
                    
                    listRows = htmlPage.text.split('\n')
                    listRows = listRows[listRows.index([x for x in listRows 
                                                        if 'DATE' in x][0])+1:]
                    listRows = [[to_datetime(x[:x.index(' ')]),
                                 float(isolate_better(x,' ','\r',b_end=1))
                                 ] 
                                for x in listRows if x!=''
                                ]
                    
                    data = pd.DataFrame(listRows,columns=['index',tpl[0]]
                                     ).set_index('index')
                    data.index.name = 'date'
                except : 
                    intErrorCount+=1
                    sleep(1)
                else : bContinue = 1
                #   endtry
            else : bContinue = 1
            #   endtry
        #   endwhile
        
        if intErrorCount>=try_limit :
            print('Failure in accessing data from:\n'+
                  'Name: '+tpl[0]+'\n'+
                  'ID:   '+tpl[1]+'\n'
                  )
        else :
            if len(dfData)==0 : dfData = data
            else : dfData = dfData.join(data,how='outer',
                                         )
        #   endif
        sleep(.5)
    #   endfor
    return dfData.sort_index()
####
def genVarDictionary (
        dfFred,
        dictNames={'Name':'Name','seriesID':'seriesID'}
        ) :
    ''' 
    '''
    
    # Takes list of tuples, listFred = [col name, seriesID]
    strURL  = 'https://fred.stlouisfed.org/data/'
    strName = dictNames['Name']
    strID   = dictNames['seriesID']
    
    
    
    dfOut = pd.DataFrame()
    for index,row in dfFred.iterrows() :
        dictRow = {strName  :row[strName],
                   strID    :row[strID]
                   }
        print('Starting: '+dictRow[strName]+' | '+dictRow[strID])
        
        
        htmlPage = dlURL(strURL+row[strID]+'.txt')
        listText = htmlPage.text.split('\n')
        listText = listText[:listText.index([x for x in listText 
                                             if 'DATE' in x][0])]
        
        
        
        #   For all rows except for notes
        listRow = ['Title','Source','Release','Seasonal Adjustment',
                   'Frequency','Units','Date Range','Last Updated'
                   ]
        for Row in listRow :
            if len([x for x in listText if Row+': ' in x])>0 :
                strTitle = [x for x in listText if Row+': ' in x][0]
                listText.remove(strTitle)
                strTitle = strTitle.replace('\r','').replace(Row+':','')
                while strTitle[0]==' ' : strTitle = strTitle[1:]
                dictRow[Row] = strTitle
            #   endif
        #   endfor
        
        
        
        #   Notes
        if len([x for x in listText if 'Notes: ' in x])>0 :
            listText = listText[listText.index([x for x in listText 
                                                if 'Notes: ' in x][0]):]
            listText = [x.replace('Notes: ','').replace('\r',' ') 
                        for x in listText]
            strEntry = ''
            for x in listText :
                if len(x.replace(' ' ,''))>0 :
                    while x[0]==' ' : x = x[1:]
                #   endif
                
                strEntry+=x
            #   endfor
            
            dictRow['Notes'] = strEntry
        #   endif
        
        dfOut = dfOut.append(dictRow,ignore_index=True)
        print('Row complete.\n\n')
    #   endfor
    
    listColsOrder = [strName,strID,'Title','Source','Release',
                     'Seasonal Adjustment','Frequency','Units','Date Range',
                     'Last Updated','Notes'
                     ]
    dfOut = dfOut[[col for col in listColsOrder if col in list(dfOut)]]
    return dfOut
####
# def dlURL (url , parser = "html.parser" ) :
#     urlClient = uReq(url)
#     pageRough = urlClient.read()
#     urlClient.close()
#     pageSoup = soup(pageRough,parser)

#     return pageSoup
# ####
def dlURL (url , parser = "html.parser" ) :
    req         = Request(url,headers={'User-Agent':'Mozilla/5.0'})
    urlClient   = urlopen(req)
    pageRough   = urlClient.read()
    urlClient.close()
    pageSoup    = soup(pageRough,parser)

    return pageSoup
####
def isolateBetter (stri , start , end, b_end = 0) :
    ''' ISOLATE-BETTER
        
    '''
    strShort    = ''
    posStart    = 0
    posEnd      = 0

    if b_end==1 :
        posEnd      = stri.find(end)
        strShort    = stri[:posEnd]
        strShort    = strShort[::-1]
        start       = start[::-1]
        posStart    = posEnd - strShort.find(start)
    #
    else :
        posStart    = stri.find(start)+len(start)
        strShort    = stri[posStart:]
        posEnd      = posStart + strShort.find(end)
    #
    return stri[posStart:posEnd]
####
def dict_to_excel (dictIN,fileName,Index=None) :
    #   Require ExcelWriter from pandas
    
    if fileName[-5:]!='.xlsx' : fileName+='.xlsx'
    
    with ExcelWriter(fileName,engine='xlsxwriter') as writer : 
        for key in dictIN.keys() :
            dictIN[key].to_excel(
                writer,
                index=Index, 
                sheet_name=key.replace('\n',' ').replace(':','-')
                )
        #   endfor
    #   endwith
####
def excel_to_dict (dirFile) :
    
    xlsxFile = ExcelFile(dirFile)
    dictOut = {}
    for sheet in xlsxFile.sheet_names :
        dictOut[sheet] = xlsxFile.parse(sheet)
    #   endfor
    
    return dictOut
####
def Linear_Outreg (inX,inY) :
    # Source: https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
    
    X = inX[(inY.notnull())&(~inX.isna().any(axis=1))]
    inY = pd.DataFrame(inY)
    #y = inY[(~inX.isna().any(axis=1))]
    y = inY.loc[X.index]
    
    assert len(y)==len(X)
    X = X.values.reshape(len(X),len(list(X)))
    y = y.values.reshape(len(y),1)
    
    lm = LinearRegression()
    lm.fit(X,y)
    params = append(lm.intercept_,lm.coef_)
    predictions = lm.predict(X)
    
    newX = pd.DataFrame({"Constant":ones(len(X))}).join(pd.DataFrame(X)).reset_index(drop=True)
    SST = sum((y-y.mean())**2)
    RSS = sum((y-predictions)**2)
    MSE = RSS/(len(newX)-len(newX.columns))
    R2  = 1 - (RSS/SST)
    aR2 = 1 - (1-R2)*((len(newX)-1)/(len(newX)-len(newX.columns)))
    
    # Note if you don't want to use a pd.DataFrame replace the two lines above with
    # newX = np.append(np.ones((len(X),1)), X, axis=1)
    # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))
    
    var_b = MSE*(linalg.inv(dot(newX.T,newX)).diagonal())
    sd_b = sqrt(var_b)
    ts_b = params/ sd_b
    
    p_values =[2*(1-stats.t.cdf(abs(i),(len(newX)-len(list(newX))))) for i in ts_b]
    
    sd_b     = np_round(sd_b,3)
    ts_b     = np_round(ts_b,3)
    p_values = np_round(p_values,4)
    params   = np_round(params,4)
    
    # I need to come back in replace this when I have time
    model2 = OLS(y,add_constant(X)).fit()
    
    
    myDF3 = pd.DataFrame()
    myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilities"] = [params,sd_b,ts_b,p_values]
    myDF3 = myDF3.set_index(Index(['Const']+list(pd.DataFrame(inX))))
    return {'model':myDF3, 'SSE' : SST[0], 'RSS' : RSS[0], 'MSE' : MSE[0],
            'R2' : R2[0] , 'adj-R2' : aR2[0], 'n' : len(X),
            'f-value' : model2.fvalue , 'f-prob' : model2.f_pvalue
            }
####
def month_to_quarter (x) :
    c = ''
    
    if      str(x) in ['1','2','3']     : c='Q1'
    elif    str(x) in ['4','5','6']     : c='Q2'
    elif    str(x) in ['7','8','9']     : c='Q3'
    elif    str(x) in ['10','11','12']  : c='Q4'
    else :  c=None

    return c
####
def plot_hist (df,min_bins=5,max_bins=35,xts=[],strName='',bSave=False) :
    rangeNorm = np.arange(-3.5,3.5,0.001)
    listBins = list(range(min_bins,max_bins+1))
    listErr = []
    
    mean,stdv,num = df.mean(),np.std(df),len(df)
    
    for n in listBins :
        dfFit = pd.DataFrame()
        heights,intervals = pd.histogram(df,bins=n)    
        listX = []
        for i in range(1,len(intervals)) :
            listX.append((intervals[i-1]+intervals[i])/2)
        #   endfor
        dfFit['x'] = listX
        dfFit['height'] = heights
        dfFit['norm'] = len(df)*norm.pdf((dfFit['x']-mean)/stdv,0,1)
        dfFit['err'] = (dfFit['height']-dfFit['norm'])**2
        listErr.append(dfFit['err'].mean())
    #   end of bins
    #bestBin = listBins[int(where(listErr==min(listErr))[0])]
    bestBin = listBins[listErr.index(min(listErr))]
    fig = plt.figure()
    plt.hist(df,bins=bestBin,edgecolor='white')
    plt.plot(rangeNorm*stdv+mean,len(df)*norm.pdf(rangeNorm,0,1))
    fig.text(.1,.03,'bins = '+str(bestBin)+'  n = '+str(num)
                     +'  mean = '+str(round(mean,3)))
    if len(xts)>0 : plt.xticks(xts)
    if len(strName)>0 : plt.title('Distribution of '+strName)
    if bSave : plt.savefig('../plot/Dist-of-'+strName+'.png')
####   
def b_round (x,i=0) :
    #   Better round -- for round<=0, returns int to remove zero decimal
    if i>0  :   x = round(x,i)
    else    :   x = int(round(x,i))
    return x
####
def sig_fig (x,n) :
    if x>0 : 
        n = max(n-math.floor(np.log10(x))-1,0)
        x = b_round(x,n)
    else : x=np.nan
    return x
####
# =========================================================================== #





# ================================ GEOPANDAS ================================ #
#   GeoPandas block
def enableGeoPandas () :
    import sys
    if sys.prefix[sys.prefix.rindex('\\')+1:] == 'geoSpyder' :
        sys.path.append('C:\\Users\\conlon\\Anaconda3\\Lib')
    #   endif
####
try : from geopandas import GeoDataFrame
except ImportError : pass
else :
    def gen_map (dfMap,X='',strFile=None,labelSize=80,borderWidth=5,
                 listMapX=[-127.5,-65],listMapY = [24,50],intMapScale=100,
                 greys=False,dfState=GeoDataFrame(),
                 colorEdge='#FFFFFF',colorWater='#A5DBF7'
                 ) :
        '''dfMap must have "colorFill", "colorFont", "pointCorr"
        
        '''    
        
        fltX = intMapScale*(listMapX[1]-listMapX[0])/(listMapY[1]-listMapY[0])
        fltY = intMapScale*(listMapY[1]-listMapY[0])/(listMapX[1]-listMapX[0])
    
    
        
        ####    MAP -- GDP by State YoY
        listColorMap = list(dfMap['colorFill'][dfMap['colorFill']!=''].unique())
        fig,ax = plt.subplots(1,1,figsize=(fltX,fltY),)
        plt.xlim(listMapX)
        plt.ylim(listMapY)
        plt.tick_params(axis='both',which='both',bottom=True,top=True,
                        labelbottom=False,right=True,left=True,labelleft=False)
        for mapColor in listColorMap :
            dfMap[(dfMap['colorFill']==mapColor)
                    ].plot(color=mapColor,
                            #edgecolor='white',
                            figsize=(fltX,fltY),
                            legend=True,
                            ax=ax,
                            )
        dfMap.boundary.plot(edgecolor=colorEdge,ax=ax,linewidth=borderWidth)    # Borders / white
        if len(dfState)>0 : 
            dfState.boundary.plot(edgecolor=colorEdge,ax=ax,linewidth=borderWidth*2)
        #   endif
        ax.set_facecolor(colorWater)                                            # Water / Light blue
        if len(X)>0 :
            if greys : mask = (dfMap['colorFill']!='#D0D0D3') | True
            else     : mask =  dfMap['colorFill']!='#D0D0D3'
                
            for index,row in dfMap[mask].iterrows():       # Annotate if not filled / Light grey
                # txt = ax.text(row['pointCorr'][0],row['pointCorr'][1],
                #               row[X],
                #               size=labelSize,
                #               color=row['colorFont'],
                #               path_effects=[pe.withStroke(linewidth=4, foreground='black')])
        
                plt.annotate(text=row[X],
                              xy=row['pointCorr'],
                              horizontalalignment='center',
                              fontsize=labelSize,
                              fontname='Calibri',
                              #fontweight='semibold',
                              color=row['colorFont']
                              )
        if strFile!=None : plt.savefig(strFile+'.jpg',bbox_inches='tight')
    ####
    def get_counties () :
        strDir = 'C:/Users/conlon/Anaconda3/Lib'
        dfCounties = pd.DataFrame()
        
        dfCounties = pd.read_csv(strDir+'/mapCounties.csv')
        dfCounties['geometry'] = dfCounties['geometry'].apply(wkt.loads)
        
        return dfCounties
    ####
    def make_state_maps (dfCounties) :
        #   Combine counties to state; Move AK, HI; Get center points;
        #   Shift points, Replace some points; 
        #   Set default colorFont, pointLabel, colorFill
        
        #   Create state boundaries in geodf
        dfStates = pd.DataFrame()
        for state in list(dfCounties['abrState'][dfCounties['abrState'].notnull()].unique()) :
            dfStates = pd.concat([dfStates,
                                 pd.DataFrame({'abrState':[state],
                                               'geometry':[cascaded_union(dfCounties['geometry'][dfCounties['abrState']==state])]
                                               }
                                              )
                                  ]
                                 )
        #   end for
        mask = list(set(dfCounties.columns).intersection(['abrState','state']))
        dfStates = dfStates.merge(dfCounties[mask],
                                  how='left',on=['abrState'])
        dfStates = GeoDataFrame(dfStates[~dfStates['geometry'].duplicated(keep='first')]
                                ,geometry='geometry')
        
        
        
        
        
        ####    Move Alaska and Hawaii where they can be seen
        listShift = [('AK',    -84,    -34,    1/4,    1/2),
                     ('HI',     45,      7,    1.1,    1.1)
                     ]
        for tpl in listShift :
            dfStates['geometry'] = dfStates[['abrState','geometry']
                                            ].apply(lambda x :
                                                    scale(x[1],
                                                              tpl[3],
                                                              tpl[4])
                                                    if x[0]==tpl[0]
                                                    else x[1]
                                                    ,
                                                    axis=1)
            dfStates['geometry'] = dfStates[['abrState','geometry']
                                            ].apply(lambda x :
                                                    translate(x[1],
                                                              tpl[1],
                                                              tpl[2])
                                                    if x[0]==tpl[0]
                                                    else x[1]
                                                    ,
                                                    axis=1)
            
        #   endfor
        
        
        
        
        
        ####    Get center points
        dfStates['pointCorr'] = dfStates['geometry'].apply(lambda x : x.representative_point().coords[:]
                                                           if not bool((type(x)==type(None)))
                                                           else None
                                                           )
        dfStates['pointCorr'] = [coords[0] for coords in dfStates['pointCorr']]
        
        
        
        
        
        
        
        
        
        
        ####    IMPORT COLOR MAPPING DATA
        ####    Shift Label Point
        listTrans = [#  WEST
                     ('ID',     1,      -2),
                     
                     
                     #  SOUTH
                     ('AL',     0,      -1),
                     ('LA',    -0.5,    0.5),
                     ('FL',     2.5,     0),
                     
                     #  MID WEST
                     ('MI',     2,      -3),
                     ('WV',     0,      -1),
                     ('TN',     0,      -0.5),
                     ('KY',     0,      -0.5),
                     
                     #  NORTH EAST
                     ('VA',     0,      -1),
                     ('PA',     0,      -0.5),
                     ('NY',     0,      -0.5),
                     ('VT',     0,       1.5),
                     ('ME',     0,      -0.5),
                     ('WV',    -1,       0.5)
                     ]
        
        for tpl in listTrans :
            element1 = dfStates['pointCorr'][dfStates['abrState']==tpl[0]
                                             ].values[0][0]+tpl[1]
            element2 = dfStates['pointCorr'][dfStates['abrState']==tpl[0]
                                             ].values[0][1]+tpl[2]
            
            dfStates.at[dfStates[dfStates['abrState']==tpl[0]].index[0],
                         'pointCorr'
                         ] = (element1,element2)
        #   endfor
        
        dfStates['colorFont']       = '#FFFFFF' # Font / White
        dfStates['pointLabel']      = ''
        dfStates['colorFill']       = '#D0D0D3' # No fill / Light grey
        
        listReplace = [#  NORTH EAST COAST
                       ('NH',    -68,     42.5),
                       ('MA',    -68.5,     40),
                       ('RI',    -68.5,     39),
                       ('CT',    -68.5,     38),
                       ('NJ',    -68.5,     37),
                       ('DE',    -68.5,     36),
                       ('MD',    -68.5,     35),
                       ('DC',    -68.5,     34),
                       ('HI',    -113,      26)
                       ]
        for tpl in listReplace :
            dfStates.at[dfStates[dfStates['abrState']==tpl[0]].index[0],
                         'pointCorr'
                         ] = (tpl[1],tpl[2])
        #   endfor
        
        return dfStates
    ####
    def make_region_maps (dfStates) :
        ''' Must have 'abrState', generates regional category
        '''
        #   Combine states to Region
        listNE = ['CT','ME','MA','NH','NJ','NY','PA','RI','VT']
        listMW = ['IL','IN','IA','KS','MI','MN','MO','NE','ND','OH','SD','WI']
        listS  = ['AL','AR','DE','DC','FL','GA','KY','LA','MD','MS','NC','OK',
                  'SC','TN','TX','VA','WV'
                  ]
        listW  = ['AK','AZ','CA','CO','HI','ID','MT','NV','NM','OR','UT','WA',
                  'WY'
                  ]
        
        dfStates['region'] = ''
        for tpl in [('Northeast',listNE),('Midwest',listMW),('South',listS),
                    ('West',listW)]  :
            dfStates.loc[(dfStates['abrState'].isin(tpl[1]))
                         ,'region'] = tpl[0]
        #   endfor
        
        #   Create state boundaries in geodf
        dfRegion = pd.DataFrame()
        for region in list(dfStates['region'][dfStates['region'].notnull()].unique()) :
            dfRegion = pd.concat([dfRegion,
                               pd.DataFrame({'region':[region],
                                          'geometry':[cascaded_union(dfStates['geometry'][dfStates['region']==region])]
                                          }
                                         )
                               ]
                              )
        #   end for
        dfRegion = GeoDataFrame(dfRegion[~dfRegion['geometry'].duplicated(keep='first')]
                                ,geometry='geometry')
        
        
        
        ####    Get center points
        dfRegion['pointCorr'] = dfRegion['geometry'].apply(lambda x : x.representative_point().coords[:]
                                                       if not bool((type(x)==type(None)))
                                                       else None
                                                       )
        dfRegion['pointCorr'] = [coords[0] for coords in dfRegion['pointCorr']]
        
        dfRegion['colorFont']       = '#FFFFFF' # Font / White
        dfRegion['pointLabel']      = ''
        dfRegion['colorFill']       = '#D0D0D3' # No fill / Light grey
        
        return dfRegion
    ####
#   end GeoPandas block

# =========================================================================== #





# =================================== MAIN ================================== #
if __name__ == "__main__" :
    print(__doc__)
        
    pass
#   endif
# =========================================================================== #



