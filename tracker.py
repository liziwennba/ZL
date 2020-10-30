from time import sleep
import random
from datetime import datetime
import itertools as it


def producer():
    'produce timestamps'
    starttime = datetime.now()
    while True:
        sleep(random.uniform(0, 0.2))
        yield datetime.now() - starttime


def tracker(p,limit):
    '''
    Track the producer and return the number of odd number seconds
    :param p: The producer
    :param limit: The number of limit
    :return: The result of whether a producer returns an odd number of seconds
    '''
    assert isinstance(limit,int) and limit>=0
    assert isinstance(p,type(producer()))
    n=0
    while n<limit:
       a = next(p).seconds
       if a%2==1:
           n=n+1
       new_limit = yield n
       if new_limit:
           assert isinstance(new_limit,int) and new_limit>=0
           limit = new_limit
