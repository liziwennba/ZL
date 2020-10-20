import calendar
def number_of_days(year,month):
    '''
    Return the number of calendar days in a given year and month
    :param year: An integer of year
    :param month: A month from 1-12
    :return: Number of days within that month
    '''
    assert isinstance(year,int) and isinstance(month,int)
    assert month>=1 and month<=12
    return calendar.monthrange(year, month)[1]

def number_of_leap_years(year1,year2):
    '''
    Return the number of leap years between the two given year
    :param year1: The first year
    :param year2: The second year
    :return: The number of leap years between the two given year
    '''
    assert isinstance(year1,int) and isinstance(year2,int)
    if year2>year1:
        return calendar.leapdays(year1, year2+1)
    else:
        return calendar.leapdays(year2,year1+1)


def get_day_of_week(year,month,day):
    '''
    Return the string name of the given day
    :param year: The given year
    :param month: The given month
    :param day: The given day
    :return: The string name of week
    '''
    assert isinstance(year,int) and isinstance(month,int) and isinstance(day,int)
    assert month>=1 and month<=12
    assert day>=1 and day<=calendar.monthrange(year, month)[1]
    day_name=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    return day_name[calendar.weekday(year, month, day)]
