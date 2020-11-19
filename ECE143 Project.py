import pandas as pd
a=pd.read_json('drive/MyDrive/Yelp_dataset_checkin_clean.json')
b = pd.read_json('drive/MyDrive/Yelp_dataset_review_clean.json')
time = b['date']
month_yr = {}
for index,value in time.items():
  temp = value.strftime('%Y/%m')
  if temp in month_yr:
    month_yr[temp]+=1
  else:
    month_yr[temp]=1
