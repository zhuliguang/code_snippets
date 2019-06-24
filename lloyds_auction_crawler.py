#%%
from bs4 import BeautifulSoup as bs
import requests,csv,re,logging
from urllib.parse import urljoin
from datetime import datetime
import shutil
import pandas as pd


# logging.Logger.setLevel(logging.INFO)

domain = "https://www.lloydsonline.com.au/"

#%%
url = domain + "AuctionLots.aspx?smode=0&aid=9713&pgn={page_number}&pgs=10"
all_data = []
for page_number in range(1,12):
    # page_number += 1
    cur_url = url.format(page_number=page_number)
    print("Fetching: ["+cur_url+"] ...")
    response = requests.get(cur_url, proxies=proxies)
    html = bs(response.text, "html5lib")

    lot_list = html.findAll("div",{"class":"gallery_item lot_list_item"})
    

    for lot in lot_list:
        lot_number = lot.find('div',{'class':'lot_num'}).text.strip()
        lot_desc = lot.find('div',{'class':'lot_desc'}).h1.text.strip()
        lot_price = lot.find('div',{"class":"lot_cur_bid"}).text.strip()
        print(lot_price)
        lot_img_url = lot.find('img').get('src').split("?")[0]
        img = requests.get(lot_img_url, stream=True, proxies=proxies)

        img_file = "lloyds/{}.jpg".format(lot_number)
        with open(img_file, 'wb') as out_file:
            shutil.copyfileobj(img.raw, out_file)
        del img
        # all_data.append([lot_number, lot_desc, lot_price, lot_img_url])
        all_data.append([lot_number, lot_desc, lot_price, img_file])

data = pd.DataFrame(all_data)
data.columns=["Lot_number","Description","Price","Image"]

data.to_csv("lloyds.csv",index=False)

        



    




