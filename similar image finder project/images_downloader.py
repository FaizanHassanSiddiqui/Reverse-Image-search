
import requests
import os
from bs4 import BeautifulSoup

import json


site = 'https://www.jumia.com.ng/watches-sunglasses/'


if  'jumia' in site:
    website_name = 'jumia'
elif 'olist' in site:
    website_name = 'olist'
elif 'payporte' in site:
    website_name = 'payporte'
elif 'cars45' in site:
    website_name = 'cars45'
elif 'jiji' in site:
    website_name = 'jiji'
elif 'ebay' in site:
    website_name = 'ebay'
elif 'konga' in site:
    website_name = 'konga'
else:
    print(site)
    website_name = 'others'

print(website_name)

response = requests.get(site)


soup = BeautifulSoup(response.text, 'html.parser')


img_tags = soup.find_all('img')

if not img_tags:
    print("no image tags found")

else:
    urls = []
    for img in img_tags:
        try:
            
            urls.append(img['data-src'])
            print("used data-src key")
        except:
            try:
                
                urls.append(img['src'])
                print("used src key")
            except:
                
                print("falls outside! nothing appended and thus downloaded. Try ammending the source key.")



    folder = 'images_base'

    try:
        with open('name_to_url.json') as f_in:
            name_to_url = json.load(f_in)

        last_filename = list(name_to_url.keys())[-1]
        last_pic_number = int(last_filename[ : last_filename.find("_")])

        
    except:
        name_to_url = {}

        
        last_pic_number = 0

    urls = list(set(urls) - set(name_to_url.values()))

    if urls:
        for index, url in enumerate(urls):

            file_name =   f'{index + last_pic_number + 1 }_{website_name}.jpg' 
            name_to_url[file_name] = url
            with open(os.path.join(folder ,file_name) , 'wb') as f:
                if 'http' not in url:
                    # sometimes an image source can be relative 
                    # if it is provide the base url which also happens 
                    # to be the site variable atm. 
                    url = '{}{}'.format(site, url)
                response = requests.get(url)
                f.write(response.content)

        with open('name_to_url.json', 'w') as fp:
            json.dump(name_to_url, fp)