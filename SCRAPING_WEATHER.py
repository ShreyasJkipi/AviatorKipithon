from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import pandas as pd
import time
from bs4 import BeautifulSoup as bs
from selenium.common.exceptions import NoSuchElementException
import re, json
from datetime import date, timedelta, datetime
import boto3


aws_access_key_id = 'AKIA3N5B4SUIGZVBVH4T'
aws_secret_access_key = '7lpHK5tvdMNtrjTC2cXrD82CZ4AN6bJZv5TLMTYL'
bucket_name = 'aviatorkipithon'
# Create a session using your AWS credentials
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Create an S3 client
s3_client = session.client('s3')


# TODO implement
optionss = webdriver.ChromeOptions() 
optionss.add_argument("start-maximized")
optionss.add_experimental_option("excludeSwitches", ["enable-automation"])
optionss.add_experimental_option('useAutomationExtension', False)
#driver = webdriver.Chrome(ChromeDriverManager().install(), options=optionss)

cService = webdriver.ChromeService(executable_path="C:\\Users\\ShreyasJ\\Desktop\\Kipithon\\chromedriver.exe")
driver = webdriver.Chrome(service=cService,options=optionss)

# Read url Excel file.
file = pd.read_excel('Cityandcode.xlsx')
cities = file.iloc[:, 0].tolist()
city_code = file.iloc[:, 1].tolist()

months = [['july','2024'],['august','2024'],['september','2024'],['october','2024'],['november','2024'],['december','2024'],['january','2025'],['february','2025'],['march','2025'],['april','2025'],['may','2025'],['june','2025']]

for indx, city in enumerate(cities):
    listOfDict = list()
    for time_indx, month in enumerate(months):
        #"https://www.accuweather.com/en/in/mumbai/204842/july-weather/204842?year=2024"
        updateurl = 'https://www.accuweather.com/en/in/' + city + '/' + str(city_code[indx]) + '/' + month[0] + '-' + 'weather' + '/' + str(city_code[indx]) + '?year=' + month[1]
        print(updateurl)
        driver.get(updateurl)
        time.sleep(4)
        soup = bs(driver.page_source, 'html.parser')
    # if soup.find_all('p')[0].getText() == "Please confirm that you are a real KAYAK user.":
        #   print("Kayak thinks I'm a bot, which I am ... so Solve it manually and then Enter 1")
        #  input('Please Enter 1')

        while True:
            try:
                # sleep for 3 seconds to load page
                time.sleep(3)
                # click to show more results
                driver.find_element(By.XPATH, '//div[contains(text(), "Show more results")]').click()
            except NoSuchElementException:
                break

        # Page source Data
        soup = bs(driver.page_source, 'html.parser')
        blocks = soup.select('a.monthly-daypanel')
        
        # Extract all data of block
        for block in blocks:
            try:
                dataDict = dict()
                dataDict['City'] = city
                dataDict['Month'] = month[0] 
                dataDict['Date'] = block.select_one('div[class="date"]').text.strip()
                dataDict['Maximum_Temperature'] = block.select_one('div.temp').select_one('div.high').text.strip()
                dataDict['Minimum_Temperature'] = block.select_one('div.temp').select_one('div.low').text.strip()
                try:
                    dataDict['Forecast'] = block.find('svg').get('alt').strip()
                except:
                    pass
                listOfDict.append(dataDict)
            except:
                pass

# Save Data in JSON file
   
    json_file_name = "Weather-" + city + ".json"
    with open(json_file_name, "w") as json_file:
        json.dump(listOfDict, json_file)

    s3_client.upload_file(json_file_name, bucket_name, 'weatherdata/{}'.format(json_file_name))

    



""""
# AWS credentials and S3 bucket information
aws_access_key_id = 'AKIA3N5B4SUIGZVBVH4T'
aws_secret_access_key = '7lpHK5tvdMNtrjTC2cXrD82CZ4AN6bJZv5TLMTYL'
bucket_name = 'aviatorkipithon'

# Create a session using your AWS credentials
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Create an S3 client
s3_client = session.client('s3')

# Upload the file to the S3 bucket
s3_client.upload_file('weatherdata/Weather-ahmedabad.json', bucket_name, 'Weather-ahmedabad.json')
"""