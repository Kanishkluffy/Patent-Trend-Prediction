from selenium import webdriver

# For using sleep function because selenium 
# works only when the all the elements of the 
# page is loaded.
import time 
import pathlib
from selenium import webdriver
from selenium.webdriver.common.by import By

from selenium.webdriver import Firefox
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options

USER_DATA_DIR: pathlib.Path = pathlib.Path('C:/Users/ktadh/AppData/Local/Google/Chrome/User Data/Default')
DESTINATION_URL: str = 'file:///C:/Users/ktadh/Downloads/US%202006_0015372%20A1%20-%20System%20and%20method%20of%20coordinating%20medical%20screening%20and%20treatment%20data%20_%20The%20Lens.html'

def getChromeDriver(userDataDir:pathlib.Path, url: str)->webdriver.FirefoxProfile:
    # Creating an instance webdriver
    #options = webdriver.ChromeOptions()
    #options.add_argument(f"user-data-dir={userDataDir}")
    # driver = webdriver.Chrome(executable_path=r'C:\path\to\chromedriver.exe', chrome_options=options)
    #driver = webdriver.Chrome(options=options)
    #driver.get(url)
    #return driver



    profile_path = r'C:/Users/ktadh/AppData/Local/Mozilla/Firefox/Profiles/n1u95a6w.default-release'
    options=Options()
    options.set_preference('profile', profile_path)
    options.set_preference("javascript.enabled", False)
    service = Service(r'geckodriver-v0.35.0-win64/geckodriver.exe')
    driver = Firefox(options=options)
    driver.get(DESTINATION_URL)
    return driver

def main()->None:
    pass


if __name__ == "__main__":
    driver = getChromeDriver(USER_DATA_DIR, DESTINATION_URL)
    # Let's the user see and also load the element 
    #login = driver.find_element(By.CLASS_NAME, '...')

    login = driver.find_element(By.XPATH, "//section[contains(@class, 'lf-main-content-panel')]")
    product_name = login.text
    print(product_name)
    # closing the browser
    driver.close()