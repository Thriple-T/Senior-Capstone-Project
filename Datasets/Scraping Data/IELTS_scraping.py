import time
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 1. Set up the Chrome Browser
options = webdriver.ChromeOptions()
# options.add_argument('--headless') # Uncomment to run in background
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

target_url = "https://writing9.com/ielts-writing-samples"

'''Other URLS
ACADEMIC TASK1: https://writing9.com/ielts-academic-writing-samples-task-1
GENERAL TASK1: https://writing9.com/ielts-writing-samples-task-1
TASK2: https://writing9.com/ielts-writing-samples
'''

driver.get(target_url)
time.sleep(3) 

print("Scrolling to gather essay links...")

# Infinite Scroll Logic
essay_links = set()
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    links = driver.find_elements(By.XPATH, "//a[contains(@href, '/text/')]") 
    for link in links:
        href = link.get_attribute('href')
        if href:
            essay_links.add(href)
    
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
    
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

print(f"Found {len(essay_links)} essays. Starting scraping...")

# Scrape Individual Essays
scraped_data = []

def extract_score(label_pattern, text):
    # Regex to find a decimal like 7.0 or 6.5 near the label
    pattern = f"{label_pattern}[^0-9]{{0,30}}?([0-9](?:\\.[0-5])?)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else None

for url in list(essay_links):
    try:
        driver.get(url)
        # Wait for the main content
        WebDriverWait(driver, 7).until(
            EC.presence_of_element_located((By.TAG_NAME, 'h1'))
        )
        
        # Usually the H1 title of the page
        question = driver.find_element(By.TAG_NAME, 'h1').text
        
        # Target the article body
        essay_element = driver.find_element(By.XPATH, '//div[@itemprop="articleBody"]')
        full_essay_text = essay_element.text
        
        # Scores: Get the text of the entire page to find scores anywhere
        page_text = driver.find_element(By.TAG_NAME, 'body').text
        
        # We look for "Estimated Band", "Overall", or "Band Score" to get the total
        overall = extract_score(r'(?:Estimated\s*Band|Overall|Band\s*Score)', page_text)
        
        # Individual Criteria
        ta = extract_score(r'Task\s*(?:Achievement|Response)', page_text)
        cc = extract_score(r'Coherence', page_text)
        lr = extract_score(r'Lexical', page_text)
        gra = extract_score(r'Grammatical', page_text)

        scraped_data.append({
            "Question": question,
            "Essay_Text": full_essay_text,
            "Overall_Score": overall,
            "Task_Achievement": ta,
            "Coherence_Cohesion": cc,
            "Lexical_Resource": lr,
            "Grammar_Range": gra
        })
        
        print(f"Scraped: {question[:30]}... | Score: {overall}")

    except Exception as e:
        print(f"Error on {url}: {e}")
    
    time.sleep(1)

#Save to CSV
df = pd.DataFrame(scraped_data)
df.to_csv("ielts_task2.csv", index=False, encoding='utf-8-sig')

print("\nSuccess!")
driver.quit()