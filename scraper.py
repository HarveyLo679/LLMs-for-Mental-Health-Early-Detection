from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import csv
import time
import os
from datetime import datetime, timedelta
import re
import json

def format_text(text):
    return text.strip().replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ')

def normalize_date(text):
    text = str(text).strip()
    today = datetime.today()
    
    try:
        return datetime.strptime(text, "%d-%m-%Y %I:%M %p").strftime("%Y-%m-%d")
    except:
        pass

    try:
        return datetime.strptime(text, "%d-%m-%Y").strftime("%Y-%m-%d")
    except:
        pass
    text = text.lower()
    
    if "yesterday" in text:
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    elif "a week ago" in text:
        return (today - timedelta(weeks=1)).strftime("%Y-%m-%d")
    elif "2 weeks ago" in text:
        return (today - timedelta(weeks=2)).strftime("%Y-%m-%d")
    elif "3 weeks ago" in text:
        return (today - timedelta(weeks=3)).strftime("%Y-%m-%d")
    elif "4 weeks ago" in text:
        return (today - timedelta(weeks=4)).strftime("%Y-%m-%d")
    elif "a month ago" in text:
        return (today - timedelta(days=30)).strftime("%Y-%m-%d")
    
    match = re.match(r"(\d+)\s+(day|week|hour)s?\s+ago", text)
    if match:
        num = int(match.group(1))
        unit = match.group(2)
        if unit == "day":
            return (today - timedelta(days=num)).strftime("%Y-%m-%d")
        elif unit == "week":
            return (today - timedelta(weeks=num)).strftime("%Y-%m-%d")
        elif unit == "hour":
            return (today - timedelta(hours=num)).strftime("%Y-%m-%d")

    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    if text in weekdays:
        days_ago = (today.weekday() - weekdays.index(text)) % 7
        if days_ago == 0:
            days_ago = 7
        return (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

    return "Unknown"

def scrape_post_and_comments(driver, post_url, max_comments=40):
    comments = []
    page_num = 1
    post_text = None
    comment_index = 1  # Start numbering from 1 (first comment after main post)

    while len(comments) < max_comments:
        driver.get(post_url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        message_blocks = soup.find_all('div', class_='lia-panel-message')

        if not message_blocks:
            break

        for idx, block in enumerate(message_blocks):
            if page_num > 1 and idx == 0:
                continue
            try:
                # Extract comment text
                text_tag = block.find('div', class_='lia-message-body-content')
                comment_text = format_text(text_tag.get_text()) if text_tag else ""

                # Extract author
                author_tag = block.find('a', class_='lia-user-name-link')
                author = author_tag.get_text(strip=True) if author_tag else "Unknown"

                # Extract timestamp
                timestamp = "Unknown"

                # Try new-style timestamp
                time_tag = block.find('span', class_='local-date')
                if time_tag:
                    timestamp_raw = time_tag.text.replace("\u200e", "").strip()
                    try:
                        timestamp = datetime.strptime(timestamp_raw, "%d-%m-%Y").strftime("%Y-%m-%d")
                    except Exception:
                        timestamp = timestamp_raw

                # Try old-style timestamp if still unknown
                if timestamp == "Unknown":
                    time_tag = block.find('span', class_='local-friendly-date')
                    if time_tag and time_tag.has_attr("title"):
                        timestamp_raw = time_tag["title"].replace("\u200e", "").strip()
                        try:
                            timestamp = datetime.strptime(timestamp_raw, "%d-%m-%Y %I:%M %p").strftime("%Y-%m-%d")
                        except Exception:
                            timestamp = timestamp_raw

                if page_num == 1 and idx == 0:
                    post_text = comment_text  # main post
                else:
                    comments.append({
                        "comment_id": f"{comment_index}",
                        "author": author,
                        "timestamp": timestamp,
                        "comment": comment_text
                    })
                    comment_index += 1

                if len(comments) >= max_comments:
                    break

            except Exception as e:
                print(f"Comment parse failed: {e}")
                continue

        # Go to next page of comments if available
        next_link = soup.find("a", rel="next")
        if not next_link:
            break
        href = next_link.get("href")
        post_url = href if href.startswith("http") else "https://forums.beyondblue.org.au" + href
        page_num += 1

    return post_text, comments

def scrape_beyondblue_to_csv(heading_list, board_code_map, max_page_list):
    chrome_options = Options()
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    output_rows = []
    post_id_counter = 1

    for i, heading in enumerate(heading_list):
        board_code = board_code_map[heading]
        for page in range(1, max_page_list[i] + 1):
            url = f"https://forums.beyondblue.org.au/t5/{heading}/bd-p/{board_code}/page/{page}"
            try:
                driver.get(url)
                time.sleep(3)
                soup = BeautifulSoup(driver.page_source, "html.parser")
                articles = soup.find('div', class_ = "custom-message-list all-discussions").find_all("article")

                for article in articles:
                    try:
                        title_tag = article.select_one("h3 > a[href*='/td-p/']")
                        title = title_tag.text.strip()
                        post_url = "https://forums.beyondblue.org.au" + title_tag["href"]

                        replies_tag = article.select_one("li.custom-tile-replies b")
                        replies = int(replies_tag.text.strip()) if replies_tag else 0

                        body_tag = article.select_one("p.body-text")
                        preview = body_tag.text.strip() if body_tag else ""

                        author_tag = article.select_one("div.custom-tile-author-info a")
                        author = author_tag.text.strip() if author_tag else "Unknown"

                        date_tag = article.select_one("div.custom-tile-date time")
                        if date_tag:
                            raw_date = date_tag.get("datetime")
                            if not raw_date:
                                raw_date = date_tag.text.strip()
                            date = normalize_date(raw_date)
                        else:
                            date = "Unknown"
                        if date == "Unknown" or date < "2019-01-01":
                            continue

                        cat_tag = article.select_one("div.custom-tile-category a")
                        category = cat_tag.text.strip() if cat_tag else heading

                        # Visit thread and get post body + comments
                        post_body, comments = scrape_post_and_comments(driver, post_url, max_comments=40)

                        output_rows.append({
                            "post_id": post_id_counter,
                            "title": title,
                            "author": author,
                            "date": date,
                            "category": category,
                            "preview": preview,
                            "post_text": post_body or "",
                            "num_comments": len(comments),
                            "comments_combined": json.dumps(comments),
                            "url": post_url
                        })
                        post_id_counter += 1
                        print(f"{post_url} - comments: {len(comments)}")
                    except Exception as e:
                        print("Error in article block:", e)
                        continue
            except Exception as e:
                print("Failed to load page:", url, e)
                continue

    driver.quit()

    if not os.path.exists("data"):
        os.makedirs("data")

    csv_path = "data/beyondblue.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["post_id", "title", "author", "date", "category", "preview", "post_text", "num_comments", "comments_combined", "url"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    print(f"CSV saved to {csv_path}")

if __name__ == "__main__":
    heading_list = ['anxiety', 'depression', 'ptsd-and-trauma', 'suicidal-thoughts-and-self-harm']
    board_code_map = {
        'anxiety': 'c1-sc2-b1',
        'depression': 'c1-sc2-b2',
        'ptsd-and-trauma': 'c1-sc2-b3',
        'suicidal-thoughts-and-self-harm': 'c1-sc2-b4'
    }
    max_page_list = [200 ,200 ,200, 200]
    scrape_beyondblue_to_csv(heading_list, board_code_map, max_page_list)
