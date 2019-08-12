import sys
from pathlib import Path

project_path = str(Path(__file__).resolve().parents[1])
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import requests
from bs4 import BeautifulSoup
import bs4.element
from pathlib import Path
from multiprocessing import Pool
import psutil
import json
from time import time
from datetime import datetime
import re
from nlp_intro.my_multiprocessing import process


MAIN_PAGE_URL = "https://www.diskusjon.no"
INDEX_PAGE_URL = "https://www.diskusjon.no/index.php"
BATCH_SIZE = 100
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "diskusjonno"
DATA_FILE = DATA_DIR / "posts.jsonl"
SCRAPED_TOPICS_IDS_FILE = DATA_DIR / "scraped_topics_ids.txt"
N_WORKERS = psutil.cpu_count(logical=False)
SKIP_FORUMS = [re.compile(r".*\(snarvei\)$"), "Forumarkiv"]


def scrape(n_per_topic=100, n_jobs=-1):
    topic_generator = _scrape_topic_ids_and_labels(n_per_topic)
    while 1:
        print(f"Scraping next batch ", end="")
        topics_batch = _next_batch(topic_generator, BATCH_SIZE)
        t0 = time()
        posts = process(_scrape_first_post_from_topic, topics_batch, n_jobs)
        n_posts = len(posts)
        print(f"[{n_posts} posts scraped in {time()-t0:.2f} seconds]")
        if n_posts:
            _store_batch(posts)
        else:
            break


def _next_batch(iterator, n):
    for _ in range(n):
        try:
            yield next(iterator)
        except StopIteration:
            break


def _store_batch(posts):
    DATA_FILE.parent.mkdir(exist_ok=True, parents=True)
    topic_ids = []
    with open(DATA_FILE, "a", encoding="utf8") as f:
        for post in posts:
            json.dump(post, f)
            topic_ids.append(post["topic_id"])
            f.write("\n")

    with open(SCRAPED_TOPICS_IDS_FILE, "a") as f:
        f.write(" ".join(str(t) for t in topic_ids) + " ")


def _scrape_topic_ids_and_labels(n_per_topic):
    scraped_topics_ids = _read_scraped_topics_ids()
    soup = _get_soup(MAIN_PAGE_URL)
    for forum_tag in soup.find_all("div", id=re.compile(r"^category_\d+$")):
        forum_name = forum_tag.find("span").text
        if not _is_valid_forum_name(forum_name):
            continue
        forum_id = _extract_forum_id(forum_tag.find("a")["href"])
        topic_generator = _scrape_topics_from_all_subforums(forum_id, n_per_topic, [forum_name])
        for topic_id, labels in topic_generator:
            if topic_id not in scraped_topics_ids:
                yield topic_id, labels
                scraped_topics_ids.append(topic_id)
                topic_generator.send(True)
            else:
                topic_generator.send(False)


def _read_scraped_topics_ids():
    if SCRAPED_TOPICS_IDS_FILE.is_file():
        with open(SCRAPED_TOPICS_IDS_FILE) as f:
            return [int(x) for x in f.read().split()]
    return []


def _scrape_topics_from_all_subforums(forum_id, n_per_topic, parent_labels):
    soup = _get_soup(INDEX_PAGE_URL, showforum=forum_id)
    yield from _scrape_topics_from_multiple_pages(soup, forum_id, n_per_topic, parent_labels)
    for next_forum_id, forum_name in _scrape_forum_ids(soup):
        if not _is_valid_forum_name(forum_name):
            continue
        yield from _scrape_topics_from_all_subforums(next_forum_id, n_per_topic, parent_labels + [forum_name])


def _is_valid_forum_name(forum_name):
    for skip_forum in SKIP_FORUMS:
        if isinstance(skip_forum, re.Pattern):
            if skip_forum.match(forum_name):
                return False
        elif skip_forum == forum_name:
            return False
    return True


def _scrape_topics_from_multiple_pages(soup, forum_id, n_per_topic, labels, n_new_scraped=0, page=1):
    n_scraped_for_page = 0
    for topic_id in _scrape_topics_from_one_page(soup):
        n_scraped_for_page += 1
        was_new_post = yield topic_id, labels
        yield
        if was_new_post:
            n_new_scraped += 1
            if n_new_scraped == n_per_topic:
                break
    else:
        if n_scraped_for_page:  # If no pages were scraped for this page, we assume there are no more pages to scrape
            soup = _get_soup(INDEX_PAGE_URL, showforum=forum_id, page=page + 1)
            yield from _scrape_topics_from_multiple_pages(soup, forum_id, n_per_topic, labels, n_new_scraped, page + 1)


def _scrape_forum_ids(forum_soup):
    forums_list = forum_soup.find("div", "category_block")
    if forums_list is None:
        return
    for forum_tag in forums_list.find_all("tr"):
        a = forum_tag.find("a")
        if a is None:
            continue
        forum_id = _extract_forum_id(a["href"])
        if forum_id is None:
            continue
        forum_name = a.text
        yield forum_id, forum_name


def _extract_forum_id(forum_url):
    match = re.search(r"(?<=showforum=)\d+", forum_url)
    if match is None:
        return None
    return int(match.group())


def _scrape_topics_from_one_page(forum_soup):
    topic_list = forum_soup.find("table", "topic_list")
    if topic_list is None:
        return
    for topic_tag in topic_list.find_all("tr"):
        a = topic_tag.find("a")
        if a is None:
            continue
        _id = a.get("id")
        if _id is None:
            continue
        topic_id = int(_id.split("-")[-1])
        yield topic_id


def _scrape_first_post_from_topic(topic_id_and_labels):
    topic_id, labels = topic_id_and_labels
    soup = _get_soup(INDEX_PAGE_URL, showtopic=topic_id, page=1)
    create_time = soup.find("abbr", "published")["title"]
    author = soup.select_one('span[itemprop="name"]').text
    title = soup.find("h1", "ipsType_pagetitle").text
    body_soup = soup.find("div", ["post", "entry-content"])
    body = _extract_and_clean_body_text(body_soup)
    scrape_time = datetime.utcnow().isoformat()+"+00:00"
    return dict(topic_id=topic_id, create_time=create_time, author=author, title=title, body=body, labels=labels,
                scrape_time=scrape_time)


def _extract_and_clean_body_text(body_soup):
    body = _extract_text(body_soup)
    body = body.strip()
    body = re.sub(r"\s*\n{3,}\s*", "\n\n", body)
    return body


def _extract_text(soup):
    text = ""
    for node in soup:
        if isinstance(node, bs4.Comment):
            continue
        if isinstance(node, bs4.element.NavigableString):
            text += str(node)
        elif isinstance(node, bs4.element.Tag):
            if "edit" in node.get("class", []):
                continue
            if re.match(r"^(?:h\d|p|strong|a|em|span)$", node.name):
                child_text = _extract_text(node)
                text += child_text if child_text else node.text
            elif node.name == "br":
                text += "\n"
            elif node.name in {"ul", "ol"}:
                for j, li_tag in enumerate(node.find_all("li")):
                    text += f"{j + 1}. {li_tag.text}\n"
                text += "\n"
    return text


def _get_soup(url, **params):
    return BeautifulSoup(requests.get(url, params=params).text, "html.parser")


if __name__ == '__main__':
    scrape(100)
