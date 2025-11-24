from os import link
from pydoc import text
from turtle import title
from bs4 import BeautifulSoup
import requests


def fetch_website_content(url, max_length=2000):
    """
    Return the title and contents of the website at the given url;
    truncated to max_length characters.
    """

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return f"Error fetching website: {e}"

    soup = BeautifulSoup(response.content, "html.parser")

    title = soup.title.string.strip() if soup.title else "No title found"

    if soup.body:
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()  # Remove these tags from the soup (DOM tree)

        text = soup.get_text(separator="\n", strip=True)
    else:
        text = ""

    return (title + "\n\n" + text)[:max_length]


def fetch_website_links(url):
    """
    Return a list of all hyperlinks found on the website at the given url.
    """

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return f"Error fetching website: {e}"

    soup = BeautifulSoup(response.content, "html.parser")

    links = [a.get("href") for a in soup.find_all("a", href=True)]
    links = [
        link.strip()
        for link in links
        if link
        and link.strip()
        and not (link.startswith("mailto:") or link.startswith("tel:"))
    ]
    unique_links = list(dict.fromkeys(links))

    return unique_links
