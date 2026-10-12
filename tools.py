
from bs4 import BeautifulSoup
from tavily import TavilyClient
from langchain.tools import tool

import os
import requests
from rich import print
from dotenv import load_dotenv

load_dotenv()

# load tavily api key
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=tavily_api_key)

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a given topic. It will return Title's, URLs and snippets"""
    results = tavily.search(query=query, max_results=5)

    out = []

    for r in results['results']:
        out.append(
            f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n"
        )
    
    return "\n---\n".join(out)

print(web_search.invoke("What is the recent new on Anime?"))