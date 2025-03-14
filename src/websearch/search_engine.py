import requests

SEARCH_API_KEY = "your_api_key_here"  # Replace with actual API key
SEARCH_ENGINE_ID = "your_custom_search_engine_id"  # Replace with real ID
SEARCH_API_URL = "https://www.googleapis.com/customsearch/v1"

def get_web_results(query):
    """
    Search the web for relevant information related to the query.
    Returns the top 3 results.
    """
    params = {
        "key": SEARCH_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": 3  # Limit to top 3 results
    }
    
    response = requests.get(SEARCH_API_URL, params=params)
    
    if response.status_code == 200:
        results = response.json().get("items", [])
        return [{"title": res["title"], "link": res["link"], "snippet": res["snippet"]} for res in results]
    else:
        print(f"⚠️ Web search error: {response.status_code} - {response.text}")
        return []

if __name__ == "__main__":
    sample_query = "Best irrigation practices for drought conditions"
    search_results = get_web_results(sample_query)
    print(search_results)
