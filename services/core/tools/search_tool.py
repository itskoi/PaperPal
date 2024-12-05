### Just a dummy search tools
import requests
import mwparserfromhell
import json

class WikipediaSearchTool:
    def __init__(self, api_url="https://en.wikipedia.org/w/api.php"):
        self.api_url = api_url

    def search(self, title: str) -> dict:
        """
        Perform a search on Wikipedia and return the parsed content.

        Args:
            title (str): The title of the Wikipedia page to search.

        Returns:
            dict: A dictionary containing the name of the tool and the content.
        """
        try:
            response = requests.get(
                self.api_url,
                params={
                    "action": "query",
                    "format": "json",
                    "titles": title,
                    "prop": "revisions",
                    "rvprop": "content",
                },
            )
            response.raise_for_status()
            data = response.json()
            page = next(iter(data["query"]["pages"].values()))

            if "revisions" not in page:
                return {"name": "wikipedia_search", "content": f"Page '{title}' not found."}

            wikicode = page["revisions"][0]["*"]
            parsed_wikicode = mwparserfromhell.parse(wikicode)
            content = parsed_wikicode.strip_code()
            return {"name": "wikipedia_search", "content": content}

        except requests.exceptions.RequestException as e:
            return {"name": "wikipedia_search", "content": f"Error: {str(e)}"}

        except Exception as e:
            return {"name": "wikipedia_search", "content": f"Unexpected error: {str(e)}"}
