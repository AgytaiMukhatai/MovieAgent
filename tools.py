import os
import json
import requests
from typing import Optional, List, Dict, Any
from langchain.tools import tool
from langchain_core.tools import StructuredTool


# ---------- Internal helper ---------- #

def _omdb_request(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Internal helper to call the OMDb API with shared logic.
    """
    api_url = os.getenv("OMDB_API_URL", "http://www.omdbapi.com/")
    api_key = os.getenv("OMDB_API_KEY")

    if not api_url:
        return {
            "success": False,
            "error": "OMDB_API_URL not configured in environment"
        }

    if api_key:
        params["apikey"] = api_key

    try:
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("Response") == "False":
            return {
                "success": False,
                "error": data.get("Error", "Movie not found")
            }

        return {
            "success": True,
            "movie" if "Title" in data else "data": data
        }

    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


# ---------- Existing tools (fixed) ---------- #

@tool
def search_movie_by_title(title: str) -> str:
    """
    Search for a movie by exact or partial title match.
    Returns detailed movie information including cast, director, ratings, and plot.

    Args:
        title: The movie title to search for (e.g., "Guardians of the Galaxy")

    Returns:
        JSON string:
        {
            "success": bool,
            "movie"?: {...},  # OMDb movie object
            "error"?: str
        }
    """
    result = _omdb_request({"t": title})
    return json.dumps(result)


@tool
def get_movie_ratings(movie_data: str) -> str:
    """
    Extract and analyze ratings from various sources (IMDb, Rotten Tomatoes, Metacritic).

    Args:
        movie_data: JSON string from OMDb or from search_movie_by_title()

    Returns:
        JSON string:
        {
            "success": bool,
            "ratings"?: {...},
            "error"?: str
        }
    """
    try:
        if isinstance(movie_data, str):
            wrapper = json.loads(movie_data)
        else:
            wrapper = movie_data

        if "movie" in wrapper:
            movie = wrapper["movie"]
        else:
            movie = wrapper

        ratings_info = {
            "title": movie.get("Title"),
            "imdb": {
                "rating": movie.get("imdbRating"),
                "votes": movie.get("imdbVotes")
            },
            "metascore": movie.get("Metascore"),
            "detailed_ratings": movie.get("Ratings", [])
        }

        return json.dumps({
            "success": True,
            "ratings": ratings_info
        })

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def search_movies_list(query: str,
                       year: Optional[str] = None,
                       movie_type: Optional[str] = None,
                       page: int = 1) -> str:
    """
    Search for a list of movies by a text query (OMDb 's' endpoint).

    Args:
        query: Text to search for (e.g. "batman")
        year: Optional year filter (e.g. "2010")
        movie_type: Optional type filter: "movie", "series", or "episode"
        page: Page number in OMDb results (1-100)

    Returns:
        JSON string:
        {
            "success": bool,
            "data"?: {
                "Search": [...],
                "totalResults": "XX",
                ...
            },
            "error"?: str
        }
    """
    params = {"s": query, "page": page}
    if year:
        params["y"] = year
    if movie_type:
        params["type"] = movie_type

    result = _omdb_request(params)
    return json.dumps(result)


@tool
def compare_movies(title_1: str, title_2: str) -> str:
    """
    Compare two movies by their titles (year, rating, genre, director, runtime, awards).

    Args:
        title_1: First movie title
        title_2: Second movie title

    Returns:
        JSON string:
        {
            "success": bool,
            "comparison"?: {
                "movie_1": {...},
                "movie_2": {...},
                "summary": str
            },
            "error"?: str
        }
    """
    movie1 = _omdb_request({"t": title_1})
    movie2 = _omdb_request({"t": title_2})

    if not movie1.get("success"):
        return json.dumps({
            "success": False,
            "error": f"First movie not found: {movie1.get('error')}"
        })

    if not movie2.get("success"):
        return json.dumps({
            "success": False,
            "error": f"Second movie not found: {movie2.get('error')}"
        })

    m1 = movie1["movie"]
    m2 = movie2["movie"]

    def safe_float(x: Optional[str]) -> Optional[float]:
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    comparison = {
        "movie_1": {
            "title": m1.get("Title"),
            "year": m1.get("Year"),
            "imdbRating": m1.get("imdbRating"),
            "genre": m1.get("Genre"),
            "director": m1.get("Director"),
            "runtime": m1.get("Runtime"),
            "awards": m1.get("Awards"),
        },
        "movie_2": {
            "title": m2.get("Title"),
            "year": m2.get("Year"),
            "imdbRating": m2.get("imdbRating"),
            "genre": m2.get("Genre"),
            "director": m2.get("Director"),
            "runtime": m2.get("Runtime"),
            "awards": m2.get("Awards"),
        }
    }

    r1 = safe_float(m1.get("imdbRating"))
    r2 = safe_float(m2.get("imdbRating"))

    if r1 is not None and r2 is not None:
        if r1 > r2:
            summary = f"{m1.get('Title')} is higher rated on IMDb ({r1} vs {r2})."
        elif r2 > r1:
            summary = f"{m2.get('Title')} is higher rated on IMDb ({r2} vs {r1})."
        else:
            summary = f"Both movies have the same IMDb rating ({r1})."
    else:
        summary = "IMDb ratings are not available for a direct comparison."

    comparison["summary"] = summary

    return json.dumps({
        "success": True,
        "comparison": comparison
    })


@tool
def get_cast_and_authors(movie_data: str) -> str:
    """
    Extract cast and authors (director, writers, actors, production) from movie data.

    Args:
        movie_data: JSON string from OMDb or from search_movie_by_title()

    Returns:
        JSON string:
        {
            "success": bool,
            "people"?: {
                "title": str,
                "year": str,
                "director": str,
                "writers": str,
                "actors": str,
                "production": str
            },
            "error"?: str
        }
    """
    try:
        if isinstance(movie_data, str):
            wrapper = json.loads(movie_data)
        else:
            wrapper = movie_data

        if "movie" in wrapper:
            movie = wrapper["movie"]
        else:
            movie = wrapper

        people = {
            "title": movie.get("Title"),
            "year": movie.get("Year"),
            "director": movie.get("Director"),
            "writers": movie.get("Writer"),
            "actors": movie.get("Actors"),
            "production": movie.get("Production")
        }

        return json.dumps({
            "success": True,
            "people": people
        })

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def find_movies_by_min_imdb_rating(query: str,
                                   min_rating: float = 8.0,
                                   max_results: int = 5) -> str:
    """
    Find movies by text query and filter by minimum IMDb rating.
    Uses OMDb search ('s') + per-movie detail lookups.

    Args:
        query: Text to search for (e.g. "batman")
        min_rating: Minimum IMDb rating (e.g. 7.5)
        max_results: Max number of movies to return

    Returns:
        JSON string:
        {
            "success": bool,
            "results"?: [
                {
                    "title": str,
                    "year": str,
                    "imdbRating": str,
                    "genre": str,
                    "director": str,
                    "imdbID": str
                },
                ...
            ],
            "error"?: str
        }
    """
    try:
        # Step 1: coarse search
        search_result = _omdb_request({"s": query})
        if not search_result.get("success"):
            return json.dumps({
                "success": False,
                "error": search_result.get("error", "Search failed")
            })

        search_data = search_result["data"]
        search_list = search_data.get("Search", [])

        def safe_float(x: Optional[str]) -> Optional[float]:
            try:
                return float(x)
            except (TypeError, ValueError):
                return None

        filtered: List[Dict[str, Any]] = []

        # Step 2: for each search hit, fetch full details and filter by rating
        for item in search_list:
            imdb_id = item.get("imdbID")
            if not imdb_id:
                continue

            detail = _omdb_request({"i": imdb_id})
            if not detail.get("success"):
                continue

            movie = detail["movie"]
            rating = safe_float(movie.get("imdbRating"))
            if rating is None or rating < min_rating:
                continue

            filtered.append({
                "title": movie.get("Title"),
                "year": movie.get("Year"),
                "imdbRating": movie.get("imdbRating"),
                "genre": movie.get("Genre"),
                "director": movie.get("Director"),
                "imdbID": movie.get("imdbID")
            })

            if len(filtered) >= max_results:
                break

        return json.dumps({
            "success": True,
            "results": filtered
        })

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# ---------- Tool list ---------- #

AVAILABLE_TOOLS = [
    search_movie_by_title,
    get_movie_ratings,
    search_movies_list,
    compare_movies,
    get_cast_and_authors,
    find_movies_by_min_imdb_rating,
]
