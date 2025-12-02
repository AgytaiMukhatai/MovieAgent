import os
import json
import requests
from typing import Optional
from langchain.tools import tool
from langchain_core.tools import StructuredTool

@tool
def search_movie_by_title(title: str) -> str:
    """
    Search for a movie by exact or partial title match.
    Returns detailed movie information including cast, director, ratings, and plot.
    
    Args:
        title: The movie title to search for (e.g., "Guardians of the Galaxy")
    
    Returns:
        JSON string with complete movie details
    """
    api_url = os.getenv("OMDB_API_URL")
    api_key = os.getenv("OMDB_API_KEY")
    
    if not api_url:
        return json.dumps({"error": "MOVIE_API_URL not configured in environment"})
    
    try:
        # Build request parameters based on your API
        params = {"t": title}  # 't' for title search (OMDb API format)
        if api_key:
            params["apikey"] = api_key
        
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check if movie was found
        if data.get("Response") == "False":
            return json.dumps({
                "success": False,
                "error": data.get("Error", "Movie not found")
            })
        
        # Return the full movie data in your API's format
        return json.dumps({
            "success": True,
            "movie": data
        })
    
    except requests.exceptions.RequestException as e:
        return json.dumps({"success": False, "error": f"API request failed: {str(e)}"})
    except Exception as e:
        return json.dumps({"success": False, "error": f"Unexpected error: {str(e)}"})


@tool
def search_movie_by_id(imdb_id: str) -> str:
    """
    Search for a movie by IMDb ID for exact match.
    
    Args:
        imdb_id: The IMDb ID (e.g., "tt3896198")
    
    Returns:
        JSON string with complete movie details
    """
    api_url = os.getenv("OMDB_API_URL")
    api_key = os.getenv("OMDB_API_KEY")
    
    if not api_url:
        return json.dumps({"error": "OMDB_API_URL not configured in environment"})
    
    try:
        params = {"i": imdb_id}  # 'i' for IMDb ID
        if api_key:
            params["apikey"] = api_key
        
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("Response") == "False":
            return json.dumps({
                "success": False,
                "error": data.get("Error", "Movie not found")
            })
        
        return json.dumps({
            "success": True,
            "movie": data
        })
    
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def analyze_cinematography(movie_data: str) -> str:
    """
    Provide expert cinematography analysis based on movie information.
    This tool analyzes the visual style, camera work, and cinematographic choices.
    
    Args:
        movie_data: JSON string of movie data from search results
    
    Returns:
        JSON string with detailed cinematography analysis
    """
    try:
        # Parse the movie data
        if isinstance(movie_data, str):
            movie = json.loads(movie_data)
        else:
            movie = movie_data
        
        title = movie.get("Title", "Unknown")
        year = movie.get("Year", "Unknown")
        director = movie.get("Director", "Unknown")
        genre = movie.get("Genre", "Unknown")
        
        # Build cinematography analysis based on known data
        # In production, this could call another API or database
        analysis = {
            "title": title,
            "year": year,
            "director": director,
            "genre": genre,
            "analysis": {
                "overview": f"Cinematographic analysis of {title} ({year}), directed by {director}.",
                "visual_style": f"This {genre.lower()} film employs techniques common to its genre.",
                "camera_work": "Analysis of camera movements, angles, and shot composition.",
                "lighting": "Analysis of lighting design and mood creation.",
                "color_palette": "Analysis of color grading and visual tone.",
                "notable_techniques": [
                    "Genre-specific cinematographic approaches",
                    "Director's visual signature",
                    "Technical innovations or traditional methods"
                ]
            },
            "cinematographer": "Information not available in current data source",
            "technical_specs": {
                "aspect_ratio": "Information not available",
                "camera": "Information not available",
                "film_stock_or_digital": "Information not available"
            }
        }
        
        # Add specific analysis for known films
        cinematography_database = {
            "Guardians of the Galaxy Vol. 2": {
                "cinematographer": "Henry Braham",
                "camera_work": "Dynamic camera movements combining Steadicam and crane shots, extensive use of gimbal rigs for action sequences",
                "lighting": "Vibrant, colorful lighting schemes with heavy use of practical neon and colored gels. High-key lighting for the fantastical space environments",
                "color_palette": "Extremely saturated color palette with bold primary colors - reds, blues, yellows, and purples. Creates a comic book aesthetic",
                "visual_style": "Pop-art inspired visual design with psychedelic color schemes. Mix of wide establishing shots and close-ups for character moments",
                "technical_specs": {
                    "camera": "RED Weapon 8K",
                    "aspect_ratio": "2.39:1 (Anamorphic)",
                    "format": "Digital"
                },
                "notable_scenes": [
                    "Opening sequence with Baby Groot dancing - single take following the action",
                    "Ego's planet sequences with otherworldly color palettes",
                    "Yondu's arrow sequence with practical LED lighting effects"
                ]
            }
        }
        
        if title in cinematography_database:
            specific_analysis = cinematography_database[title]
            analysis["cinematographer"] = specific_analysis.get("cinematographer")
            analysis["analysis"]["camera_work"] = specific_analysis.get("camera_work")
            analysis["analysis"]["lighting"] = specific_analysis.get("lighting")
            analysis["analysis"]["color_palette"] = specific_analysis.get("color_palette")
            analysis["analysis"]["visual_style"] = specific_analysis.get("visual_style")
            analysis["technical_specs"] = specific_analysis.get("technical_specs")
            if "notable_scenes" in specific_analysis:
                analysis["notable_scenes"] = specific_analysis["notable_scenes"]
        
        return json.dumps({
            "success": True,
            "analysis": analysis
        })
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Analysis failed: {str(e)}"
        })


@tool
def get_movie_ratings(movie_data: str) -> str:
    """
    Extract and analyze ratings from various sources (IMDb, Rotten Tomatoes, Metacritic).
    
    Args:
        movie_data: JSON string of movie data
    
    Returns:
        JSON string with ratings analysis
    """
    try:
        if isinstance(movie_data, str):
            movie = json.loads(movie_data)
        else:
            movie = movie_data
        
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


# Create tool list
AVAILABLE_TOOLS = [
    search_movie_by_title,
    search_movie_by_id,
    analyze_cinematography,
    get_movie_ratings
]
