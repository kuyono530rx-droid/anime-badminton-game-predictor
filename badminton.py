import json
import math
import os
import random
import urllib.parse
import urllib.request
from dataclasses import dataclass
import time
import urllib.error


# create cache file
CACHE_FILE = "character_stats_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# urls
JIKAN_BASE_URL = "https://api.jikan.moe/v4"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"


# class characters
@dataclass
class Character:
    name: str
    power: int
    agility: int
    stamina: int
    technique: int
    decision: int
    mental: int
    consistency: int


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def fetch_json(url):
    time.sleep(0.5)  # avoid speed limit error
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "anime-badminton-ranker/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url, payload, headers=None, retries=5):
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)

    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=request_headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))

        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = 2 * (attempt + 1)
                print(f"OpenAI rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            raise

    raise RuntimeError("OpenAI request failed after retries.")


def search_character(name):
    query = urllib.parse.urlencode({"q": name, "limit": 10})
    data = fetch_json(f"{JIKAN_BASE_URL}/characters?{query}")
    matches = data.get("data", [])
    if not matches:
        raise ValueError(f"Character not found: {name}")

    lowered_name = name.casefold()
    for match in matches:
        if match.get("name", "").casefold() == lowered_name:
            return match

    return max(matches, key=lambda item: item.get("favorites", 0))


def get_character_context(name):
    match = search_character(name)
    details = fetch_json(f"{JIKAN_BASE_URL}/characters/{match['mal_id']}/full").get("data", {})
    anime_entries = details.get("anime", [])
    primary_anime = anime_entries[0]["anime"]["title"] if anime_entries else "Unknown"
    related_names = []

    if anime_entries:
        anime_id = anime_entries[0]["anime"]["mal_id"]
        cast = fetch_json(f"{JIKAN_BASE_URL}/anime/{anime_id}/characters").get("data", [])
        related_names = [
            entry.get("character", {}).get("name", "")
            for entry in cast
            if entry.get("character", {}).get("mal_id") != match["mal_id"]
        ][:8]

    about = (details.get("about") or "").strip()
    if len(about) > 2000:
        about = about[:2000]

    return {
        "mal_id": match["mal_id"],
        "name": details.get("name", match.get("name", name)),
        "anime": primary_anime,
        "about": about or "No description available.",
        "related_names": related_names,
        "favorites": details.get("favorites", match.get("favorites", 0)),
    }


def extract_response_text(response_json):
    if response_json.get("output_text"):
        return response_json["output_text"]

    for item in response_json.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                return text

    raise ValueError("The OpenAI response did not contain text output.")


def estimate_attributes(context):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    prompt = "\n".join(
        [
            f"Character: {context['name']}",
            f"Anime: {context['anime']}",
            f"Favorites: {context['favorites']}",
            f"Related characters: {', '.join(context['related_names']) or 'None'}",
            "Description:",
            context["about"],
        ]
    )
    schema = {
        "type": "object",
        "properties": {
            "power": {"type": "integer", "minimum": 1, "maximum": 100},
            "agility": {"type": "integer", "minimum": 1, "maximum": 100},
            "stamina": {"type": "integer", "minimum": 1, "maximum": 100},
            "technique": {"type": "integer", "minimum": 1, "maximum": 100},
            "decision": {"type": "integer", "minimum": 1, "maximum": 100},
            "mental": {"type": "integer", "minimum": 1, "maximum": 100},
            "consistency": {"type": "integer", "minimum": 1, "maximum": 100},
        },
        "required": [
            "power",
            "agility",
            "stamina",
            "technique",
            "decision",
            "mental",
            "consistency",
        ],
        "additionalProperties": False,
    }
    payload = {
        "model": os.environ.get("OPENAI_MODEL", DEFAULT_LLM_MODEL),
        "instructions": (
            "Estimate badminton attributes for the anime character using only the provided context. "
            "Return integers from 1 to 100. Return valid JSON only."
        ),
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "badminton_attributes",
                "strict": True,
                "schema": schema,
            }
        },
    }
    response = post_json(
        OPENAI_RESPONSES_URL,
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    return json.loads(extract_response_text(response))


def build_character(name, cache):
    context = get_character_context(name)
    cache_key = str(context["mal_id"])

    if cache_key in cache:
        stats = cache[cache_key]
    else:
        stats = estimate_attributes(context)
        cache[cache_key] = stats
        save_cache(cache)

    return Character(
        name=context["name"],
        power=stats["power"],
        agility=stats["agility"],
        stamina=stats["stamina"],
        technique=stats["technique"],
        decision=stats["decision"],
        mental=stats["mental"],
        consistency=stats["consistency"],
    )

def point_win_prob(player_a, player_b, rally_count):
    fatigue_a = rally_count * max(0.15, (110 - player_a.stamina) / 220)
    fatigue_b = rally_count * max(0.15, (110 - player_b.stamina) / 220)
    score_a = (
        0.24 * player_a.power
        + 0.22 * player_a.agility
        + 0.18 * player_a.technique
        + 0.14 * player_a.decision
        + 0.12 * player_a.mental
        + 0.10 * player_a.stamina
        - fatigue_a
    )
    score_b = (
        0.24 * player_b.power
        + 0.22 * player_b.agility
        + 0.18 * player_b.technique
        + 0.14 * player_b.decision
        + 0.12 * player_b.mental
        + 0.10 * player_b.stamina
        - fatigue_b
    )
    noise_a = random.gauss(0, max(0.5, (105 - player_a.consistency) / 40))
    noise_b = random.gauss(0, max(0.5, (105 - player_b.consistency) / 40))
    return sigmoid((score_a - score_b + noise_a - noise_b) / 12)


def simulate_game(player_a, player_b):
    points_a = 0
    points_b = 0
    rally_count = 0

    while True:
        rally_count += 1
        if random.random() < point_win_prob(player_a, player_b, rally_count):
            points_a += 1
        else:
            points_b += 1

        if (points_a >= 21 or points_b >= 21) and abs(points_a - points_b) >= 2:
            return points_a > points_b
        if points_a == 30 or points_b == 30:
            return points_a > points_b


def simulate_match(player_a, player_b):
    games_a = 0
    games_b = 0

    while games_a < 2 and games_b < 2:
        if simulate_game(player_a, player_b):
            games_a += 1
        else:
            games_b += 1

    return player_a if games_a > games_b else player_b


def simulate_tournament(players):
    bracket = list(players)
    random.shuffle(bracket)

    total_wins = {player.name: 0 for player in players}

    while len(bracket) > 1:

        next_round = []

        for i in range(0, len(bracket), 2):

            if i + 1 >= len(bracket):
                # odd number → bye
                next_round.append(bracket[i])
                continue

            winner = simulate_match(bracket[i], bracket[i+1])
            total_wins[winner.name] += 1
            next_round.append(winner)

        bracket = next_round

    return total_wins


def run_ranking(names, simulations=200):
    
    cache = load_cache()
    players = [build_character(name, cache) for name in names]
    win_totals = {player.name: 0 for player in players}

    for _ in range(simulations):
        tournament_wins = simulate_tournament(players)
        for name, wins in tournament_wins.items():
            win_totals[name] += wins

    ranking = sorted(win_totals.items(), key=lambda item: (-item[1], item[0]))
    print(f"Final Ranking ({simulations} simulations)")
    for name, _ in ranking:
        print(name)


if __name__ == "__main__":
    example_players = [
        "Saitama",
        "Goku",
        "Levi Ackerman",
        "Mikasa Ackerman",
        "Naruto Uzumaki",
        "Monkey D. Luffy",
        "Gojo Satoru",
        "Tanjiro Kamado",
    ]
    run_ranking(example_players)

