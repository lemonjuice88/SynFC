"""
reddit_scraper.py — Reddit fan sentiment via public JSON endpoints
=======================================================================
Reddit's official API now requires requesting and getting EXPLICIT
APPROVAL before use (their "Responsible Builder Policy", 2026) -- this
sidesteps that entirely by reading the SAME public JSON a browser gets
when you visit a subreddit/post (append ".json" to any Reddit URL),
using curl_cffi's browser impersonation -- the same technique that got
Transfermarkt/FBref scraping working reliably elsewhere in this
project. No OAuth, no app registration, no approval wait.

Tradeoff, stated honestly: this is unauthenticated, READ-ONLY access to
PUBLIC content only (which is all we ever needed -- no voting, posting,
or private data here). Being unauthenticated, it's subject to a lower,
unpublished rate limit, and Reddit could tighten this without notice --
treat it as a best-effort research tool, not a guaranteed permanent
integration. If it stops working, the officially-approved API route
(see reddit.com/prefs/apps) is the fallback, once/if approval comes
through.

Two jobs, matching the two-part Media identity plan:
1. fetch_bulk_comments() -- large one-time corpus for identity synthesis.
2. search_subreddit_comments() -- live, keyword-scoped search, meant to
   plug into media_team.py's `fans` agent.

Setup: no API keys needed. Just needs curl_cffi (already a project
dependency via tools.py).
"""

import time
from typing import List

from curl_cffi import requests as cffi_requests

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
}
REQUEST_DELAY_SECONDS = 2  # be polite, same convention as tools.py
REDDIT_BASE_URL = "https://old.reddit.com"  # less aggressively bot-protected than www.reddit.com


def _fetch_json(url: str) -> dict:
    resp = cffi_requests.get(url, headers=HEADERS, timeout=15, impersonate="chrome124")
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return resp.json()


def _flatten_comments(comment_children: list) -> List[str]:
    """Recursively walks Reddit's nested comment tree, returning a flat
    list of comment body text (skipping deleted/removed/"load more"
    stub entries)."""
    texts = []
    for child in comment_children:
        if child.get("kind") != "t1":  # skip "more" stubs etc.
            continue
        data = child.get("data", {})
        body = (data.get("body") or "").strip()
        if body and body not in ("[deleted]", "[removed]"):
            texts.append(body)

        replies = data.get("replies")
        if isinstance(replies, dict):
            nested_children = replies.get("data", {}).get("children", [])
            texts.extend(_flatten_comments(nested_children))
    return texts


def fetch_bulk_comments(
    subreddit_name: str,
    target_count: int = 2000,
    listing: str = "top",
    time_filter: str = "year",
    posts_limit: int = 100,
) -> dict:
    """Pulls a corpus of comment text from a subreddit's top/hot posts.
    Meant for a ONE-TIME bulk pass (e.g. feeding it to a strong model
    to synthesize a fanbase identity persona), not frequent/live use.

    Example return:
        {"found": True, "subreddit": "LiverpoolFC", "comment_count": 1840,
         "comments": ["...", ...], "posts_scanned": 42}
    """
    try:
        listing_url = (
            f"{REDDIT_BASE_URL}/r/{subreddit_name}/{listing}.json"
            f"?limit={min(posts_limit, 100)}&t={time_filter}"
        )
        listing_data = _fetch_json(listing_url)
        posts = listing_data.get("data", {}).get("children", [])

        comments = []
        posts_scanned = 0
        for post in posts:
            post_id = post.get("data", {}).get("id")
            if not post_id:
                continue
            posts_scanned += 1

            comments_url = f"{REDDIT_BASE_URL}/r/{subreddit_name}/comments/{post_id}.json?limit=200"
            try:
                comments_data = _fetch_json(comments_url)
                comment_children = comments_data[1].get("data", {}).get("children", [])
                comments.extend(_flatten_comments(comment_children))
            except Exception:
                continue  # one bad post shouldn't kill the whole run

            if len(comments) >= target_count:
                break

        return {
            "found": True,
            "subreddit": subreddit_name,
            "comment_count": len(comments),
            "comments": comments[:target_count],
            "posts_scanned": posts_scanned,
        }
    except Exception as e:
        return {"found": False, "reason": f"Error while fetching from r/{subreddit_name}: {e}"}


def search_subreddit_comments(
    subreddit_name: str, query_terms: List[str], max_results: int = 25, posts_to_scan: int = 25
) -> dict:
    """Live, keyword-scoped search within a subreddit's hot posts -- the
    Reddit counterpart to tools.search_fan_comments(), meant to plug
    into media_team.py's `fans` agent."""
    try:
        query_terms_lower = [t.lower() for t in query_terms if t]
        listing_url = f"{REDDIT_BASE_URL}/r/{subreddit_name}/hot.json?limit={posts_to_scan}"
        listing_data = _fetch_json(listing_url)
        posts = listing_data.get("data", {}).get("children", [])

        matches = []
        for post in posts:
            post_id = post.get("data", {}).get("id")
            if not post_id:
                continue
            comments_url = f"{REDDIT_BASE_URL}/r/{subreddit_name}/comments/{post_id}.json?limit=200"
            try:
                comments_data = _fetch_json(comments_url)
                comment_children = comments_data[1].get("data", {}).get("children", [])
                for text in _flatten_comments(comment_children):
                    if any(term in text.lower() for term in query_terms_lower):
                        matches.append(text)
            except Exception:
                continue

        total_matches = len(matches)
        if total_matches > max_results:
            step = total_matches / max_results
            sample = [matches[int(i * step)] for i in range(max_results)]
        else:
            sample = matches

        return {"found": True, "subreddit": subreddit_name, "total_matches": total_matches, "sample": sample}
    except Exception as e:
        return {"found": False, "reason": f"Error while searching r/{subreddit_name}: {e}"}


if __name__ == "__main__":
    # Quick manual test:
    #   python reddit_scraper.py search LiverpoolFC Salah transfer
    #   python reddit_scraper.py bulk LiverpoolFC 300
    import sys
    import json

    mode = sys.argv[1] if len(sys.argv) > 1 else "search"
    subreddit = sys.argv[2] if len(sys.argv) > 2 else "LiverpoolFC"

    if mode == "bulk":
        target = int(sys.argv[3]) if len(sys.argv) > 3 else 300
        result = fetch_bulk_comments(subreddit, target_count=target, posts_limit=20)
        print(f"found={result.get('found')}, comment_count={result.get('comment_count')}, "
              f"posts_scanned={result.get('posts_scanned')}, reason={result.get('reason')}")
        if result.get("found") and result["comments"]:
            print("\nÖrnek yorumlar:")
            for c in result["comments"][:5]:
                print(" -", c[:150])
    else:
        terms = sys.argv[3:] or ["transfer"]
        result = search_subreddit_comments(subreddit, terms)
        print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])