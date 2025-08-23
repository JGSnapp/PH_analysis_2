import os, sys, json, argparse, time
import httpx
from dotenv import load_dotenv
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler

from antisleep import keep_awake

load_dotenv()

API_URL = "https://api.producthunt.com/v2/api/graphql"
TOKEN = os.getenv("PRODUCTHUNT_TOKEN")

# Запрашиваем просто самые новые посты, пагинация курсором
GQL = """
query FetchPosts($after: String) {
  posts(first: 10, after: $after, order: NEWEST) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id slug name url
        createdAt featuredAt
        tagline
        description
        votesCount
        commentsCount
        topics { edges { node { name } } }
        thumbnail { url }
        website
        media { type url videoUrl }
      }
    }
  }
}
"""

# -------------------- logging --------------------
def setup_logging():
    logger = logging.getLogger("ph_fetch")
    if logger.handlers:
        return logger  # уже настроен
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = RotatingFileHandler("fetch.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

logger = setup_logging()

# -------------------- state --------------------
def save_state(cursor, fetched):
    try:
        with open("state.json", "w", encoding="utf-8") as f:
            json.dump({"cursor": cursor, "fetched": fetched}, f, ensure_ascii=False, indent=2)
        logger.info(f"[state] saved: cursor={bool(cursor)}, fetched_total={fetched}")
    except Exception as e:
        logger.error(f"[state] save failed: {e}")

def load_state():
    if os.path.exists("state.json"):
        try:
            with open("state.json", "r", encoding="utf-8") as f:
                s = json.load(f)
            return s
        except Exception as e:
            logger.warning(f"[state] load failed, starting fresh: {e}")
    return {"cursor": None, "fetched": 0}

# === запись/мердж тем ===
def write_topics_merged(topics_dict, path="output_topics.xlsx"):
    rows = []
    for t, s in topics_dict.items():
        rows.append({
            "topic": t,
            "sum_votes": float(s.get("sum_votes", 0)),
            "sum_comments": float(s.get("sum_comments", 0)),
            "count_posts": int(s.get("count_posts", 0)),
        })
    df_curr = pd.DataFrame(rows)
    if df_curr.empty:
        logger.info("[topics] nothing to write this run.")
        return

    if os.path.exists(path):
        try:
            df_prev = pd.read_excel(path)
            need_cols = {"topic","sum_votes","sum_comments","count_posts"}
            if not need_cols.issubset(df_prev.columns):
                logger.warning("[topics] existing file has unexpected schema, will overwrite.")
                df_prev = pd.DataFrame(columns=["topic","sum_votes","sum_comments","count_posts"])
        except Exception as e:
            logger.warning(f"[topics] failed to read existing file, will overwrite: {e}")
            df_prev = pd.DataFrame(columns=["topic","sum_votes","sum_comments","count_posts"])
    else:
        df_prev = pd.DataFrame(columns=["topic","sum_votes","sum_comments","count_posts"])

    df_all = pd.concat([df_prev, df_curr], ignore_index=True)
    df_all = (
        df_all.groupby("topic", as_index=False)
              .agg({"sum_votes":"sum","sum_comments":"sum","count_posts":"sum"})
    )
    df_all["mean_votes"] = df_all["sum_votes"] / df_all["count_posts"].replace(0, pd.NA)
    df_all["mean_comments"] = df_all["sum_comments"] / df_all["count_posts"].replace(0, pd.NA)
    df_all = df_all.sort_values(["mean_votes","count_posts"], ascending=[False, False])

    df_all.to_excel(path, index=False)
    logger.info(f"[topics] written {len(df_all)} rows -> {path}")

# -------------------- main --------------------
def main():

    state = load_state()
    cursor = state.get("cursor")
    fetched_total = state.get("fetched", 0)
    logger.info(f"[resume] cursor exists={bool(cursor)}, fetched_total={fetched_total}")

    parser = argparse.ArgumentParser(description="Minimal Product Hunt fetcher (prints JSONL to stdout)")
    parser.add_argument("--limit", type=int, default=20000, help="Сколько постов собрать (по умолчанию 100)")
    parser.add_argument("--sleep", type=float, default=0.15, help="Пауза между запросами (сек), по умолчанию 0.15")
    args = parser.parse_args()

    if not TOKEN:
        logger.error("ERROR: PRODUCTHUNT_TOKEN не задан. Положи его в .env или env.")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    # анти-сон вынесен в отдельный модуль
    with keep_awake("PH fetch is running", logger=logger), httpx.Client(timeout=30) as client:
        base_df = pd.DataFrame()
        topics = {}

        try:
            while fetched_total < args.limit:
                payload = {"query": GQL, "variables": {"after": cursor}}
                resp = client.post(API_URL, headers=headers, json=payload)

                if resp.status_code == 429:
                    logger.warning("RATE-LIMIT (429). Sleeping 15 min…")
                    save_state(cursor, fetched_total)
                    time.sleep(5*60)
                    continue

                resp.raise_for_status()
                data = resp.json()

                if "errors" in data and data["errors"]:
                    logger.error(f"GraphQL errors: {data['errors']}")
                    save_state(cursor, fetched_total)
                    time.sleep(5)
                    continue

                posts = data["data"]["posts"]
                edges = posts["edges"]
                added = len(edges)
                fetched_total += added
                logger.info(f"[page] fetched {added} posts; total={fetched_total}")

                df = pd.DataFrame([e["node"] for e in edges])

                df["createdAt"] = pd.to_datetime(df["createdAt"], utc=True, errors="coerce")
                df["featuredAt"] = pd.to_datetime(df["featuredAt"], utc=True, errors="coerce")

                for col in ["createdAt", "featuredAt"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
                        df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)

                df["votesCount"] = df["votesCount"].apply(lambda x: int(x) if pd.notna(x) else 0)
                df["commentsCount"] = df["commentsCount"].apply(lambda x: int(x) if pd.notna(x) else 0)
                df["tagline_len"] = df["tagline"].fillna("").str.len()
                df["desc_len"] = df["description"].fillna("").str.len()
                df["topics_count"] = df["topics"].apply(lambda t: len(t["edges"]) if isinstance(t, dict) and "edges" in t else 0)
                df["topics_str"] = df["topics"].apply(lambda t: ",".join(edge["node"]["name"] for edge in t.get("edges", [])) if isinstance(t, dict) else "")

                df["has_media"] = df["media"].notna().astype(int)
                df["hour"] = df["createdAt"].dt.hour
                df["dow"] = df["createdAt"].dt.dayofweek
                df["month"] = df["createdAt"].dt.month
                df["img_count"] = df["media"].apply(lambda arr: sum(1 for i in (arr or []) if i.get("type") == "image"))
                df["video_count"] = df["media"].apply(lambda arr: sum(1 for i in (arr or []) if i.get("type") == "video"))
                df["first_img"] = df["media"].apply(lambda arr: next((i.get("url") for i in (arr or []) if i.get("type") == "image"), None))
                df["first_video"] = df["media"].apply(lambda arr: next((i.get("videoUrl") or i.get("url") for i in (arr or []) if i.get("type") == "video"), None))

                base_df = pd.concat([base_df, df], ignore_index=True)

                # агрегируем топики
                for row in df.itertuples(index=False):
                    new_topics = [t.strip() for t in row.topics_str.split(",") if t.strip()]
                    votes = int(row.votesCount or 0)
                    comments = int(row.commentsCount or 0)
                    # наивный вариант — весь вклад поста в каждый его топик
                    for t in new_topics:
                        s = topics.setdefault(t, {"sum_votes": 0.0, "sum_comments": 0.0, "count_posts": 0})
                        s["sum_votes"] += votes
                        s["sum_comments"] += comments
                        s["count_posts"] += 1

                # пагинация + чекпоинт
                if posts["pageInfo"]["hasNextPage"]:
                    cursor = posts["pageInfo"]["endCursor"]
                    save_state(cursor, fetched_total)
                    if fetched_total >= args.limit:
                        break
                    time.sleep(args.sleep)
                else:
                    cursor = None
                    save_state(cursor, fetched_total)
                    logger.info("[page] no more pages.")
                    break

        except KeyboardInterrupt:
            logger.warning("[STOP] interruption by user. Saving progress…")
            save_state(cursor, fetched_total)
            try:
                tmp_df = base_df.drop_duplicates(subset="id", keep="first")
                tmp_df.to_excel("output.xlsx", index=False)
                write_topics_merged(topics, "output_topics.xlsx")
                logger.info("[STOP] partial files written: output.xlsx, output_topics.xlsx")
            except Exception as e:
                logger.error(f"[STOP] failed to write partial results: {e}")

        # финализация
        try:
            base_df = base_df.drop_duplicates(subset="id", keep="first")
            base_df.to_excel("output.xlsx", index=False)
            write_topics_merged(topics, "output_topics.xlsx")
            logger.info("final files written: output.xlsx, output_topics.xlsx")
        except Exception as e:
            logger.error(f"final write failed: {e}")

    logger.info(f"Done. Fetched {fetched_total} posts.")

if __name__ == "__main__":
    main()
