#!/usr/bin/env python3

import argparse
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
import pandas as pd


class RedditRefinedPipeline:
    def __init__(self, min_len=10, drop_bots=True):
        self.min_len = min_len
        self.drop_bots = drop_bots
        
        self.bot_keywords = {'automoderator', 'remindmebot', 'moderator', 'preview-bot'}
        
        # Pre-compiled high-speed regex engines
        self.md_link_pattern = re.compile(r'\[([^\]]+)\]\([^)]+\)')
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.entity_pattern = re.compile(r'[ru]/([A-Za-z0-9_-]+)')
        self.whitespace_pattern = re.compile(r'\s+')
        self.blockquote_pattern = re.compile(r'^>.*$', re.MULTILINE)

        # Standard HTML entity translation maps
        self.html_replacements = [
            ('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'), ('&quot;', '"')
        ]

    def clean_text(self, text):
        """Universal cleaning vector for structural Markdown strings, URLs, and HTML codes."""
        if not text or not isinstance(text, str):
            return "", 0
        
        # Count and extract blockquotes before they are stripped
        quotes = len(self.blockquote_pattern.findall(text))
        text = self.blockquote_pattern.sub('', text)
        
        # Strip syntactic elements
        text = self.md_link_pattern.sub(r'\1', text)
        text = self.url_pattern.sub('', text)
        text = self.entity_pattern.sub(r'\1', text)
        
        # Convert HTML fragments back to native characters
        for html_ent, raw_char in self.html_replacements:
            text = text.replace(html_ent, raw_char)
            
        # Standardize spacing profiles
        text = self.whitespace_pattern.sub(' ', text).strip()
        return text, quotes

    def extract_emojis(self, text):
        """Non-destructively isolates emojis away from native text prose."""
        if not text:
            return "", ""
        emojis = []
        clean_chars = []
        for char in text:
            if unicodedata.category(char) in ('So', 'Sk'):
                emojis.append(char)
            else:
                clean_chars.append(char)
        return "".join(clean_chars), "".join(emojis)

    def calculate_lexical_diversity(self, text):
        """Computes vocabulary depth ratio to catch low-effort spam arrays."""
        words = text.lower().split()
        if not words:
            return 0.0
        return round(len(set(words)) / len(words), 2)

    def parse_stream(self, input_path):
        """Generates processed records line-by-line."""
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    
                    # 1. Automated Bot Account Elimination
                    author = data.get('author', 'Unknown')
                    if self.drop_bots:
                        if author.lower() in self.bot_keywords or any(b in author.lower() for b in ['-bot', '_bot']):
                            continue
                    
                    # 2. Handles Submissions vs Comments
                    is_post = 'title' in data
                    raw_text = ""
                    
                    if is_post:
                        title = data.get('title', '').strip()
                        selftext = data.get('selftext', '').strip()
                        if selftext in ('[deleted]', '[removed]'):
                            selftext = ""
                        raw_text = f"{title} [SEP] {selftext}" if selftext else title
                    else:
                        raw_text = data.get('body', '').strip()
                        if raw_text in ('[deleted]', '[removed]'):
                            continue
                    
                    # 3. Refinement and Extraction Sequences
                    base_clean, quote_count = self.clean_text(raw_text)
                    final_text, emojis = self.extract_emojis(base_clean)
                    
                    # Apply quality minimum text filter
                    if len(final_text) < self.min_len:
                        continue
                        
                    # 4. Standardize Temporal Formats
                    utc_ts = data.get('created_utc')
                    dt_str = datetime.fromtimestamp(int(utc_ts), timezone.utc).strftime('%Y-%m-%d %H:%M:%S') if utc_ts else ""
                    
                    # 5. Graph Placement Identifiers
                    parent_id = data.get('parent_id', '')
                    is_root = parent_id.startswith('t3_') if (not is_post and parent_id) else None
                    
                    # Align keys dynamically for structural matches
                    link_id = data.get('link_id', '')
                    clean_link_id = link_id.replace('t3_', '') if link_id else None
                    
                    yield {
                        "id": data.get("id"),
                        "type": "post" if is_post else "comment",
                        "timestamp": dt_str,
                        "created_utc_raw": utc_ts,
                        "author": author,
                        "score": data.get("score", 0),
                        "parent_id": parent_id if not is_post else None,
                        "link_id": link_id if not is_post else None,
                        "clean_link_id": clean_link_id,
                        "is_root_reply": is_root,
                        "text": final_text,
                        "emojis": emojis if emojis else None,
                        "quote_count": quote_count,
                        "lexical_diversity": self.calculate_lexical_diversity(final_text),
                        "num_comments": data.get("num_comments") if is_post else None,
                        "flair_text": data.get("link_flair_text") if is_post else None
                    }
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

    def run(self, input_path, output_path):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Source JSONL file not found at: {input_path}")
            
        print(f"⏳ Initiating 'Reddit Refined' pipeline processing for: {input_path}")
        df = pd.DataFrame(self.parse_stream(input_path))
        
        # Save output using columnar Parquet formatting
        df.to_parquet(output_path, index=False, compression='snappy')
        print(f"🎉 Process completed successfully! Saved {len(df)} rows to -> {output_path}")
        return df


def join_and_enrich_metrics(posts_parquet, comments_parquet, final_master_parquet):
    """Optional utility function showing users how to create a relational database link."""
    print("🔗 Initiating relational merge and context latency enrichments...")
    df_p = pd.read_parquet(posts_parquet)
    df_c = pd.read_parquet(comments_parquet)
    
    # Merge context features directly to corresponding message streams
    master_df = pd.merge(
        df_c,
        df_p[['id', 'flair_text', 'score', 'created_utc_raw']],
        left_on='clean_link_id',
        right_on='id',
        suffixes=('_comment', '_post')
    )
    
    # Calculate Latency Delay Metrics (Response Latency in minutes)
    if 'created_utc_raw_comment' in master_df.columns and 'created_utc_raw_post' in master_df.columns:
        master_df['response_latency_mins'] = (
            (master_df['created_utc_raw_comment'].astype(float) - master_df['created_utc_raw_post'].astype(float)) / 60
        ).round(1)
        
    master_df.to_parquet(final_master_parquet, index=False, compression='snappy')
    print(f"🎉 Fully relational master matrix compiled ({len(master_df)} records) -> {final_master_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reddit Refined Data Engineering Pipeline")
    parser.add_argument('--input', '-i', required=True, help="Input raw text data path (.jsonl file)")
    parser.add_argument('--output', '-o', required=True, help="Output destination target path (.parquet file)")
    parser.add_argument('--min-len', '-m', type=int, default=10, help="Minimum text extraction cutoff character threshold")
    parser.add_argument('--keep-bots', action='store_false', dest='drop_bots', help="Preserve default moderator script logs and bot comments")
    
    args = parser.parse_args()
    
    pipeline = RedditRefinedPipeline(min_len=args.min_len, drop_bots=args.drop_bots)
    pipeline.run(args.input, args.output)
