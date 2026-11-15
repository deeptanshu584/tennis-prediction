---
name: scraper
description: Polite, cached web scraper for tennisabstract.com match data when no ready-made dataset covers the needed dates. Use only after the human has approved scraping.
tools: Read, Write, Edit, Bash, Glob, Grep, WebFetch
---
Read robots.txt first and obey it. One request every 3+ seconds. Cache every raw page under data/raw/scraped/ and never re-fetch a cached page.
Parse into the same columns as Sackmann's atp_matches_YYYY.csv (see PLAN.md section 4). Write tests against saved HTML fixtures.
Report coverage per tournament and any parsing failures. Never invent missing values.
