#!/usr/bin/env python3
"""
Concurrent Nepali News Corpus Crawler
======================================

Architecture:

    URL discovery
          |
          v
    async downloaders
          |
          v
      HTML queue
          |
          v
    CPU extraction workers
          |
          v
   normalization/filter
          |
          v
      JSONL writer

This separates network I/O from CPU-heavy Trafilatura parsing.

Edit CONFIGURATION below and run:

    python nepali_news_crawler_v3.py

Install:

    pip install aiohttp beautifulsoup4 trafilatura lxml tqdm

IMPORTANT:
    Only crawl sites for which your collection is permitted.
    robots.txt is fetched first and the crawler fails closed if it
    cannot be retrieved.
"""

import asyncio
import hashlib
import json
import logging
import multiprocessing
import os
import re
import time
import unicodedata
import urllib.robotparser
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.parse import urljoin, urlparse

import aiohttp
import trafilatura
from bs4 import BeautifulSoup
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================
DOMAIN = "ratopati.com"

SEEDS = [
    "https://www.ratopati.com/",
]

OUTPUT_FILE = "Nepalese/data.nosync/onlinekhabar.jsonl"

# ------------------------------------------------------------
# NETWORKING
# ------------------------------------------------------------

# Number of concurrent HTTP download workers.
DOWNLOAD_WORKERS = 16

# Maximum simultaneous connections to this domain.
MAX_CONNECTIONS = 16

# Minimum delay between requests to this domain.
#
# Start conservatively. Increase throughput only after verifying
# the site handles the crawl appropriately.
REQUEST_DELAY = 0.20

REQUEST_TIMEOUT = 30

MAX_RETRIES = 3

RETRY_BACKOFF = 1.0

# ------------------------------------------------------------
# CPU extraction
# ------------------------------------------------------------

# Number of processes running Trafilatura + normalization.
#
# A good starting point on a modern CPU is:
#
#   number of physical cores - 1
#
# None -> automatically choose max(1, os.cpu_count() - 1)
EXTRACTION_WORKERS = None

# Number of downloaded pages allowed to wait for extraction.
EXTRACTION_QUEUE_SIZE = 100

# ------------------------------------------------------------
# Crawl
# ------------------------------------------------------------

MAX_PAGES = 100_000

MIN_CHARS = 300

MIN_DEVANAGARI_RATIO = 0.35

# If True, crawl same-domain links discovered from pages.
FALLBACK_LINK_CRAWL = True

# If True, use URLs discovered from sitemaps.
USE_SITEMAP_URLS = True

TRY_COMMON_SITEMAPS = True

MAX_LINKS_PER_PAGE = None

# ------------------------------------------------------------
# Debugging
# ------------------------------------------------------------

DEBUG_LINKS = False
DEBUG_LINK_LIMIT = 20

# ------------------------------------------------------------
# HTTP
# ------------------------------------------------------------

USER_AGENT = (
    "NepaliCorpusResearchBot/1.0 "
    "(academic language-resource collection; contact: sairithvik.achutuni@students.iiit.ac.in)"
)

ACCEPT_LANGUAGE = "ne-NP,ne;q=0.9,en;q=0.5"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("nepali-crawler")


# ============================================================
# URL HELPERS
# ============================================================

def normalize_domain(domain_or_url: str) -> str:
    value = domain_or_url.strip()

    if "://" in value:
        parsed = urlparse(value)
        hostname = parsed.hostname
    else:
        hostname = value.split("/")[0]

    if not hostname:
        raise ValueError(
            f"Could not determine hostname from: {domain_or_url}"
        )

    hostname = hostname.lower().rstrip(".")

    if hostname.startswith("www."):
        hostname = hostname[4:]

    return hostname


def normalize_url(url: str) -> str:
    url = url.strip()

    if not url:
        return url

    parsed0 = urlparse(url)

    # Reject non-web schemes before trying to access .port.
    if parsed0.scheme and parsed0.scheme.lower() not in {
        "http",
        "https",
    }:
        return url

    if not parsed0.scheme:
        url = "https://" + url
        parsed0 = urlparse(url)

    hostname = (parsed0.hostname or "").lower()

    if hostname.startswith("www."):
        hostname = hostname[4:]

    try:
        port = parsed0.port
    except ValueError:
        # Malformed URL.
        return url

    if port is not None:
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    return parsed0._replace(
        scheme=parsed0.scheme.lower(),
        netloc=netloc,
        path=parsed0.path or "/",
        fragment="",
    ).geturl()


def same_domain(url: str, domain: str) -> bool:
    hostname = urlparse(url).hostname

    if not hostname:
        return False

    hostname = hostname.lower()

    if hostname.startswith("www."):
        hostname = hostname[4:]

    return hostname == domain or hostname.endswith("." + domain)


def is_http_url(url: str) -> bool:
    return urlparse(url).scheme.lower() in {"http", "https"}


IGNORED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".mp4", ".m4v", ".mov", ".avi", ".mp3", ".wav",
    ".zip", ".rar", ".7z", ".gz", ".tar",
    ".css", ".js", ".woff", ".woff2", ".ttf",
    ".ico", ".pdf", ".doc", ".docx", ".xls", ".xlsx",
}


def is_ignored_file(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in IGNORED_EXTENSIONS)


EXCLUDED_PATH_PREFIXES = (
    "/search",
    "/tag/",
    "/tags/",
    "/author/",
    "/category/",
    "/page/",
    "/video/",
    "/videos/",
    "/photo/",
    "/photos/",
    "/gallery/",
    "/login",
    "/register",
)


def is_excluded_path(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.startswith(p) for p in EXCLUDED_PATH_PREFIXES)


# ============================================================
# RATE LIMITER
# ============================================================

class AsyncRateLimiter:
    """
    A single serialized request schedule per domain.

    This controls request initiation while allowing response
    processing to happen concurrently.
    """

    def __init__(self, delay: float):
        self.delay = max(0.0, delay)
        self.next_allowed = {}
        self.locks = {}

    async def wait(self, domain: str):
        if domain not in self.locks:
            self.locks[domain] = asyncio.Lock()

        async with self.locks[domain]:
            now = time.monotonic()
            next_time = self.next_allowed.get(domain, now)

            if next_time > now:
                await asyncio.sleep(next_time - now)

            # Schedule from the actual current time rather than from
            # an old timestamp, preventing bursts after slow requests.
            self.next_allowed[domain] = (
                time.monotonic() + self.delay
            )


# ============================================================
# ROBOTS
# ============================================================

class RobotsManager:

    def __init__(
        self,
        user_agent: str,
        session: aiohttp.ClientSession,
        limiter: AsyncRateLimiter,
        domain: str,
    ):
        self.user_agent = user_agent
        self.session = session
        self.limiter = limiter
        self.domain = domain

        self.parser = None
        self.loaded = False

    async def load(self) -> bool:
        robots_url = f"https://{self.domain}/robots.txt"

        try:
            await self.limiter.wait(self.domain)

            async with self.session.get(
                robots_url
            ) as response:

                if response.status != 200:
                    logger.error(
                        "Could not fetch robots.txt for %s: HTTP %d",
                        self.domain,
                        response.status,
                    )
                    return False

                text = await response.text(
                    errors="replace"
                )

        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
        ) as exc:

            logger.error(
                "Could not fetch robots.txt for %s: %s",
                self.domain,
                exc,
            )

            return False

        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(text.splitlines())

        self.parser = parser
        self.loaded = True

        logger.info(
            "Loaded robots.txt: %s",
            robots_url,
        )

        sitemaps = parser.site_maps() or []

        if sitemaps:
            logger.info(
                "robots.txt advertises %d sitemap(s)",
                len(sitemaps),
            )
        else:
            logger.info(
                "robots.txt does not advertise a sitemap"
            )

        return True

    def allowed(self, url: str) -> bool:
        if not self.loaded or self.parser is None:
            return False

        return self.parser.can_fetch(
            self.user_agent,
            url,
        )

    def sitemaps(self):
        if not self.loaded or self.parser is None:
            return []

        return self.parser.site_maps() or []


# ============================================================
# ASYNC HTTP CLIENT
# ============================================================

class AsyncHTTPClient:

    def __init__(
        self,
        session: aiohttp.ClientSession,
        limiter: AsyncRateLimiter,
        robots: RobotsManager,
        domain: str,
    ):
        self.session = session
        self.limiter = limiter
        self.robots = robots
        self.domain = domain

    async def get_text(self, url: str):
        if not self.robots.allowed(url):
            return None, "robots", None

        last_status = None

        for attempt in range(MAX_RETRIES + 1):

            await self.limiter.wait(self.domain)

            try:
                async with self.session.get(url) as response:

                    last_status = response.status

                    if response.status == 200:
                        content = await response.text(
                            errors="replace"
                        )

                        content_type = (
                            response.headers.get(
                                "Content-Type",
                                "",
                            ).lower()
                        )

                        return content, None, content_type

                    if response.status in {
                        408,
                        425,
                        429,
                        500,
                        502,
                        503,
                        504,
                    }:
                        if attempt < MAX_RETRIES:

                            backoff = (
                                RETRY_BACKOFF
                                * (2 ** attempt)
                            )

                            retry_after = (
                                response.headers.get(
                                    "Retry-After"
                                )
                            )

                            if retry_after:
                                try:
                                    backoff = max(
                                        backoff,
                                        float(retry_after),
                                    )
                                except ValueError:
                                    pass

                            logger.debug(
                                "HTTP %d: %s; retrying in %.1fs",
                                response.status,
                                url,
                                backoff,
                            )

                            await asyncio.sleep(
                                backoff
                            )

                            continue

                    return (
                        None,
                        f"http_{response.status}",
                        None,
                    )

            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
            ) as exc:

                if attempt < MAX_RETRIES:

                    backoff = (
                        RETRY_BACKOFF
                        * (2 ** attempt)
                    )

                    logger.debug(
                        "Request failed: %s; retrying in %.1fs",
                        exc,
                        backoff,
                    )

                    await asyncio.sleep(backoff)
                    continue

                return None, "request_error", None

        return None, f"http_{last_status}", None


# ============================================================
# UNICODE NORMALIZATION
# ============================================================

# These are treated as formatting artifacts.
REMOVE_CHARS = {
    "\ufeff",  # BOM / ZERO WIDTH NO-BREAK SPACE
    "\u200b",  # ZERO WIDTH SPACE
    "\u2060",  # WORD JOINER
    "\u00ad",  # SOFT HYPHEN
}

# Normalize special spaces.
SPACE_CHARS = {
    "\u00a0",
    "\u2009",
    "\u200a",
    "\u202f",
}


def normalize_nepali_text(text: str) -> str:

    if not text:
        return ""

    # Canonical Unicode composition.
    text = unicodedata.normalize("NFC", text)

    # Remove formatting artifacts.
    for ch in REMOVE_CHARS:
        text = text.replace(ch, "")

    # Normalize special spaces.
    for ch in SPACE_CHARS:
        text = text.replace(ch, " ")

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    cleaned = []

    for ch in text:

        # Preserve paragraph/newline structure.
        if ch in ("\n", "\t"):
            cleaned.append(ch)
            continue

        # IMPORTANT:
        # Preserve Devanagari ZWNJ and ZWJ.
        if ch in ("\u200c", "\u200d"):
            cleaned.append(ch)
            continue

        category = unicodedata.category(ch)

        # Remove remaining control/format characters.
        if category.startswith("C"):
            continue

        cleaned.append(ch)

    text = "".join(cleaned)

    lines = []

    for line in text.split("\n"):

        line = re.sub(
            r"[ \t]+",
            " ",
            line,
        ).strip()

        if line:
            lines.append(line)

    return "\n\n".join(lines).strip()


# ============================================================
# LANGUAGE FILTER
# ============================================================

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")


def devanagari_ratio(text: str) -> float:
    devanagari = len(
        DEVANAGARI_RE.findall(text)
    )

    latin = len(
        LATIN_RE.findall(text)
    )

    total = devanagari + latin

    if total == 0:
        return 0.0

    return devanagari / total


def looks_like_nepali(text: str) -> bool:

    if len(text) < MIN_CHARS:
        return False

    return (
        devanagari_ratio(text)
        >= MIN_DEVANAGARI_RATIO
    )


# ============================================================
# LINK EXTRACTION
# ============================================================

def extract_links(
    base_url: str,
    html: str,
    domain: str,
):
    soup = BeautifulSoup(
        html,
        "lxml",
    )

    links = set()

    for anchor in soup.find_all(
        "a",
        href=True,
    ):

        href = anchor["href"].strip()

        if not href:
            continue

        href_lower = href.lower()

        # Non-web links.
        if href_lower.startswith((
            "tel:",
            "mailto:",
            "javascript:",
            "whatsapp:",
            "sms:",
            "data:",
            "blob:",
            "#",
        )):
            continue

        try:
            absolute = urljoin(
                base_url,
                href,
            )

            parsed = urlparse(absolute)

            if parsed.scheme.lower() not in {
                "http",
                "https",
            }:
                continue

            url = normalize_url(absolute)

        except (
            ValueError,
            TypeError,
        ):
            continue

        if not is_http_url(url):
            continue

        if not same_domain(
            url,
            domain,
        ):
            continue

        if is_ignored_file(url):
            continue

        if is_excluded_path(url):
            continue

        links.add(url)

    return links


# ============================================================
# SITEMAPS
# ============================================================

def parse_sitemap(xml_text: str):

    soup = BeautifulSoup(
        xml_text,
        "xml",
    )

    if soup.find("sitemapindex"):
        urls = [
            x.get_text(strip=True)
            for x in soup.find_all("loc")
        ]

        return "sitemapindex", urls

    if soup.find("urlset"):
        urls = [
            x.get_text(strip=True)
            for x in soup.find_all("loc")
        ]

        return "urlset", urls

    urls = [
        x.get_text(strip=True)
        for x in soup.find_all("loc")
    ]

    if urls:
        return "urlset", urls

    return None, []


async def discover_sitemaps(
    domain: str,
    http: AsyncHTTPClient,
    robots: RobotsManager,
):

    candidates = []

    candidates.extend(
        robots.sitemaps()
    )

    if TRY_COMMON_SITEMAPS:

        candidates.extend([
            f"https://{domain}/sitemap.xml",
            f"https://{domain}/sitemap_index.xml",
            f"https://{domain}/sitemap-index.xml",
            f"https://{domain}/post-sitemap.xml",
            f"https://{domain}/news-sitemap.xml",
        ])

    candidates = [
        normalize_url(x)
        for x in dict.fromkeys(candidates)
    ]

    queue = deque(candidates)

    seen = set()
    article_urls = set()

    while queue:

        sitemap_url = queue.popleft()

        if sitemap_url in seen:
            continue

        seen.add(sitemap_url)

        if len(seen) > 1000:
            logger.warning(
                "Stopping sitemap traversal at 1000 sitemap files."
            )
            break

        xml, error, _ = await http.get_text(
            sitemap_url
        )

        if error is not None:
            logger.debug(
                "Unable to read sitemap %s: %s",
                sitemap_url,
                error,
            )
            continue

        kind, urls = parse_sitemap(xml)

        if kind == "sitemapindex":

            logger.info(
                "Sitemap index: %s (%d children)",
                sitemap_url,
                len(urls),
            )

            for child in urls:

                child = normalize_url(child)

                if same_domain(
                    child,
                    domain,
                ):
                    queue.append(child)

        elif kind == "urlset":

            logger.info(
                "Sitemap: %s (%d URLs)",
                sitemap_url,
                len(urls),
            )

            for url in urls:

                url = normalize_url(url)

                if same_domain(
                    url,
                    domain,
                ):

                    if not is_ignored_file(url):
                        article_urls.add(url)

    return list(article_urls)


# ============================================================
# CPU WORK
# ============================================================

def extract_and_process(
    item,
):
    """
    This function runs inside a separate process.

    Keep it as a top-level function so it can be spawned by
    ProcessPoolExecutor on macOS.
    """

    url, html = item

    try:

        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            include_links=False,
            include_images=False,
            favor_precision=True,
            output_format="txt",
        )

        if not text:
            return {
                "status": "empty",
                "url": url,
            }

        text = normalize_nepali_text(text)

        if not text:
            return {
                "status": "empty",
                "url": url,
            }

        title = ""
        date = None
        author = ""

        try:
            metadata = (
                trafilatura.extract_metadata(
                    html
                )
            )

            if metadata:

                title = (
                    metadata.title
                    or ""
                )

                # Coerce to str defensively -- trafilatura normally
                # returns a date string, but an unexpected type here
                # (e.g. a datetime object on some parse paths) would
                # otherwise crash json.dumps at write time.
                date = (
                    str(metadata.date)
                    if metadata.date
                    else None
                )

                author = (
                    metadata.author
                    or ""
                )

        except Exception:
            # Metadata is optional.
            pass

        title = normalize_nepali_text(
            title
        )

        author = normalize_nepali_text(
            author
        )

        if len(text) < MIN_CHARS:

            return {
                "status": "short",
                "url": url,
            }

        ratio = devanagari_ratio(
            text
        )

        if ratio < MIN_DEVANAGARI_RATIO:

            return {
                "status": "language",
                "url": url,
            }

        digest = hashlib.sha256(
            text.strip().encode("utf-8")
        ).hexdigest()

        return {
            "status": "ok",
            "id": digest,
            "source": normalize_domain(
                DOMAIN
            ),
            "url": url,
            "title": title,
            "date": date,
            "author": author,
            "language": "ne",
            "text": text,
            "num_chars": len(text),
            "devanagari_ratio": ratio,
        }

    except Exception as exc:

        return {
            "status": "error",
            "url": url,
            "error": repr(exc),
        }


# ============================================================
# CRAWLER
# ============================================================

class ConcurrentCrawler:

    def __init__(self):

        self.domain = normalize_domain(
            DOMAIN
        )

        timeout = aiohttp.ClientTimeout(
            total=REQUEST_TIMEOUT
        )

        connector = aiohttp.TCPConnector(
            limit=MAX_CONNECTIONS,
            limit_per_host=MAX_CONNECTIONS,
            ttl_dns_cache=300,
        )

        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Language":
                    ACCEPT_LANGUAGE,
                "Accept": (
                    "text/html,application/xhtml+xml,"
                    "application/xml;q=0.9,*/*;q=0.8"
                ),
            },
        )

        self.limiter = AsyncRateLimiter(
            REQUEST_DELAY
        )

        self.robots = RobotsManager(
            USER_AGENT,
            self.session,
            self.limiter,
            self.domain,
        )

        self.http = AsyncHTTPClient(
            self.session,
            self.limiter,
            self.robots,
            self.domain,
        )

        self.download_queue = asyncio.Queue()

        self.extraction_queue = asyncio.Queue(
            maxsize=EXTRACTION_QUEUE_SIZE
        )

        self.seen_urls = set()

        self.seen_hashes = set()

        self.writer = None

        self.stats = {
            "queued": 0,
            "downloaded": 0,
            "completed_downloads": 0,
            "extracted": 0,
            "saved": 0,
            "duplicates": 0,
            "rejected_short": 0,
            "rejected_language": 0,
            "empty": 0,
            "failed": 0,
            "robots_blocked": 0,
            "extraction_errors": 0,
        }

        # Number of URLs currently being processed by each stage.
        self.active_downloads = 0
        self.active_extractions = 0

    async def enqueue(
        self,
        url: str,
    ):

        url = normalize_url(url)

        if not url:
            return

        if not same_domain(
            url,
            self.domain,
        ):
            return

        if is_ignored_file(url):
            return

        if is_excluded_path(url):
            return

        if url in self.seen_urls:
            return

        if self.stats["queued"] >= MAX_PAGES:
            return

        self.seen_urls.add(url)

        self.stats["queued"] += 1

        await self.download_queue.put(url)

    async def downloader(
        self,
        worker_id: int,
    ):

        while True:

            url = await self.download_queue.get()
            self.active_downloads += 1

            try:

                html, error, content_type = (
                    await self.http.get_text(
                        url
                    )
                )

                if error == "robots":

                    self.stats[
                        "robots_blocked"
                    ] += 1

                elif error is not None:

                    self.stats[
                        "failed"
                    ] += 1

                elif html is not None:

                    self.stats[
                        "downloaded"
                    ] += 1

                    # Send to CPU extraction pool.
                    await self.extraction_queue.put(
                        (url, html)
                    )

                    # Link discovery is intentionally done in the
                    # download stage. BeautifulSoup link extraction
                    # is much cheaper than Trafilatura extraction.
                    if FALLBACK_LINK_CRAWL:

                        links = extract_links(
                            url,
                            html,
                            self.domain,
                        )

                        if (
                            MAX_LINKS_PER_PAGE
                            is not None
                        ):
                            links = set(
                                list(links)[
                                    :MAX_LINKS_PER_PAGE
                                ]
                            )

                        if DEBUG_LINKS:

                            logger.info(
                                "Extracted %d links from %s",
                                len(links),
                                url,
                            )

                            for link in list(
                                links
                            )[:DEBUG_LINK_LIMIT]:

                                logger.info(
                                    "LINK: %s",
                                    link,
                                )

                        for link in links:

                            if (
                                self.stats["queued"]
                                >= MAX_PAGES
                            ):
                                break

                            await self.enqueue(
                                link
                            )

            except Exception:

                logger.exception(
                    "Downloader %d failed on %s",
                    worker_id,
                    url,
                )

                self.stats["failed"] += 1

            finally:

                self.active_downloads -= 1
                self.stats[
                    "completed_downloads"
                ] += 1

                self.download_queue.task_done()

    async def extraction_worker(
        self,
        worker_id: int,
        executor: ProcessPoolExecutor,
    ):

        loop = asyncio.get_running_loop()

        while True:

            item = await self.extraction_queue.get()
            self.active_extractions += 1

            try:

                result = await loop.run_in_executor(
                    executor,
                    extract_and_process,
                    item,
                )

                self.handle_extraction_result(
                    result
                )

            except Exception:

                logger.exception(
                    "Extraction worker %d failed",
                    worker_id,
                )

                self.stats[
                    "extraction_errors"
                ] += 1

            finally:

                self.active_extractions -= 1
                self.stats[
                    "extracted"
                ] += 1

                self.extraction_queue.task_done()

    def handle_extraction_result(
        self,
        result,
    ):

        status = result.get("status")

        if status == "ok":

            digest = result["id"]

            if digest in self.seen_hashes:

                self.stats[
                    "duplicates"
                ] += 1

                return

            self.seen_hashes.add(
                digest
            )

            record = {
                "id": digest,
                "source": result[
                    "source"
                ],
                "url": result["url"],
                "title": result[
                    "title"
                ],
                "date": result[
                    "date"
                ],
                "author": result[
                    "author"
                ],
                "language": "ne",
                "text": result[
                    "text"
                ],
                "num_chars": result[
                    "num_chars"
                ],
                "devanagari_ratio":
                    result[
                        "devanagari_ratio"
                    ],
            }

            self.writer.write(
                record
            )

            self.stats[
                "saved"
            ] += 1

        elif status == "short":

            self.stats[
                "rejected_short"
            ] += 1

        elif status == "language":

            self.stats[
                "rejected_language"
            ] += 1

        elif status == "empty":

            self.stats[
                "empty"
            ] += 1

        elif status == "error":

            self.stats[
                "extraction_errors"
            ] += 1

            logger.debug(
                "Extraction error for %s: %s",
                result.get("url"),
                result.get("error"),
            )

    async def run(self):

        # Load state from any prior run of this crawl so we don't
        # re-fetch or re-save articles already collected. seen_urls
        # is seeded (not just seen_hashes) so already-visited pages
        # are skipped at the download stage too, not just deduped
        # after the fact -- saves bandwidth and time on resume.
        existing_hashes, existing_urls = load_existing_corpus_state(
            OUTPUT_FILE
        )
        self.seen_hashes |= existing_hashes
        self.seen_urls |= existing_urls

        self.writer = JSONLWriter(
            OUTPUT_FILE
        )

        try:

            logger.info(
                "Starting concurrent crawl for %s",
                self.domain,
            )

            if not await self.robots.load():

                logger.error(
                    "Stopping: robots.txt could "
                    "not be retrieved."
                )

                return

            sitemap_urls = []

            if USE_SITEMAP_URLS:

                sitemap_urls = (
                    await discover_sitemaps(
                        self.domain,
                        self.http,
                        self.robots,
                    )
                )

                logger.info(
                    "Discovered %d article URL candidates "
                    "from sitemaps.",
                    len(sitemap_urls),
                )

                for url in sitemap_urls:

                    if (
                        self.stats["queued"]
                        >= MAX_PAGES
                    ):
                        break

                    await self.enqueue(
                        url
                    )

            for seed in SEEDS:

                if (
                    self.stats["queued"]
                    >= MAX_PAGES
                ):
                    break

                await self.enqueue(
                    seed
                )

            if self.download_queue.empty():

                logger.warning(
                    "No URLs discovered."
                )

                return

            # --------------------------------------------
            # Extraction process pool
            # --------------------------------------------

            if EXTRACTION_WORKERS is None:

                extraction_workers = max(
                    1,
                    (os.cpu_count() or 2) - 1,
                )

            else:

                extraction_workers = (
                    EXTRACTION_WORKERS
                )

            logger.info(
                "Download workers: %d",
                DOWNLOAD_WORKERS,
            )

            logger.info(
                "HTTP connections: %d",
                MAX_CONNECTIONS,
            )

            logger.info(
                "Extraction processes: %d",
                extraction_workers,
            )

            logger.info(
                "Request delay: %.2fs",
                REQUEST_DELAY,
            )

            executor = ProcessPoolExecutor(
                max_workers=extraction_workers
            )

            download_workers = [
                asyncio.create_task(
                    self.downloader(i)
                )
                for i in range(
                    DOWNLOAD_WORKERS
                )
            ]

            extraction_workers_tasks = [
                asyncio.create_task(
                    self.extraction_worker(
                        i,
                        executor,
                    )
                )
                for i in range(
                    extraction_workers
                )
            ]

            # A dynamic queue means download workers can add
            # discovered pages while we crawl.
            progress = tqdm(
                total=MAX_PAGES,
                desc=self.domain,
                unit="page",
            )

            last_progress = 0

            try:

                # Keep the main coroutine alive while work is
                # flowing through either queue.
                while True:

                    await asyncio.sleep(
                        0.5
                    )

                    current = (
                        self.stats[
                            "extracted"
                        ]
                    )

                    if current > last_progress:

                        progress.update(
                            current
                            - last_progress
                        )

                        last_progress = (
                            current
                        )

                    progress.set_postfix(
                        queued=self.stats[
                            "queued"
                        ],
                        downloaded=self.stats[
                            "downloaded"
                        ],
                        saved=self.stats[
                            "saved"
                        ],
                        dup=self.stats[
                            "duplicates"
                        ],
                        failed=self.stats[
                            "failed"
                        ],
                    )

                    # Stop after all work has drained.
                    #
                    # We explicitly check active workers as well
                    # as queue emptiness, because a worker may have
                    # taken an item out of a queue while the queue
                    # itself is empty.
                    queues_empty = (
                        self.download_queue.empty()
                        and
                        self.extraction_queue.empty()
                    )

                    workers_idle = (
                        self.active_downloads == 0
                        and
                        self.active_extractions == 0
                    )

                    # Stop once both queues are empty and no worker
                    # is mid-task. This covers both the "crawl fully
                    # exhausted" case and the "MAX_PAGES reached,
                    # remaining work drained" case -- MAX_PAGES simply
                    # stops new enqueues (in enqueue()), so once that
                    # happens the queues naturally empty out and this
                    # same condition fires.
                    if (
                        queues_empty
                        and
                        workers_idle
                    ):
                        break

            finally:

                for task in (
                    download_workers
                    + extraction_workers_tasks
                ):
                    task.cancel()

                await asyncio.gather(
                    *download_workers,
                    *extraction_workers_tasks,
                    return_exceptions=True,
                )

                executor.shutdown(
                    wait=True,
                    cancel_futures=True,
                )

                progress.close()

            logger.info(
                "Finished crawl: %s",
                self.stats,
            )

        finally:

            self.writer.close()

    async def close(self):

        if not self.session.closed:
            await self.session.close()


# ============================================================
# JSONL WRITER
# ============================================================

class JSONLWriter:

    def __init__(
        self,
        path: str,
    ):

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Line-buffered append mode.
        self.file = self.path.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )

    def write(
        self,
        record: dict,
    ):

        self.file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

    def close(self):

        self.file.close()


def load_existing_corpus_state(path: str):
    """
    On startup, read any previously-saved JSONL output and return
    the set of content hashes and URLs already collected, so a
    restarted crawl doesn't re-fetch or re-save the same articles.

    Without this, JSONLWriter's append-mode file plus fresh in-memory
    seen_urls/seen_hashes sets on every run means every restart
    re-crawls from the seeds and appends duplicate records.
    """

    existing_hashes = set()
    existing_urls = set()

    file_path = Path(path)

    if not file_path.exists():
        return existing_hashes, existing_urls

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate a truncated last line from a prior
                # crash mid-write; skip rather than fail startup.
                continue

            if "id" in record:
                existing_hashes.add(record["id"])
            if "url" in record:
                existing_urls.add(normalize_url(record["url"]))

    logger.info(
        "Resuming from existing corpus: %d articles, %d URLs already collected",
        len(existing_hashes),
        len(existing_urls),
    )

    return existing_hashes, existing_urls


# ============================================================
# ENTRY POINT
# ============================================================

async def main():

    crawler = ConcurrentCrawler()

    try:
        await crawler.run()
    finally:
        await crawler.close()


if __name__ == "__main__":

    # Explicitly use spawn on macOS.
    multiprocessing.set_start_method(
        "spawn",
        force=True,
    )

    asyncio.run(main())