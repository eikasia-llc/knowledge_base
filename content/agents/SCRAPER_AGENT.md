# Event Scraper Implementation Guide
- status: active
- type: guideline
<!-- content -->
This document defines the implementation patterns for the MCMP event scraper, including critical lessons learned from production usage.

---

## Critical: Dynamic Loading
- id: event_scraper_implementation_guide.critical_dynamic_loading
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->
> [!CAUTION]
> The events-overview page uses a **"Load more" button** to dynamically load events. Static `requests.get()` only captures 16 of 53+ events.

### Problem
- id: event_scraper_implementation_guide.critical_dynamic_loading.problem
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->
- Initial page load shows ~16 events
- Remaining events load via JavaScript when clicking "Load more"
- Button class: `button.filterable-list__load-more`
- Requires 4+ clicks to reveal all events

### Solution: Selenium
- id: event_scraper_implementation_guide.critical_dynamic_loading.solution_selenium
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->
```python
def _fetch_events_with_selenium(self, url):
    driver = webdriver.Chrome(options=headless_options)
    driver.get(url)
    
    # Click "Load more" until it disappears
    while True:
        try:
            btn = driver.find_element(By.CSS_SELECTOR, "button.filterable-list__load-more")
            if btn.is_displayed():
                btn.click()
                time.sleep(1)
            else:
                break
        except NoSuchElementException:
            break
    
    # Now extract all event links
    links = driver.find_elements(By.CSS_SELECTOR, "a.filterable-list__list-item-link.is-events")
```

**Dependencies**: `selenium`, `webdriver-manager`

---

## Website Structure
- id: event_scraper_implementation_guide.website_structure
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->

### Event Sources
- id: event_scraper_implementation_guide.website_structure.event_sources
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->
1. **Events Overview** (Primary): `https://www.philosophie.lmu.de/mcmp/en/latest-news/events-overview/` ⚠️ Dynamic
2. **Events Page**: `https://www.philosophie.lmu.de/mcmp/en/events/`
3. **Homepage**: `https://www.philosophie.lmu.de/mcmp/en/`

### DOM Structure
- id: event_scraper_implementation_guide.website_structure.dom_structure
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->
- **Listing pages**: Events in `<a>` tags with class `.filterable-list__list-item-link.is-events`
- **Individual event pages**:
  - `<h1>` with speaker/event name
  - `<h2>` labels for "Date:", "Location:", "Title:", "Abstract:"
  - Location in `<address>` tag

---

## Implementation Patterns
- id: event_scraper_implementation_guide.implementation_patterns
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->

### 1. Deduplication (URL-based)
- id: event_scraper_implementation_guide.implementation_patterns.1_deduplication_url_based
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->
```python
seen_urls = set()
for link in event_links:
    url = self._normalize_url(link['href'])
    if url not in seen_urls:
        seen_urls.add(url)
```

### 2. Event Details Extraction
- id: event_scraper_implementation_guide.implementation_patterns.2_event_details_extraction
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->
```python

# Labeled sections
- id: labeled_sections
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->
for h2 in soup.find_all('h2'):
    label = h2.get_text(strip=True).rstrip(':').lower()
    if label == 'abstract':
        event['abstract'] = self._extract_section_content(h2)

# Location from address tag
- id: location_from_address_tag
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->
address = soup.find('address')
if address:
    event['metadata']['location'] = address.get_text(' ', strip=True)
```

### 3. Date Parsing
- id: location_from_address_tag.3_date_parsing
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->
```python

# "4 February 2026" → "2026-02-04"
- id: 4_february_2026_2026_02_04
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->
match = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})', date_text)
```

---

## Output Schema
- id: 4_february_2026_2026_02_04.output_schema
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->
```json
{
    "title": "Talk: Simon Saunders (Oxford)",
    "url": "https://...",
    "talk_title": "Bell inequality violation is evidence for many worlds",
    "abstract": "Given two principles (a) no action-at-a-distance...",
    "metadata": {
        "date": "2026-02-04",
        "time_start": "4:00 pm",
        "location": "Ludwigstr. 31 Ground floor, room 021",
        "speaker": "Simon Saunders (Oxford)"
    }
}
```

---

## Verification
- id: 4_february_2026_2026_02_04.verification
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->
- [x] All 53+ events captured
- [x] Abstracts extracted from individual pages
- [x] No duplicate URLs
- [x] Dates in ISO format

## People Scraper Implementation
- id: 4_february_2026_2026_02_04.people_scraper_implementation
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->

### Sources
- id: 4_february_2026_2026_02_04.people_scraper_implementation.sources
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->
- **People Index**: `https://www.philosophie.lmu.de/mcmp/en/people/`
- **Profile Pages**: Individual pages linked from the index (e.g., `/people/contact-page/...`)

### DOM Structure (Profile Page)
- id: 4_february_2026_2026_02_04.people_scraper_implementation.dom_structure_profile_page
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->
| Field | Selector / Logic | Notes |
|-------|------------------|-------|
| **Name** | `h1.header-person__name` | Fallback to `h1` |
| **Position** | `p.header-person__job` | e.g., "Doctoral Fellow" |
| **Org Unit** | `p.header-person__department` | e.g., "Chair of Philosophy of Science" |
| **Email** | `a.header-person__contentlink.is-email` | Strip "Send an email", check `mailto:` |
| **Phone** | `a.header-person__contentlink.is-phone` | |
| **Room** | `div.header-person__detail_area p` | **CRITICAL**: Exclude text containing "Room finder" |
| **Image** | `img.picture__image` | Get `src` attribute |
| **Website** | `a` with text "Personal website" | |
| **Publications** | `h2` "Selected publications" | Parse sibling `ul` or `ol` lists |

### Output Schema
- id: 4_february_2026_2026_02_04.people_scraper_implementation.output_schema
- status: active
- type: context
- last_checked: 2026-02-02
<!-- content -->
```json
{
    "name": "Dr. Conrad Friedrich",
    "url": "https://...",
    "description": "Personal information...",
    "metadata": {
        "position": "Postdoctoral fellow",
        "organizational_unit": "MCMP",
        "email": "Conrad.Friedrich@lmu.de",
        "office_address": "Ludwigstr. 31",
        "room": "Room 225",
        "website": "https://conradfriedrich.github.io/",
        "image_url": "https://...",
        "selected_publications": ["Pub 1", "Pub 2"]
    }
}
```

---
