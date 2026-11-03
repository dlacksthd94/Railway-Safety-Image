# Railway Safety

## I'm currently still putting things together!

## How It Works
- On the backend, a persistent news-monitoring bot continuously crawls news articles reporting train accidents. The scraped articles undergo a filtering process to retain only recent train–vehicle or train–pedestrian incidents. For each article, an LLM extracts the relevant details required for every field of Form 57.
- On the frontend, once the user adjusts the date-range filter in the sidebar and selects an article, the system displays the article’s original webpage on the left panel and its corresponding retrieved information on the right panel.

## Upcoming updates:
- Multiple articles referring to the same accident will be aggregated and presented as a single group.
- Scraped articles will be periodically re-checked to detect updates or newly published information.

## Data

### mapillary

`image_seq/`
- Street-level images
- `df_image_seq.csv`
    - Metadata table for `image_seq/`
    - Columns:
        - `crossing_id`: Crossing ID.
        - `seq_id`: ID for each group of image sequences (matches folder names).
        - `img_pos`: Position index of the image within a sequence.
        - `img_id`: Image ID (matches filename suffix).
        - `bearing`: Compass direction of the camera facing toward the center of the crossing.
- *[92 image sequence folders]*
    - Named with 22-digit sequence ID (e.g. `0Ct3ZUQwHMklErx1YDm4fA/`)
    - *[a sequence of images]*
        - Named with `[img_pos]_[img_id].jpg` (e.g. `0834_1401964997887726.jpg`)
        - Within a range of 10~20m from a railgrade crossing
