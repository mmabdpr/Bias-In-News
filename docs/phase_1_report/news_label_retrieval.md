
# notes on retrieving data labels

## Steps to extract data

### From allsides.com

- install "SingleFile" extension
- go to this url: `https://www.allsides.com/media-bias/media-bias-ratings`
- press `End` key until all the data is loaded
- save page into one html file using "SingleFile" extension into `data/raw` directory
- copy and rename the html file into `data/intermediate/allsides.html`
- open the file using vscode and run `>Format Document`
- find `<tbody>...</tbody>` tag which contains main data and remove every thing before and after it
- run `>Format Document` again
- double check regular expressions given below (they should all find the same number of items. for me, it was 851)
  - select channel names: `<td class="views-field views-field-title source-title">\n.*<a href=.*>(.*\n?.*)</a>`
  - select bias ratings: `<td class="views-field views-field-field-bias-image">\n.*\n.*\n.*title="AllSides Media Bias Rating: (.*)"\n`
  - select community feedback: `<span class=agree>([0-9]+)</span>/<span class=disagree>([0-9]+)</span>`
- the code in `src/data/channels_utils.py` extracts data from this file using above regexps and returns a dataframe with 4 columns
  - channel_names \[str\]
  - bias_ratings \[str\] from \["Left", "LeanLeft", "Center", "LeanRight", "Right", "Mixed"\]
  - community_feedbacks_agree \[int\]
  - community_feedbacks_disagree \[int\]

### From adfontesmedia.com

- open this url using a Chromium-based browser: `https://www.adfontesmedia.com/interactive-media-bias-chart/`
- open network tab
- open stack trace of a random image file used in interactive chart
- find `chart.js` file and press reveal in sidebar
- data file is in `data` directory named `sourceData.js`
- my url: `https://imbc-public.d321epf4979uy6.amplifyapp.com/static/js/data/sourceData.js`
- download this file in `data/raw` directory
- copy and rename the file into `data/intermediate/adfontesmedia.json`
- open the file using vscode and clean the first, and the last line to have a nicely formatted json file
- use following regex to clean the file
  - find and replace `"all_metrics": \{\n.*"bias":(.*)\n.*"quality":(.*)\n.*"comparison":(.*)\n.*"expression":(.*)\n.*"headline":(.*)\n.*"poli_term":(.*)\n.*"veracity":(.*)\n.*"poli_pos":(.*)\n.*\}` with `"metrics_bias":$1\n  "metrics_quality":$2\n  "metrics_comparison":$3\n  "metrics_expression":$4\n  "metrics_headline":$5\n  "metrics_poli_term":$6\n  "metrics_veracity":$7\n  "metrics_poli_pos":$8`
- run `>Format Document`
- the code in `src/data/channels_utils.py` extracts data from this file and returns a dataframe with 20 columns
  - method `AdFontesMediaUtils.extract_data_from_json`:
    - article_id \[int\]
    - source_id \[int\]
    - url \[str\]
    - score_count \[int\]
    - domain \[str\]
    - source \[str\]
    - image_path \[str\]
    - reach \[int\]
    - mediatype \[int\]
    - article_count \[int\]
    - bias \[float\] between -42 and 42
    - quality \[float\] between 0 and 64
    - metrics_bias \[float\]
    - metrics_quality \[float\]
    - metrics_comparison \[float\]
    - metrics_expression \[float\]
    - metrics_headline \[float\]
    - metrics_poli_term \[float\]
    - metrics_veracity \[float\]
    - metrics_poli_pos \[float\]
  - method `AdFontesMediaUtils.extract_channels_data_from_json`:
    - source_id \[int\]
    - source \[str\]
    - domain \[str\]
    - reach \[int\]
    - mediatype \[int\]
    - article_count \[int\]
    - avg_bias \[float\] between -42 and 42
    - avg_quality \[float\] between 0 and 64

### From mediabiasfactcheck.com

- download this url using wget into `data/raw/mediabiasfactcheck.html`: `https://mediabiasfactcheck.com/filtered-search/`
- copy this file into `data/intermediate/mediabiasfactcheck.json`
- open it using vscode and run `>Format Document`
- find `<script>const filter_reporting =` tag and remove everything before and after the return statement in it to have a loosely formatted json array
- use following regexps to clean the file (keep in mind that all matching counts should be the same. mine was 3738)
  - find and replace `\n\tbias: "` with `\n\t"bias": "`
  - find and replace `", country: "` with `", "country": "`
  - find and replace `", credibility: "` with `", "credibility": "`
  - find and replace `", freporting:` with `", "freporting":`
  - find and replace `, traffic: "` with `, "traffic": "`
  - find and replace `": filter_reporting\[(.*)\], "` with `": $1, "`
  - find and replace ``row: \[\n\t\t`<td><a href="/?(.*)/?">(.*)</a>.*\n`` with `"link": "$1", "name": "$2"\n`
- run `>Format Document`
- the code in `src/data/channels_utils.py` extracts data from this file and returns a dataframe with 7 columns
  - bias \[str\] from \["Left", "Left-Center", "Least Biased", "Right-Center", "Right", "Questionable Sources", "Conspiracy-Pseudoscience", "Pro-Science", "Satire"\]
  - country \[str\]
  - credibility \[str\] from \["Low Credibility", "Medium Credibility", "High Credibility", "Unknown" \]
  - freporting \[str\] from \['Very Low', 'Low', 'High', 'Very High', 'Unknown', 'Mostly Factual', 'Mixed' \]
  - traffic \[str\] from \['Medium Traffic', 'High Traffic', 'Minimal Traffic', 'Unknown' \]
  - link \[str\]
  - name \[str\]
