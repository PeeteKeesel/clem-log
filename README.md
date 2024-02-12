# Clem'Log

[https://peetekeesel.github.io/clem-log/](https://peetekeesel.github.io/clem-log/)



One can test and change the page locally by first changing the `__config.yml` to

```yaml
title: Clem's Log
name: "Clem's Logs"
author: Clem
# baseurl: "/clem-log"
description: Learning notes.
profile_url: "https://peetekeesel.github.io"

# Build settings
markdown: kramdown
highlighter: rouge
excerpt_separator: <!--more-->
plugins:
  - jekyll-feed
  - jekyll-archives
exclude:
  - Gemfile
  - Gemfile.lock
  - vendor/bundle
kramdown:
  input: GFM
  auto_ids: true
  syntax_highlighter: rouge
  toc_levels: 1..4
```

and then running

```bash
# (Optional) Install necessary dependencies
bundle install

# Build and serve your site locally
bundle exec jekyll serve
```

Then open `http://localhost:4000/clem-log`. Changing back to the actual page needs the following `__config.yml` 

```yaml
title: Clem's Log
name: "Clem's Logs"
author: Clem
baseurl: "/clem-log"
description: Learning notes.
profile_url: "https://peetekeesel.github.io"

# Build settings
markdown: kramdown
highlighter: rouge
excerpt_separator: <!--more-->
plugins:
  - jekyll-feed
  - jekyll-archives
exclude:
  - Gemfile
  - Gemfile.lock
kramdown:
  input: GFM
  auto_ids: true
  syntax_highlighter: rouge
  toc_levels: 1..4
```
