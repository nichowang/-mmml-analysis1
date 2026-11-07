#!/bin/bash
# render each idea2_slide*.html to a 1920x1080 PNG (headless viewport is 87px short, so render taller and crop)
CH=~/.cache/ms-playwright/chromium-1208/chrome-linux64/chrome
cd "$(dirname "$0")"
for f in idea2_slide.html; do
  b=${f%.html}
  $CH --headless=old --no-sandbox --disable-gpu --hide-scrollbars --virtual-time-budget=5000 --window-size=1920,1167 --screenshot=$PWD/$b.png file://$PWD/$f 2>/dev/null
  python3 -c "from PIL import Image; Image.open('$b.png').crop((0,0,1920,1080)).save('$b.png')"
done
