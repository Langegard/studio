#!/usr/bin/env bash
# Mutationskontroll: ändra en rad avsiktligt och visa att provet faller.
# En grön svit utan detta är ett påstående, inte ett bevis.
set -u
cd "$(dirname "$0")"
CSS=../fenestra.css
MD=../md.js
cp "$CSS" "$CSS.bak"; cp "$MD" "$MD.bak"
trap 'mv -f "$CSS.bak" "$CSS"; mv -f "$MD.bak" "$MD"' EXIT

fall=0
kor() { # namn, filter, förväntas RÖTT
  echo "== mutation: $1"
  if npx playwright test --grep "$2" --reporter=./utfall-reporter.js 2>/dev/null | grep -q '^RÖTT'; then
    echo "   provet föll som det ska"
  else
    echo "   PROVET FÖLL INTE — det kan inte bli rött"; fall=1
  fi
}

sed -i 's/--mal: 44px;/--mal: 20px;/' "$CSS"
kor "mål 44px → 20px" "B ·"
cp "$CSS.bak" "$CSS"

sed -i 's/--dampad: #9AA4B2;/--dampad: #4A5260;/' "$CSS"
kor "dämpad text sänkt under AA" "D ·"
cp "$CSS.bak" "$CSS"

sed -i 's|/\^(#{1,4})\\s+(.+)\$/|/^(#{1,6})\\s+(.+)$/|' "$MD"
kor "h5 släpps igenom" "G ·"
cp "$MD.bak" "$MD"

sed -i 's/font-size: 16px; color: var(--text); background: var(--panel);/font-size: 13px; color: var(--text); background: var(--panel);/' "$CSS"
sed -i 's/@media (pointer: coarse) { input, textarea, select { font-size: 16px; } }/@media (pointer: coarse) { input, textarea, select { font-size: 13px; } }/' "$CSS"
kor "fält 16px → 13px" "C ·"
cp "$CSS.bak" "$CSS"

exit $fall
