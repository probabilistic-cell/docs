cp ../latenta_manuscript/manuscript/latenta_manuscript.json ./content
sass content/_static/css/custom.scss content/_static/css/custom.css
python scripts/convert.py

rsync -rav --update --delete --exclude '.ipynb_checkpoints' content/* _book/
jb build --path-output ./ _book
