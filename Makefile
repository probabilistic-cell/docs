all: compile clean build

build:
	rsync -rav --update --delete --exclude '.ipynb_checkpoints' content/* _book/
	jb build --path-output ./ _book

clean:
	jb clean ./

compile:
	cp ../latenta_manuscript/manuscript/latenta_manuscript.json ./content
	sass content/_static/css/custom.scss content/_static/css/custom.css
	python scripts/convert.py

publish:
	ghp-import -n -p -c latenta.org -f _build/html