all: compile clean build

build:
	rsync -rav --update --delete --exclude '.ipynb_checkpoints' content/* _book/
	python scripts/build.py
	jb build --path-output ./ _book

clean:
	jb clean ./

compile:
	# cp ../latenta_manuscript/manuscript/v2/references.json ./content
	sass content/_static/css/custom.scss content/_static/css/custom.css
	python scripts/convert.py

publish:
	ghp-import -n -p -c latenta.org -f _build/html

serve:
	python -m http.server --directory _build/html

install_fonts:
	wget https://github.com/ipython/xkcd-font/blob/master/xkcd-script/font/xkcd-script.ttf -P /tmp/
