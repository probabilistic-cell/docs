all: compile clean build

build:
	python convert.py
	jb build .

clean:
	jb clean .

compile:
	sass _static/css/custom.scss _static/css/custom.css

publish:
	ghp-import -n -p -c latenta.org -f _build/html