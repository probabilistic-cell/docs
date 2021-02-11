all: compile clean build

build:
	jb build .

clean:
	jb clean .

compile:
	C:\Users\woute\Software\dart-sass\sass.bat _static/css/custom.scss _static/css/custom.css

publish:
	ghp-import -n -p -c latenta.org -f _build/html