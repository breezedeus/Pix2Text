package:
	python setup.py sdist bdist_wheel

VERSION = 0.0.1
upload:
	python -m twine upload  dist/pix2text-$(VERSION)* --verbose

# 开启 OCR HTTP 服务
serve:
	cnocr serve -p 8501 --reload

# 开启监控截屏文件夹的守护进程
daemon:
	python scripts/screenshot_daemon.py

docker-build:
	docker build -t breezedeus/cnocr:v$(VERSION) .

.PHONY: package upload serve daemon
