docker run -it -p 7860:7860 -p 8001:8001 vincentqin/image-matching-webui:latest python app.py --server_name "0.0.0.0" --server_port=7860 && python -m imcui.api.flask_server
