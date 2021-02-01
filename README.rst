uvicorn app:app --port 8000
curl -X POST "https://bert-bqjlnzid4q-uc.a.run.app/search/?top=10" -H  "accept: application/json" -H  "Content-Type: application/json" -d @sample.json
curl -X POST "http://127.0.0.1:8000/search/?top=10" -H  "accept: application/json" -H  "Content-Type: application/json" -d @sample.json
