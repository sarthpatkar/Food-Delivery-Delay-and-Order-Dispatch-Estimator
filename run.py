import os

from web.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Delivery Delay Estimator running at http://127.0.0.1:{port} (models train on the first page load)")
    app.run(host="127.0.0.1", port=port, debug=False)
