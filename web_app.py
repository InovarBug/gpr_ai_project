
from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the GPR AI Web Interface"}

@app.post("/run_gpr_ai")
def run_gpr_ai():
    try:
        subprocess.Popen(["python3", "gpr_ai.py"])
        return {"status": "GPR AI is running"}
    except Exception as e:
        return {"status": "Error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
