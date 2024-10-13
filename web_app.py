
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sqlite3
import pickle
import os
import tensorflow as tf

app = FastAPI()
app.mount("/static", StaticFiles(directory="gpr_ai_project/static"), name="static")
templates = Jinja2Templates(directory="gpr_ai_project/templates")

class Pattern(BaseModel):
    name: str
    description: str
    pattern: bytes

class AdvancedGPRAI:
    def __init__(self):
        self.cnn_model = self._build_cnn_model()
        self.db_path = 'gpr_patterns.db'
        self._create_db()
        self.load_patterns()
        self.knowledge_base = {}

    def _create_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns
            (id INTEGER PRIMARY KEY, name TEXT, description TEXT, pattern BLOB)
        ''')
        conn.commit()
        conn.close()

    def _build_cnn_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def add_pattern(self, name, pattern, description):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        pattern_blob = pickle.dumps(pattern)
        cursor.execute('INSERT INTO patterns (name, description, pattern) VALUES (?, ?, ?)',
                       (name, description, pattern_blob))
        conn.commit()
        conn.close()
        self.load_patterns()

    def load_patterns(self):
        self.patterns_db = {}
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, description, pattern FROM patterns')
        for row in cursor.fetchall():
            id, name, description, pattern_blob = row
            pattern = pickle.loads(pattern_blob)
            self.patterns_db[id] = {'name': name, 'pattern': pattern, 'description': description}
        conn.close()

    def delete_pattern(self, pattern_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM patterns WHERE id = ?', (pattern_id,))
        conn.commit()
        conn.close()
        self.load_patterns()

    def update_pattern(self, pattern_id, name, description):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE patterns SET name = ?, description = ? WHERE id = ?',
                       (name, description, pattern_id))
        conn.commit()
        conn.close()
        self.load_patterns()

ai = AdvancedGPRAI()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "patterns": ai.patterns_db})

@app.post("/add_pattern", response_class=HTMLResponse)
async def add_pattern(request: Request, name: str = Form(...), description: str = Form(...), pattern: bytes = Form(...)):
    ai.add_pattern(name, pattern, description)
    return templates.TemplateResponse("index.html", {"request": request, "patterns": ai.patterns_db})

@app.post("/delete_pattern", response_class=HTMLResponse)
async def delete_pattern(request: Request, pattern_id: int = Form(...)):
    ai.delete_pattern(pattern_id)
    return templates.TemplateResponse("index.html", {"request": request, "patterns": ai.patterns_db})

@app.post("/update_pattern", response_class=HTMLResponse)
async def update_pattern(request: Request, pattern_id: int = Form(...), name: str = Form(...), description: str = Form(...)):
    ai.update_pattern(pattern_id, name, description)
    return templates.TemplateResponse("index.html", {"request": request, "patterns": ai.patterns_db})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
