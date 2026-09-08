"""Local browser interface for Ask My Docs.

Run with: uv run uvicorn web:app --reload
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from main import ROOT, answer, ingest

DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", ROOT / "documents")).resolve()
DATABASE_DIR = Path(os.getenv("DATABASE_DIR", ROOT / ".rag_db")).resolve()
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

app = FastAPI(title="Ask My Docs")


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE


@app.post("/api/ask")
def ask_documents(payload: Question) -> dict[str, object]:
    try:
        response, sources = answer(payload.question.strip(), DATABASE_DIR, MODEL, EMBEDDING_MODEL, 4)
        return {"answer": response, "sources": sources}
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/ingest")
def index_documents() -> dict[str, str]:
    try:
        ingest(DOCUMENTS_DIR, DATABASE_DIR, EMBEDDING_MODEL, rebuild=False)
        return {"message": "Documents indexed successfully."}
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ask My Docs</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#101827;color:#e8edf7;font:16px system-ui,sans-serif}.wrap{max-width:850px;margin:8vh auto;padding:24px}h1{font-size:2.3rem;margin:0 0 8px}p{color:#aebbd0}.card{background:#182235;border:1px solid #2a3a55;border-radius:14px;padding:20px;margin-top:24px}form{display:flex;gap:10px}input{min-width:0;flex:1;padding:14px;border-radius:8px;border:1px solid #4c6387;background:#0d1523;color:white;font:inherit}button{border:0;border-radius:8px;padding:12px 18px;background:#5b9dff;color:#071120;font-weight:700;cursor:pointer}button:disabled{opacity:.55;cursor:wait}.secondary{background:#263853;color:#dce8fb;margin-top:14px}.answer{white-space:pre-wrap;line-height:1.55}.hidden{display:none}.error{color:#ffb4b4}.sources{margin:14px 0 0;padding-left:20px;color:#b8c9e7}small{color:#8fa4c5}</style></head>
<body><main class="wrap"><h1>Ask My Docs</h1><p>Ask questions about the documents in your local <code>documents</code> folder.</p>
<section class="card"><form id="question-form"><input id="question" aria-label="Your question" placeholder="What would you like to know?" required autofocus><button id="ask-button">Ask</button></form><button class="secondary" id="index-button">Index documents</button><small id="status"></small></section>
<section class="card hidden" id="result"><h2>Answer</h2><div class="answer" id="answer"></div><div id="source-area" class="hidden"><h3>Sources</h3><ul class="sources" id="sources"></ul></div></section>
</main><script>
const form=document.querySelector('#question-form'),question=document.querySelector('#question'),askButton=document.querySelector('#ask-button'),indexButton=document.querySelector('#index-button'),status=document.querySelector('#status'),result=document.querySelector('#result'),answer=document.querySelector('#answer'),sources=document.querySelector('#sources'),sourceArea=document.querySelector('#source-area');
async function call(url,body){const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});const data=await response.json();if(!response.ok)throw Error(data.detail||'Request failed');return data}
form.addEventListener('submit',async event=>{event.preventDefault();askButton.disabled=true;status.textContent='Searching documents and generating an answer…';status.className='';try{const data=await call('/api/ask',{question:question.value});answer.textContent=data.answer;result.classList.remove('hidden');sources.replaceChildren(...data.sources.map(source=>{const item=document.createElement('li');item.textContent=source;return item}));sourceArea.classList.toggle('hidden',!data.sources.length);status.textContent='';}catch(error){status.textContent=error.message;status.className='error'}finally{askButton.disabled=false}});
indexButton.addEventListener('click',async()=>{indexButton.disabled=true;status.textContent='Indexing documents…';status.className='';try{status.textContent=(await call('/api/ingest')).message}catch(error){status.textContent=error.message;status.className='error'}finally{indexButton.disabled=false}});
</script></body></html>"""
