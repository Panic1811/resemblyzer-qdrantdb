import os, time, random, shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from resemblyzer import preprocess_wav, VoiceEncoder
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

app = FastAPI()

COLLECTION_NAME = "identities"
VECTOR_SIZE = 256
IDENTITIES_DIR = "/data/identities"

os.makedirs(IDENTITIES_DIR, exist_ok=True)
encoder = VoiceEncoder("cpu")
qdrant = QdrantClient(host="qdrant", port=6333)

if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

def generate_id():
    return int(time.time() * 1000) + random.randint(0, 999)

@app.post("/upload_identity")
async def upload_identity(identity_name: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    # Save file to persistent storage
    file_path = os.path.join(IDENTITIES_DIR, f"{identity_name}_{int(time.time())}_{file.filename}")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        wav = preprocess_wav(file_path)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Error processing audio: {e}")

    embedding = encoder.embed_utterance(wav)
    point = PointStruct(
        id=generate_id(),
        vector=embedding.tolist(),
        payload={"name": identity_name, "file": file_path}
    )

    qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
    return JSONResponse(content={"status": "identity uploaded", "name": identity_name})

@app.post("/identify")
async def identify(file: UploadFile = File(...), top_k: int = 3):
    start_time = time.time()

    if not file.filename.lower().endswith((".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    temp_path = f"/tmp/{int(time.time())}_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        wav = preprocess_wav(temp_path)
    except Exception as e:
        os.remove(temp_path)
        raise HTTPException(status_code=400, detail=f"Error processing audio: {e}")

    os.remove(temp_path)
    embedding = encoder.embed_utterance(wav)

    search_results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding.tolist(),
        limit=top_k
    )

    if not search_results:
        exec_time = time.time() - start_time
        print(f"Identity check execution time: {exec_time:.3f} seconds")
        return JSONResponse(content={"message": "No identities found.", "execution_time": exec_time})

    matches = {}
    for point in search_results:
        name = point.payload.get("name", "unknown")
        matches.setdefault(name, []).append(point.score)

    avg_matches = {name: sum(scores)/len(scores) for name, scores in matches.items()}
    best_match = max(avg_matches.items(), key=lambda x: x[1])

    exec_time = time.time() - start_time
    print(f"Identity check execution time: {exec_time:.3f} seconds")

    return JSONResponse(content={
        "best_match": best_match[0],
        "score": best_match[1],
        "raw_results": [
            {"id": p.id, "name": p.payload.get("name", "unknown"), "score": p.score} for p in search_results
        ],
        "execution_time": exec_time
    })
