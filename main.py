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
CONFIDENCE_THRESHOLD = 0.75  # Threshold for unknown speaker detection

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
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a")):
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
async def identify(file: UploadFile = File(...), top_k: int = 3, threshold: float = CONFIDENCE_THRESHOLD, fallback_speaker: str = "Unknown Speaker"):
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
    
    exec_time = time.time() - start_time
    print(f"Identity check execution time: {exec_time:.3f} seconds")
    
    if not search_results:
        return JSONResponse(content={
            "best_match": fallback_speaker,
            "score": 0.0,
            "confidence": "low",
            "reason": "No identities in database",
            "raw_results": [],
            "execution_time": exec_time
        })
    
    # Group scores by identity name
    matches = {}
    for point in search_results:
        name = point.payload.get("name", "unknown")
        matches.setdefault(name, []).append(point.score)
    
    # IMPROVED: Use max score instead of average for each identity
    best_matches = {name: max(scores) for name, scores in matches.items()}
    best_match = max(best_matches.items(), key=lambda x: x[1])
    
    best_name = best_match[0]
    best_score = best_match[1]
    
    # Apply threshold check
    if best_score < threshold:
        return JSONResponse(content={
            "best_match": fallback_speaker,
            "score": best_score,
            "confidence": "low",
            "reason": f"Best match '{best_name}' score {best_score:.3f} below threshold {threshold}",
            "raw_results": [
                {"id": p.id, "name": p.payload.get("name", "unknown"), "score": p.score} for p in search_results
            ],
            "execution_time": exec_time
        })
    
    # Determine confidence level
    if best_score >= 0.85:
        confidence = "high"
    elif best_score >= 0.75:
        confidence = "medium"
    else:
        confidence = "low"
    
    return JSONResponse(content={
        "best_match": best_name,
        "score": best_score,
        "confidence": confidence,
        "reason": f"Identified with {len(matches[best_name])} sample(s)",
        "raw_results": [
            {"id": p.id, "name": p.payload.get("name", "unknown"), "score": p.score} for p in search_results
        ],
        "all_matches": {name: f"{max_score:.3f} (from {len(scores)} samples)" 
                       for name, (max_score, scores) in zip(best_matches.keys(), 
                                                           [(max(scores), scores) for scores in matches.values()])},
        "execution_time": exec_time
    })
