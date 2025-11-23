# pmu_ai_server.py
# FastAPI + PillMeUp 모델 추론 서버

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import shutil
import os
import uuid

from pmu_model import infer_pills

app = FastAPI(
    title="PillMeUp AI Server",
    description="YOLO + ResNet 기반 다중 알약 item_seq 추론 API",
    version="1.0.0"
)

# -----------------------------------------------------
# 0. CORS 허용
# -----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Spring Boot / Flutter / React 등 모두 허용
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# 1. 헬스체크
# -----------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "AI server is running"}

# -----------------------------------------------------
# 2. 실제 추론 API
# -----------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # 확장자 추출
    ext = os.path.splitext(file.filename)[1].lower()
    if ext == "":
        ext = ".png"   # 확장자 없으면 PNG로 강제

    # temp 파일 이름 생성
    temp_path = f"{uuid.uuid4().hex}{ext}"

    # temp 파일 저장
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # AI 모델 예측
    result = infer_pills(temp_path, max_pills=4)

    # temp 삭제
    os.remove(temp_path)

    return {"result": result}

# -----------------------------------------------------
# 3. 로컬 실행 진입점
# -----------------------------------------------------
if __name__ == "__main__":
    # Uvicorn 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)
