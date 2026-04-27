# 나만의 로컬 AI 문서 비서

PDF 파일을 업로드하면 AI가 내용을 분석하여 질문에 답변해주는 서비스

## 주요기능
- PDF 파일 업로드 및 문서별 독립 채팅방 (사이드바 관리)
- FAISS 벡터 데이터베이스 기반 RAG(검색 증강 생성) 시스템
- 연속 대화형 UI 및 스트리밍 답변 기능
- 노트북 환경에 최적화된 로컬 초경량 모델 사용 (Ollama)

## 아키텍처
```
Streamlit (Web UI)
└── LangChain (RAG Pipeline)
    ├── FAISS (Vector DB)
    ├── OllamaEmbeddings (nomic-embed-text)
    └── Ollama (qwen2.5:3b LLM)
```

## 사용법
```bash
uv run streamlit run app.py
```
