# 나만의 로컬 AI 문서 비서

PDF 파일을 업로드하면 AI가 내용을 분석하여 질문에 답변해주는 서비스

## 주요기능
- **멀티 PDF 관리:** 사이드바에서 여러 문서를 업로드하고 각 문서별로 독립적인 채팅 이력을 관리
- **정밀 RAG 시스템:** 500자 단위의 세밀한 청킹과 FAISS 기반 고속 검색 (k=5)
- **메모리 최적화:** 로컬 환경(노트북)을 고려하여 컨텍스트 창 최적화 설정을 적용한 8B 모델 사용
- **스트리밍 UI:** AI의 답변 과정을 실시간 타이핑 효과로 시각화

## 아키텍처
```
Streamlit (Web UI)
└── LangChain (RAG Pipeline)
    ├── FAISS (Vector DB)
    ├── OllamaEmbeddings (nomic-embed-text)
    └── Ollama (llama3.1 8B LLM - Memory Optimized)
```

## 사용법
```bash
uv run python -m streamlit run app.py
```
