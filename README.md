# 나만의 로컬 AI 문서 비서

PDF 파일을 업로드하면 AI가 내용을 분석하여 질문에 답변해주는 서비스

## 주요기능
- PDF 파일 업로드 및 텍스트 분할
- 연속 대화형 UI
- 스트리밍 기능 적용
- Ollama Llama 3.1 모델 사용

## 아키텍처
```
Streamlit (Web UI) -> LangChain (RAG Pipeline) -> Ollama (Llama 3.1 LLM)
```

## 사용법
```bash
uv run streamlit run app.py
```

## LLM 설정 
https://github.com/kubionet/llm-setup

