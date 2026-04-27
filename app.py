import streamlit as st
import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title("📚 나만의 로컬 AI 문서 비서")

# 1. 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

if uploaded_file:
    # 임시 파일 저장 및 로드
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    
    # 2. 텍스트 분할 (GPU 메모리 효율을 위해 적절히 자름)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    st.success(f"문서 로드 완료! (총 {len(splits)}개의 조각)")

    # 3. 대화 세션 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 기존 대화 내역 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 4. 질문 입력 (연속 대화형 UI)
    if user_question := st.chat_input("문서 내용에 대해 질문해보세요:"):
        # 사용자 질문 화면에 표시 및 저장
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Ollama에 보낼 메시지 구성 (시스템 프롬프트 + 문서 내용)
        context = "\n".join([d.page_content for d in splits[:3]]) # 우선 상위 3개 조각만 참조
        system_prompt = {'role': 'system', 'content': f'당신은 문서 도우미입니다. 다음 내용을 참고해서 대답하세요: {context}'}
        
        # 시스템 프롬프트와 지금까지의 대화 내역을 모두 합쳐서 전달
        ollama_messages = [system_prompt] + st.session_state.messages

        # AI 답변 화면에 표시 및 저장 (스트리밍 기능 적용)
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # 스트리밍 모드로 답변 받기 (글자가 타이핑되는 효과로 체감 속도 향상)
            for chunk in ollama.chat(model='llama3.1', messages=ollama_messages, stream=True):
                content = chunk.get('message', {}).get('content', '')
                if content:
                    full_response += content
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})