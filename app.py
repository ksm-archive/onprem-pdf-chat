import streamlit as st
import ollama
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="나만의 로컬 AI 문서 비서", page_icon="📚", layout="wide")

# 임시 PDF 저장 폴더 생성
if not os.path.exists("uploaded_pdfs"):
    os.makedirs("uploaded_pdfs")

# 1. 상태 관리(Session State) 초기화
if "docs_library" not in st.session_state:
    st.session_state.docs_library = {} # 파일명 -> 분할된 텍스트 청크들
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {} # 파일명 -> 대화 내역(messages) 리스트
if "current_doc" not in st.session_state:
    st.session_state.current_doc = None

# 2. 사이드바 (Sidebar) UI
with st.sidebar:
    st.title("📚 문서 라이브러리")
    
    # 파일 업로드
    uploaded_file = st.file_uploader("새로운 PDF 파일 업로드", type="pdf")
    if uploaded_file:
        file_name = uploaded_file.name
        # 아직 라이브러리에 없는 파일인 경우만 처리
        if file_name not in st.session_state.docs_library:
            with st.spinner(f"'{file_name}' 처리 중..."):
                file_path = os.path.join("uploaded_pdfs", file_name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # PDF 로드 및 분할
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                splits = text_splitter.split_documents(docs)
                
                # 상태 저장
                st.session_state.docs_library[file_name] = splits
                st.session_state.chat_histories[file_name] = []
                st.session_state.current_doc = file_name
            
            st.success(f"'{file_name}' 추가 완료!")
            
    st.divider()
    
    # 업로드된 문서 목록 (라디오 버튼)
    if st.session_state.docs_library:
        st.subheader("업로드된 문서")
        
        doc_names = list(st.session_state.docs_library.keys())
        
        # current_doc가 유효하지 않으면 첫 번째 문서로 지정
        if st.session_state.current_doc not in doc_names:
            st.session_state.current_doc = doc_names[0]
            
        current_index = doc_names.index(st.session_state.current_doc)
        
        selected_doc = st.radio(
            "대화할 문서를 선택하세요:",
            doc_names,
            index=current_index
        )
        st.session_state.current_doc = selected_doc
    else:
        st.info("업로드된 문서가 없습니다.")

# 3. 메인 화면 (Main Chat Area)
if st.session_state.current_doc:
    current_doc = st.session_state.current_doc
    st.title(f"💬 {current_doc} 대화방")
    
    # 기존 대화 내역 출력
    for msg in st.session_state.chat_histories[current_doc]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # 질문 입력
    if user_question := st.chat_input("문서 내용에 대해 질문해보세요:"):
        # 사용자 질문 화면에 표시 및 저장
        st.session_state.chat_histories[current_doc].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
            
        # Ollama에 보낼 메시지 구성
        splits = st.session_state.docs_library[current_doc]
        context = "\n".join([d.page_content for d in splits[:3]]) # 우선 상위 3개 조각만 참조
        system_prompt = {'role': 'system', 'content': f'당신은 문서 도우미입니다. 다음 내용을 참고해서 대답하세요: {context}'}
        
        ollama_messages = [system_prompt] + st.session_state.chat_histories[current_doc]
        
        # AI 답변 화면에 표시 및 저장
        with st.chat_message("assistant"):
            # 로딩 메시지 표시용 공간
            status_placeholder = st.empty()
            status_placeholder.info("⏳ AI가 답변을 생각하고 있습니다...")
            
            try:
                # 스트리밍 모드로 답변 받기 (제너레이터 함수 활용)
                stream = ollama.chat(model='llama3.1', messages=ollama_messages, stream=True)
                
                def generate_response():
                    first_chunk = True
                    for chunk in stream:
                        if first_chunk:
                            status_placeholder.empty() # 첫 글자 도착 시 로딩 메시지 지움
                            first_chunk = False
                            
                        content = chunk.get('message', {}).get('content', '')
                        if content:
                            yield content
                            
                # st.write_stream이 잔상 문제를 방지하고 타이핑 효과를 깔끔하게 처리합니다.
                full_response = st.write_stream(generate_response())
                
                st.session_state.chat_histories[current_doc].append({"role": "assistant", "content": full_response})
            except Exception as e:
                status_placeholder.empty()
                st.error(f"오류가 발생했습니다: {str(e)}")
                st.info("💡 팁: 이전 대화 내용이 너무 길어 메모리가 부족하거나 Ollama 서버가 멈췄을 수 있습니다.")

else:
    st.title("📚 나만의 로컬 AI 문서 비서")
    st.info("👈 왼쪽 사이드바에서 PDF 문서를 업로드하여 채팅을 시작하세요.")