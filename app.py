import streamlit as st
import ollama
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

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
                
                # PDF 로드 및 분할 (정교한 검색을 위해 조각 크기를 500으로 조정)
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                splits = text_splitter.split_documents(docs)
                
                # 임베딩 생성 및 벡터 DB 구축 (FAISS)
                # 가볍고 성능이 뛰어난 임베딩 전용 모델(nomic-embed-text)을 사용합니다.
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                vectorstore = FAISS.from_documents(splits, embeddings)
                
                # 상태 저장 (이제 텍스트 리스트 대신 벡터 DB 객체를 저장)
                st.session_state.docs_library[file_name] = vectorstore
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
            
        # 질문 기반으로 벡터 검색 (유사한 문서 조각 추출)
        vectorstore_or_splits = st.session_state.docs_library[current_doc]
        
        # 이전 버전(splits 리스트) 캐시 호환성 처리
        if isinstance(vectorstore_or_splits, list):
            context = "\n".join([d.page_content for d in vectorstore_or_splits[:3]])
        else:
            # 질문과 가장 유사한 내용 5개를 벡터 DB에서 검색하여 문맥으로 사용 (정확도 향상)
            retrieved_docs = vectorstore_or_splits.similarity_search(user_question, k=5)
            context = "\n".join([d.page_content for d in retrieved_docs])
            
        # Ollama에 보낼 메시지 구성
        # 문서 팩트 우선 + 외부 지식 보완의 균형 잡힌 프롬프트
        system_prompt = {
            'role': 'system', 
            'content': (
                f'당신은 문서의 내용을 기반으로 전문적인 답변을 제공하는 기술 파트너입니다.\n'
                f'지식 활용 우선순위:\n'
                f'1. [최우선]: 제공된 [참고 자료]의 팩트를 기반으로 답변하세요.\n'
                f'2. [보완]: [참고 자료]의 내용을 설명하기 위해 필요한 배경 지식이나 기술적 원리는 당신의 전문 지식을 활용해 풍부하게 덧붙이세요.\n'
                f'3. [확장]: 사용자가 추가 설명을 원할 경우, 문서의 맥락을 벗어나지 않는 선에서 외부 지식을 활용해 상세히 설명하세요.\n\n'
                f'규칙:\n'
                f'1. 반드시 한국어로 정중하게 답변하세요.\n'
                f'2. 가독성을 위해 소제목(##)과 강조(**text**)를 적절히 사용하세요.\n'
                f'3. 한자나 일본어는 절대 사용하지 마세요.\n\n'
                f'[참고 자료]\n{context}'
            )
        }
        
        ollama_messages = [system_prompt] + st.session_state.chat_histories[current_doc][-10:]
        
        # AI 답변 화면에 표시 및 저장
        with st.chat_message("assistant"):
            try:
                # 1. AI가 생각을 시작하는 동안만 스피너 표시
                with st.spinner("AI가 답변을 생성 중입니다..."):
                    stream = ollama.chat(
                        model='llama3.1', 
                        messages=ollama_messages, 
                        stream=True,
                        options={
                            'temperature': 0.3,      # 풍부한 설명을 위해 창의성을 약간 허용
                            'num_ctx': 2048,
                            'repeat_penalty': 1.2,
                        }
                    )
                    # 첫 번째 청크가 도착할 때까지 대기 (여기서 스피너가 유지됨)
                    first_chunk = next(stream)
                
                # 2. 첫 글자가 도착하면 여기서부터 타이핑 시작 (스피너는 위 블록을 벗어나며 사라짐)
                def generate_response():
                    content = first_chunk.get('message', {}).get('content', '')
                    if content:
                        yield content
                    for chunk in stream:
                        content = chunk.get('message', {}).get('content', '')
                        if content:
                            yield content
                            
                full_response = st.write_stream(generate_response())
                st.session_state.chat_histories[current_doc].append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                st.info("💡 팁: 이전 대화 내용이 너무 길어 메모리가 부족하거나 Ollama 서버가 멈췄을 수 있습니다.")

else:
    st.title("📚 나만의 로컬 AI 문서 비서")
    st.info("👈 왼쪽 사이드바에서 PDF 문서를 업로드하여 채팅을 시작하세요.")