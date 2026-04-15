import os
import glob
import pickle
from pathlib import Path

import streamlit as st
from openai import OpenAI
import numpy as np
from pypdf import PdfReader

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DOCS_DIR = Path(__file__).parent
INDEX_PATH = DOCS_DIR / ".rag_index.pkl"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4

client = OpenAI(api_key=OPENAI_API_KEY)


def read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        if suffix in {".txt", ".md", ".py", ".json", ".csv", ".html"}:
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        st.warning(f"읽기 실패 {path.name}: {e}")
    return ""


def chunk_text(text: str, source: str):
    text = " ".join(text.split())
    chunks = []
    i = 0
    while i < len(text):
        chunks.append({"source": source, "text": text[i : i + CHUNK_SIZE]})
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def collect_files():
    files = []
    for p in DOCS_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"} and not p.name.startswith("."):
            files.append(p)
    return sorted(files)


def build_index():
    files = collect_files()
    chunks = []
    for f in files:
        text = read_file(f)
        if text.strip():
            chunks.extend(chunk_text(text, f.name))
    if not chunks:
        return None
    embeddings = []
    batch = 64
    for i in range(0, len(chunks), batch):
        embeddings.append(embed([c["text"] for c in chunks[i : i + batch]]))
    embeddings = np.vstack(embeddings)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    index = {"chunks": chunks, "embeddings": embeddings, "files": [f.name for f in files]}
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    return index


@st.cache_resource
def load_index():
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "rb") as f:
            return pickle.load(f)
    return build_index()


def retrieve(query: str, index, k=TOP_K):
    q = embed([query])[0]
    q /= np.linalg.norm(q) + 1e-10
    sims = index["embeddings"] @ q
    top = np.argsort(-sims)[:k]
    return [(index["chunks"][i], float(sims[i])) for i in top]


def answer(query: str, history, index):
    hits = retrieve(query, index)
    context = "\n\n".join(f"[{i+1}] ({h['source']}) {h['text']}" for i, (h, _) in enumerate(hits))
    system = (
        "너는 업로드된 문서에 기반해 답하는 RAG 어시스턴트야.\n"
        "규칙:\n"
        "1. 아래 [컨텍스트]에 근거가 있을 때만 답변해. 컨텍스트로 답할 수 없거나 관련 내용이 없으면, "
        "다른 어떤 말도 붙이지 말고 정확히 이 한 줄만 출력해: \"해당 내용은 시설팀에 유선 문의 부탁드립니다\" "
        "(이 폴백 문장에는 아래 2,3,4번 규칙을 적용하지 않는다.)\n"
        "2. 답변은 반드시 국문과 영문을 동시에 출력해. 먼저 [국문] 섹션, 그 다음 [English] 섹션.\n"
        "3. 톤앤매너는 반말로 해.\n"
        "4. 답변의 매 문장 끝(마침표/물음표/느낌표 직전)에 접미사 'e1'을 붙여. 국문/영문 모두 적용.\n"
        "5. 답변 끝에 사용한 출처 번호를 [1][2] 형식으로 표기해.\n\n"
        f"[컨텍스트]\n{context}"
    )
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": query})
    stream = client.chat.completions.create(model=CHAT_MODEL, messages=messages, stream=True)
    return stream, hits


st.set_page_config(page_title="RAG Agent", page_icon="📚")
st.title("📚 RAG Agent")

with st.sidebar:
    st.header("문서 인덱스")
    if st.button("🔄 인덱스 재생성"):
        load_index.clear()
        with st.spinner("인덱싱 중..."):
            build_index()
        st.success("완료")
        st.rerun()
    index = load_index()
    if index:
        st.write(f"파일 {len(index['files'])}개 / 청크 {len(index['chunks'])}개")
        for n in index["files"]:
            st.caption(f"• {n}")
    else:
        st.warning("문서가 없습니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("문서에 대해 질문하세요"):
    if not index:
        st.error("인덱스가 비어 있습니다.")
        st.stop()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        stream, hits = answer(prompt, st.session_state.messages[:-1], index)
        placeholder = st.empty()
        full = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full += delta
            placeholder.markdown(full + "▌")
        placeholder.markdown(full)
        with st.expander("🔎 근거"):
            for i, (h, s) in enumerate(hits, 1):
                st.markdown(f"**[{i}] {h['source']}** (score={s:.3f})")
                st.caption(h["text"][:300] + "...")
    st.session_state.messages.append({"role": "assistant", "content": full})
