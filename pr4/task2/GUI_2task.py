import streamlit as st
import os
import numpy as np
import faiss
from deepface import DeepFace
from glob import glob
from tqdm import tqdm
import cv2

DATASET_PATH = "./hse_faces_miem/"
INDEX_FILE = "./face_index.faiss"
EMBEDDINGS_FILE = "./embeddings.npy"
NAMES_FILE = "./names.npy"
glob(os.path.join(DATASET_PATH, "*.jpeg"))

def create_embedding_database():
    images = glob(os.path.join(DATASET_PATH, "*.jpeg")) + glob(os.path.join(DATASET_PATH, "*.jpg")) + glob(
        os.path.join(DATASET_PATH, "*.png"))
    embeddings = []
    names = []

    progress_bar = st.progress(0)
    num_images = len(images)

    for i, img_path in enumerate(images):

        try:

            embedding = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            embeddings.append(embedding)
            names.append(os.path.basename(img_path))

        except Exception as e:
            st.error(f"Ошибка обработки {img_path}: {e}")

        progress_bar.progress((i + 1) / num_images)  

    embeddings = np.array(embeddings, dtype=np.float32)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    faiss.write_index(faiss_index, INDEX_FILE)

    np.save(EMBEDDINGS_FILE, embeddings)
    np.save(NAMES_FILE, names)

    st.success("База данных эмбеддингов создана.") 


def find_my_clone(image, faiss_index, embeddings, names, top_k=5):

    try:

        embedding = DeepFace.represent(image, model_name="Facenet", enforce_detection=False)[0]["embedding"]
    
    except Exception as e:
        st.error(f"Не удалось получить эмбеддинг для изображения: {e}")

        return []

    embedding = np.array([embedding], dtype=np.float32)
    D, I = faiss_index.search(embedding, top_k)
    results = []

    for i in range(len(I[0])):
        results.append((names[I[0][i]], D[0][i]))

    return results


def main():
    st.title("Поиск похожих лиц")

    if st.button("Пересоздать базу данных"):
        with st.spinner("Создаем базу данных..."):

            create_embedding_database()

            st.session_state.index = faiss.read_index(INDEX_FILE)
            st.session_state.embeddings = np.load(EMBEDDINGS_FILE)
            st.session_state.names = np.load(NAMES_FILE)

    if "index" not in st.session_state:

        if os.path.exists(INDEX_FILE):

            st.session_state.index = faiss.read_index(INDEX_FILE)
            st.session_state.embeddings = np.load(EMBEDDINGS_FILE)
            st.session_state.names = np.load(NAMES_FILE)

        else:

            st.warning("База данных не найдена. Пожалуйста, создайте ее.")
            return

    uploaded_file = st.file_uploader("Загрузите изображение для поиска:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        with open('result.jpeg', "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Ищем похожие лица..."):
            results = find_my_clone('result.jpeg', st.session_state.index, st.session_state.embeddings,
                                         st.session_state.names)

        st.subheader("Результаты поиска:")
        if results:
            for name, distance in results:
                st.write(f"**Имя файла:** {name}, **Расстояние:** {distance:.4f}")
                st.image(os.path.join(DATASET_PATH, name), width=150)
        else:
            st.write("Похожих лиц не найдено")

        os.remove('result.jpeg')


if __name__ == "__main__":
    main()
