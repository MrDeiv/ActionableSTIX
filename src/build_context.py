from langchain_core.documents import Document

def build_context(documents:list[Document]) -> str:
    context = ""
    for doc in documents:
        context += doc.page_content + "\n\n"
    return context