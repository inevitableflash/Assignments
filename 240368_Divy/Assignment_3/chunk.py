from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text_into_chunks(text: str, chunk_size: int = 50, chunk_overlap: int = 10):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split the text
    chunks = text_splitter.split_text(text)
    
    return chunks, len(chunks)