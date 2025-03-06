# Multi-Modal RAG System with Milvus, LLMs, and Human-in-the-Loop

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Component Details](#component-details)
   - [Data Ingestion](#data-ingestion)
   - [Content Processing](#content-processing)
   - [Embedding Generation](#embedding-generation)
   - [Vector Storage](#vector-storage)
   - [Retrieval System](#retrieval-system)
   - [LLM Integration](#llm-integration)
   - [User Interface](#user-interface)
   - [Governance Layer](#governance-layer)
4. [Implementation Guide](#implementation-guide)
   - [Environment Setup](#environment-setup)
   - [Implementing Data Ingestion](#implementing-data-ingestion)
   - [Document Processing](#document-processing)
   - [Media Processing](#media-processing)
   - [Setting Up Milvus](#setting-up-milvus)
   - [Building the Retrieval Pipeline](#building-the-retrieval-pipeline)
   - [Implementing the LLM Interface](#implementing-the-llm-interface)
   - [Creating the Gradio UI](#creating-the-gradio-ui)
   - [Governance Implementation](#governance-implementation)
5. [Deployment Guide](#deployment-guide)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

## System Overview

This multi-modal Retrieval Augmented Generation (RAG) system processes and indexes various types of media (text documents, PDFs, Word files, images, videos, and audio) to enable semantic search and AI-powered knowledge extraction. The system combines:

- Multi-modal document processing
- Chunking strategies for different media types
- Vector embedding for semantic search
- Milvus for scalable vector storage
- LLM integration for generation capabilities
- Human-in-the-loop governance for quality control
- Gradio for an intuitive user interface

The system enables organizations to build knowledge bases that can understand context across multiple media types, providing high-quality information retrieval with human oversight.

## Architecture

![System Architecture](https://placeholder-for-architecture-diagram.png)

The architecture follows these high-level components:

1. **Data Ingestion Layer**: Handles the intake of various file formats
2. **Content Processing Layer**: Extracts and processes content from each media type
3. **Embedding Generation Layer**: Creates vector embeddings for all content
4. **Vector Storage Layer**: Manages vector storage and retrieval using Milvus
5. **Retrieval Layer**: Implements retrieval logic based on semantic similarity
6. **LLM Integration Layer**: Connects to LLMs for generation tasks
7. **Governance Layer**: Provides human oversight and quality control
8. **User Interface Layer**: Gradio-based interface for end-users

## Component Details

### Data Ingestion

The data ingestion component handles various file formats:

#### Text-Based Documents
- **PDF Processing**: Extract text, maintain structure, and handle OCR when needed
- **Word Documents**: Parse DOCX/DOC formats preserving document structure
- **Plain Text**: Process TXT and similar formats

#### Media Files
- **Images**: Process JPG, PNG, and other common formats
- **Video**: Extract frames and audio from video files
- **Audio**: Process MP3, WAV, and other audio formats

### Content Processing

#### Document Chunking Strategies
The system implements intelligent chunking strategies based on content type:

- **Semantic Chunking**: Break documents by semantic units instead of arbitrary length
- **Hierarchical Chunking**: Maintain document hierarchy (headings, sections)
- **Contextual Chunking**: Preserve context between related chunks
- **Overlap Strategies**: Implement configurable overlap between chunks

Sample chunking code:
```python
def chunk_document(document, chunk_size=500, chunk_overlap=50):
    """
    Chunk a document into smaller pieces with overlap.
    
    Args:
        document (str): The document text
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of document chunks
    """
    chunks = []
    start = 0
    
    while start < len(document):
        # Take a chunk of text
        end = min(start + chunk_size, len(document))
        chunk = document[start:end]
        
        # Add chunk to list
        chunks.append(chunk)
        
        # Move start pointer with overlap
        start = end - chunk_overlap
        
    return chunks
```

#### Media Processing

- **Image Analysis**: Extract visual features and text (OCR) from images
- **Video Processing**: Extract key frames and transcribe audio
- **Audio Processing**: Transcribe speech to text

### Embedding Generation

#### Text Embeddings
- Generate dense vector representations using models like:
  - OpenAI Embeddings API
  - Sentence Transformers (BERT-based)
  - Custom embedding models

#### Image Embeddings
- Use CLIP for image embedding generation:

```python
from transformers import CLIPProcessor, CLIPModel

def generate_image_embeddings(image_path):
    """
    Generate embeddings for an image using CLIP.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: The image embedding vector
    """
    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load and process the image
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt", padding=True)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        
    # Convert to numpy array
    embedding = outputs.numpy()
    
    return embedding
```

#### Video Embeddings
- Extract key frames and apply CLIP
- Process audio track with Whisper for transcription

#### Audio Embeddings
- Use Whisper for transcription and then generate text embeddings:

```python
import whisper

def transcribe_audio(audio_path):
    """
    Transcribe audio using OpenAI's Whisper model.
    
    Args:
        audio_path (str): Path to the audio file
        
    Returns:
        str: Transcribed text
    """
    # Load Whisper model
    model = whisper.load_model("base")
    
    # Transcribe audio
    result = model.transcribe(audio_path)
    
    return result["text"]
```

### Vector Storage

#### Milvus Configuration
Milvus is used for scalable vector database storage:

```python
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

def setup_milvus_collection(collection_name, dimension):
    """
    Setup a Milvus collection for vector storage.
    
    Args:
        collection_name (str): Name of the collection
        dimension (int): Dimension of the vectors
        
    Returns:
        Collection: The Milvus collection object
    """
    # Connect to Milvus
    connections.connect("default", host="localhost", port="19530")
    
    # Check if collection exists
    if utility.has_collection(collection_name):
        return Collection(collection_name)
    
    # Define fields
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="media_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
    ]
    
    # Create schema
    schema = CollectionSchema(fields, description=f"Collection for {collection_name}")
    
    # Create collection
    collection = Collection(name=collection_name, schema=schema)
    
    # Create index for vector field
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    
    return collection
```

#### Collection Structure
- Design multiple collections or partitions based on media type
- Define appropriate index types for performance (e.g., HNSW, IVF_FLAT)
- Implement appropriate sharding strategy for large-scale deployments

### Retrieval System

#### Search Implementation
Implement hybrid search combining:
- Vector similarity search for semantic understanding
- Metadata filtering for additional context

```python
def search_milvus(collection, query_embedding, media_type=None, top_k=5):
    """
    Search for similar documents in Milvus.
    
    Args:
        collection: Milvus collection
        query_embedding: Query vector embedding
        media_type (str, optional): Filter by media type
        top_k (int): Number of results to return
        
    Returns:
        list: Search results
    """
    collection.load()
    
    # Construct search parameters
    search_params = {"metric_type": "L2", "params": {"ef": 64}}
    
    # Execute search
    if media_type:
        expr = f'media_type == "{media_type}"'
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["content", "source", "media_type"]
        )
    else:
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["content", "source", "media_type"]
        )
    
    return results
```

#### Multi-Modal Query Processing
- Process text queries directly with text embeddings
- Support image queries by encoding the image
- Process audio queries through transcription first

### LLM Integration

#### LLM Connection
- Support for multiple LLM providers:
  - OpenAI GPT models
  - Local open-source models
  - Custom fine-tuned models

```python
def generate_response(query, context, model_name="gpt-4"):
    """
    Generate a response using an LLM with retrieved context.
    
    Args:
        query (str): User query
        context (list): Retrieved context documents
        model_name (str): Name of the LLM to use
        
    Returns:
        str: Generated response
    """
    # Format context
    formatted_context = "\n\n".join([doc["content"] for doc in context])
    
    # Create prompt
    prompt = f"""
    Context information:
    {formatted_context}
    
    Question: {query}
    
    Please provide a detailed answer based on the context information:
    """
    
    # Call OpenAI API
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    
    return response.choices[0].message["content"]
```

#### RAG Implementation
- Optimize prompt templates for different query types
- Implement few-shot learning examples for better responses
- Handle source attribution and citations

### User Interface

#### Gradio Implementation
Implement a user-friendly interface using Gradio:

```python
import gradio as gr

def create_gradio_interface(search_function, generate_function):
    """
    Create a Gradio interface for the RAG system.
    
    Args:
        search_function: Function to search the knowledge base
        generate_function: Function to generate responses
        
    Returns:
        Gradio interface
    """
    # Define the interface
    with gr.Blocks() as demo:
        gr.Markdown("# Multi-Modal RAG System")
        
        with gr.Tab("Text Query"):
            text_input = gr.Textbox(label="Your Question")
            text_button = gr.Button("Search")
            text_output = gr.Textbox(label="Answer")
            
            # Show retrieved documents for transparency
            with gr.Accordion("Retrieved Documents", open=False):
                text_docs = gr.JSON(label="Sources")
        
        with gr.Tab("Image Query"):
            image_input = gr.Image(label="Upload Image")
            image_text = gr.Textbox(label="Additional Question (Optional)")
            image_button = gr.Button("Search")
            image_output = gr.Textbox(label="Answer")
            
        with gr.Tab("Audio Query"):
            audio_input = gr.Audio(label="Upload Audio")
            audio_button = gr.Button("Search")
            audio_output = gr.Textbox(label="Answer")
        
        # Add governance features
        with gr.Tab("Admin Panel"):
            with gr.Accordion("Human Review Queue", open=True):
                review_table = gr.Dataframe(
                    headers=["Query", "Response", "Sources", "Status"],
                    datatype=["str", "str", "str", "str"]
                )
                approve_button = gr.Button("Approve")
                reject_button = gr.Button("Reject")
        
        # Handle events and callbacks
        text_button.click(
            fn=lambda q: process_query(q, search_function, generate_function),
            inputs=text_input,
            outputs=[text_output, text_docs]
        )
        
        # Define similar handlers for image and audio
        
    return demo

def process_query(query, search_fn, generate_fn):
    """Process a text query through the RAG pipeline."""
    # Get query embedding
    embedding = get_embedding(query)
    
    # Search for relevant context
    search_results = search_fn(embedding)
    
    # Generate response
    response = generate_fn(query, search_results)
    
    return response, search_results

# Launch the interface
if __name__ == "__main__":
    demo = create_gradio_interface(search_milvus, generate_response)
    demo.launch()
```

### Governance Layer

#### Human-in-the-Loop Components
- Implement a review queue for responses
- Add feedback mechanisms for continuous improvement
- Create dashboards for system monitoring

```python
def add_to_review_queue(query, response, sources, priority="normal"):
    """
    Add a response to the human review queue.
    
    Args:
        query (str): User query
        response (str): Generated response
        sources (list): Retrieved documents used
        priority (str): Priority level (low, normal, high)
        
    Returns:
        str: Queue ID
    """
    # Connect to database
    conn = get_database_connection()
    
    # Create queue entry
    queue_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    entry = {
        "id": queue_id,
        "query": query,
        "response": response,
        "sources": sources,
        "priority": priority,
        "status": "pending",
        "timestamp": timestamp
    }
    
    # Add to database
    conn.execute(
        "INSERT INTO review_queue VALUES (:id, :query, :response, :sources, :priority, :status, :timestamp)",
        entry
    )
    
    return queue_id
```

#### Approval Workflows
- Define escalation paths for uncertain responses
- Implement correction mechanisms
- Maintain audit logs for compliance

## Implementation Guide

### Environment Setup

#### Required Dependencies
```bash
# Core dependencies
pip install pymilvus langchain gradio torch transformers pillow pytesseract python-docx PyPDF2

# Media processing
pip install openai-whisper opencv-python moviepy

# LLM integration
pip install openai
```

#### Docker Setup (Optional)
```yaml
# docker-compose.yml
version: '3'
services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.0
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2020-12-03T00-03-10Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.2.8
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

  gradio:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./app:/app
    depends_on:
      - "standalone"
```

### Implementing Data Ingestion

#### Document Processor Implementation

Create a DocumentProcessor class to handle various document types:

```python
import PyPDF2
import docx
import pytesseract
from PIL import Image

class DocumentProcessor:
    """Process various document types and extract text content."""
    
    def __init__(self):
        # Initialize OCR if needed
        pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update path as needed
    
    def process_pdf(self, file_path):
        """Extract text from PDF files."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        
        # If text extraction fails, try OCR
        if not text.strip():
            text = self._pdf_ocr(file_path)
        
        return text
    
    def _pdf_ocr(self, file_path):
        """OCR a PDF file if text extraction fails."""
        # Convert PDF to images and OCR
        # Implementation depends on specific requirements
        pass
    
    def process_docx(self, file_path):
        """Extract text from DOCX files."""
        doc = docx.Document(file_path)
        text = "\n\n".join([para.text for para in doc.paragraphs])
        return text
    
    def process_txt(self, file_path):
        """Extract text from plain text files."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
```

### Media Processing

#### Image Processor

```python
from PIL import Image
import pytesseract
from transformers import CLIPProcessor, CLIPModel

class ImageProcessor:
    """Process images and extract features and text."""
    
    def __init__(self):
        # Initialize models
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def extract_text(self, image_path):
        """Extract text from image using OCR."""
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    
    def generate_embedding(self, image_path):
        """Generate CLIP embedding for image."""
        image = Image.open(image_path)
        inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
        
        return outputs.numpy().flatten()
    
    def process_image(self, image_path):
        """Process image and return text and embedding."""
        text = self.extract_text(image_path)
        embedding = self.generate_embedding(image_path)
        
        return {
            "text": text,
            "embedding": embedding,
            "media_type": "image",
            "source": image_path
        }
```

#### Video Processor

```python
import cv2
import whisper
import numpy as np
from moviepy.editor import VideoFileClip

class VideoProcessor:
    """Process video files and extract frames and audio."""
    
    def __init__(self):
        # Initialize models
        self.whisper_model = whisper.load_model("base")
        self.image_processor = ImageProcessor()
    
    def extract_frames(self, video_path, max_frames=10):
        """Extract key frames from video."""
        video = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        # Calculate frame indices to extract
        interval = total_frames / min(max_frames, total_frames)
        frame_indices = [int(i * interval) for i in range(min(max_frames, total_frames))]
        
        frames = []
        for idx in frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = video.read()
            if ret:
                frames.append(frame)
        
        video.release()
        return frames
    
    def extract_audio(self, video_path, output_path):
        """Extract audio from video."""
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_path)
        return output_path
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper."""
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]
    
    def process_video(self, video_path):
        """Process video and return frames, audio transcription, and embeddings."""
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Save frames temporarily
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = f"temp_frame_{i}.jpg"
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        
        # Process frames
        frame_data = [self.image_processor.process_image(path) for path in frame_paths]
        
        # Extract and transcribe audio
        audio_path = f"temp_audio.mp3"
        self.extract_audio(video_path, audio_path)
        transcription = self.transcribe_audio(audio_path)
        
        # Clean up temporary files
        # ...
        
        return {
            "frames": frame_data,
            "transcription": transcription,
            "media_type": "video",
            "source": video_path
        }
```

#### Audio Processor

```python
import whisper
import numpy as np

class AudioProcessor:
    """Process audio files and extract transcriptions."""
    
    def __init__(self):
        # Initialize Whisper model
        self.model = whisper.load_model("base")
    
    def transcribe(self, audio_path):
        """Transcribe audio file."""
        result = self.model.transcribe(audio_path)
        return result["text"]
    
    def process_audio(self, audio_path):
        """Process audio and return transcription."""
        transcription = self.transcribe(audio_path)
        
        return {
            "text": transcription,
            "media_type": "audio",
            "source": audio_path
        }
```

### Setting Up Milvus

#### Milvus Collection Management

```python
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

class MilvusManager:
    """Manage Milvus collections and operations."""
    
    def __init__(self, host="localhost", port="19530"):
        # Connect to Milvus
        connections.connect("default", host=host, port=port)
        
        # Collection configurations
        self.collections = {
            "text": {
                "name": "text_collection",
                "dim": 1536,  # OpenAI embeddings dimension
                "fields": [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1000),
                    FieldSchema(name="chunk_id", dtype=DataType.INT64),
                    FieldSchema(name="media_type", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
                ]
            },
            "image": {
                "name": "image_collection",
                "dim": 512,  # CLIP embedding dimension
                "fields": [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1000),
                    FieldSchema(name="media_type", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
                ]
            }
        }
    
    def create_collections(self):
        """Create all collections if they don't exist."""
        for coll_type, config in self.collections.items():
            self.create_collection(config["name"], config["fields"])
    
    def create_collection(self, name, fields):
        """Create a collection with given name and fields."""
        if utility.has_collection(name):
            print(f"Collection {name} already exists")
            return Collection(name)
        
        # Create schema
        schema = CollectionSchema(fields, description=f"Collection for {name}")
        
        # Create collection
        collection = Collection(name=name, schema=schema)
        
        # Create index for vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        print(f"Created collection {name}")
        return collection
    
    def get_collection(self, name):
        """Get a collection by name."""
        if utility.has_collection(name):
            return Collection(name)
        return None
    
    def insert_data(self, collection_name, data):
        """Insert data into a collection."""
        collection = self.get_collection(collection_name)
        if collection is None:
            raise ValueError(f"Collection {collection_name} does not exist")
        
        # Insert data
        collection.insert(data)
        
        # Make sure the collection is loaded for search
        collection.load()
        
        return len(data)
    
    def search(self, collection_name, query_vectors, top_k=5, expr=None):
        """Search for similar vectors in a collection."""
        collection = self.get_collection(collection_name)
        if collection is None:
            raise ValueError(f"Collection {collection_name} does not exist")
        
        # Load collection
        collection.load()
        
        # Prepare search parameters
        search_params = {"metric_type": "L2", "params": {"ef": 64}}
        
        # Execute search
        results = collection.search(
            data=query_vectors,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["content", "source", "media_type"]
        )
        
        return results
```

### Building the Retrieval Pipeline

#### RAG Pipeline Implementation

```python
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGPipeline:
    """Implements the RAG pipeline for multi-modal content."""
    
    def __init__(self, milvus_manager, openai_api_key=None):
        self.milvus_manager = milvus_manager
        self.document_processor = DocumentProcessor()
        self.image_processor = ImageProcessor()
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
        
        # Initialize OpenAI for embeddings and LLM
        if openai_api_key:
            openai.api_key = openai_api_key
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def process_document(self, file_path, file_type=None):
        """Process a document and store in Milvus."""
        # Determine file type if not provided
        if file_type is None:
            file_type = file_path.split('.')[-1].lower()
        
        # Extract text based on file type
        if file_type in ['pdf']:
            text = self.document_processor.process_pdf(file_path)
            media_type = "pdf"
        elif file_type in ['docx', 'doc']:
            text = self.document_processor.process_docx(file_path)
            media_type = "docx"
        elif file_type in ['txt']:
            text = self.document_processor.process_txt(file_path)
            media_type = "txt"
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Chunk the text
        chunks = self.text_splitter.split_text(text)
        
        # Generate embeddings for each chunk
        data = []
        for i, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk)
            data.append({
                "content": chunk,
                "source": file_path,
                "chunk_id": i,
                "media_type": media_