# AI Resume Screener + Resume Ranking System

## Project Overview
An AI-powered Resume Screening and Candidate Ranking System that helps recruiters analyze multiple resumes against a single Job Description (JD). It ranks candidates based on semantic similarity, extracts relevant skills, and provides a candidate summary with strengths, missing skills, and hiring recommendations.

## Features
- **Resume Upload System**: Upload multiple Resume PDFs and a Job Description (via Text or PDF).
- **Text Extraction**: Extracts text from uploaded PDF resumes using `PyPDF2` and `pdfplumber`.
- **NLP Preprocessing**: Tokenization, lowercase conversion, lemmatization, and stopword removal using `NLTK` and `spaCy`.
- **Semantic Embedding Generation**: Generates embeddings for resumes and JD using Sentence Transformers (`all-MiniLM-L6-v2`).
- **Resume Ranking**: Ranks candidates based on Cosine Similarity using `scikit-learn`.
- **Skill Extraction**: Extracts technical skills and compares present vs. missing skills.
- **AI Candidate Summary**: Generates rule-based hiring recommendations, strengths, and missing skills.
- **ATS-style Matching**: Displays Match %, Skills, Missing Skills, and Recommendation.

## Technologies Used
- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: scikit-learn
- **Deep Learning**: sentence-transformers
- **NLP**: NLTK, spaCy
- **PDF Processing**: PyPDF2, pdfplumber
- **Data Processing**: pandas, NumPy

## Requirements
See `pyproject.toml` for the full list of dependencies.

## Installation & How to Run

1. Clone or download this repository.
2. Navigate to the project directory:
   ```bash
   cd resume_screener
   ```
3. Create a virtual environment and install dependencies using `uv`:
   ```bash
   uv venv
   uv sync
   ```
4. Run the Streamlit application using the virtual environment:
   ```bash
   uv run streamlit run app.py
   ```

## Folder Structure
```text
resume_screener/
│
├── app.py
├── utils.py
├── pyproject.toml
├── README.md
├── resumes/
└── models/
```

## Workflow
1. Upload Job Description (Text or PDF)
2. Upload Resume PDFs
3. Extract PDF Text
4. NLP Preprocessing
5. Generate Sentence Embeddings
6. Compute Cosine Similarity
7. Rank Candidates
8. Extract Skills
9. Generate AI Summary
10. Display Results

## Sample Output
Candidate Name: Rahul Sharma  
Match Score: 87%  

Strengths  
* Python  
* SQL  
* Machine Learning  
* NLP  

Missing Skills  
* AWS  
* Docker  

Recommendation  
Strong candidate for technical interview.
