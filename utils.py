import re
import PyPDF2
import pdfplumber
import spacy
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def setup_nlp():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    try:
        nlp_model = spacy.load("en_core_web_sm")
    except (OSError, ImportError):
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp_model = spacy.load("en_core_web_sm")
    return nlp_model

# Initialize NLP models globally
nlp = setup_nlp()
stop_words = set(stopwords.words('english'))
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        pass
    
    # Fallback to PyPDF2 if pdfplumber fails or returns empty text
    if not text.strip():
        try:
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception:
            pass
            
    return text

def preprocess_text(text):
    if not text:
        return ""
    
    # Lowercase conversion
    text = text.lower()
    
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Tokenization and Lemmatization
    doc = nlp(text)
    
    cleaned_tokens = []
    for token in doc:
        lemma = token.lemma_
        # Stopword removal
        if lemma not in stop_words and not token.is_space:
            cleaned_tokens.append(lemma)
            
    return " ".join(cleaned_tokens)

def generate_embedding(text):
    return model.encode(text)

def compute_similarity(emb1, emb2):
    return cosine_similarity([emb1], [emb2])[0][0]

def extract_skills(text):
    common_skills = {
        "python", "java", "c++", "c#", "javascript", "typescript", "html", "css",
        "sql", "nosql", "mongodb", "postgresql", "mysql", "react", "angular", "vue",
        "node.js", "express", "django", "flask", "spring", "machine learning", "deep learning",
        "nlp", "natural language processing", "computer vision", "data science", "pandas",
        "numpy", "scikit-learn", "tensorflow", "keras", "pytorch", "aws", "azure", "gcp",
        "docker", "kubernetes", "git", "github", "gitlab", "ci/cd", "agile", "scrum",
        "linux", "bash", "rest api", "graphql", "microservices", "statistics", "mathematics"
    }
    
    extracted = set()
    text_lower = text.lower()
    
    for skill in common_skills:
        # Check for whole word match
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            extracted.add(skill)
            
    return extracted

def generate_ai_summary(resume_skills, jd_skills, match_score):
    present_skills = resume_skills.intersection(jd_skills)
    missing_skills = jd_skills.difference(resume_skills)
    
    # Handle case where JD has no matching predefined skills
    if not jd_skills:
        strengths = list(resume_skills) if resume_skills else ["General Qualifications"]
    else:
        strengths = list(present_skills) if present_skills else ["No matching technical skills identified"]
        
    missing = list(missing_skills) if missing_skills else ["None"]
    
    # Generate Recommendation based on semantic similarity score
    if match_score >= 75:
        recommendation = "Strong candidate for technical interview."
    elif match_score >= 50:
        recommendation = "Potential candidate. Consider for preliminary screening."
    else:
        recommendation = "Does not meet core requirements. Review manually if needed."
        
    return {
        "strengths": strengths,
        "missing_skills": missing,
        "recommendation": recommendation
    }
