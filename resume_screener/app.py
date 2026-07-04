import streamlit as st
import pandas as pd
import numpy as np
from utils import (
    extract_text_from_pdf, 
    preprocess_text, 
    generate_embedding, 
    extract_skills, 
    generate_ai_summary
)
from models.vector_db import HybridVectorDB

def main():
    st.set_page_config(page_title="AI Resume Screener", layout="wide")
    
    st.title("AI Resume Screener + Resume Ranking System")
    st.write("Upload a Job Description and multiple Resumes to rank candidates based on semantic similarity and skill matching.")
    
    st.header("1. Upload Job Description")
    st.write("You can paste the Job Description text, upload a PDF, or both.")
    
    jd_text_pasted = st.text_area("Paste Job Description here:", height=150)
    jd_file = st.file_uploader("Upload Job Description PDF", type=["pdf"], key="jd_uploader")
    
    jd_text = jd_text_pasted
    if jd_file is not None:
        with st.spinner("Extracting text from JD PDF..."):
            jd_text += "\n" + extract_text_from_pdf(jd_file)
            st.success("Job Description PDF Extracted Successfully!")
        
    st.header("2. Upload Resumes")
    resume_files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True, key="resume_uploader")
    
    if st.button("Analyze Candidates"):
        if not jd_text.strip():
            st.error("Please provide a Job Description.")
        elif not resume_files:
            st.error("Please upload at least one Resume.")
        else:
            with st.spinner("Processing Job Description..."):
                jd_clean = preprocess_text(jd_text)
                jd_embedding = generate_embedding(jd_clean)
                jd_skills = extract_skills(jd_text)
                
            # Initialize FAISS Vector Database
            vector_db = HybridVectorDB(embedding_dim=len(jd_embedding))
            
            progress_bar = st.progress(0)
            
            for i, resume_file in enumerate(resume_files):
                progress_bar.progress((i) / len(resume_files), text=f"Processing {resume_file.name}...")
                
                res_text = extract_text_from_pdf(resume_file)
                
                if not res_text.strip():
                    st.warning(f"Could not extract text from {resume_file.name}. Skipping.")
                    continue
                    
                res_clean = preprocess_text(res_text)
                res_embedding = generate_embedding(res_clean)
                res_skills = extract_skills(res_text)
                
                # Add to Vector Database
                vector_db.add_resume(
                    candidate_name=resume_file.name.replace('.pdf', ''),
                    embedding=res_embedding,
                    skills=res_skills
                )
                
            progress_bar.progress(1.0, text="Processing Complete!")
            
            # Perform FAISS Hybrid Search
            search_results = vector_db.hybrid_search(jd_embedding, jd_skills, k=len(resume_files))
            
            results = []
            for res in search_results:
                candidate_data = vector_db.candidates[res["candidate_idx"]]
                match_score = res["hybrid_score"] * 100
                
                summary = generate_ai_summary(candidate_data["skills"], jd_skills, match_score)
                
                results.append({
                    "Candidate Name": res["name"],
                    "Match Score": match_score,
                    "Strengths": summary["strengths"],
                    "Missing Skills": summary["missing_skills"],
                    "Recommendation": summary["recommendation"]
                })
            
            if results:
                st.header("Analysis Results")
                
                # Data Processing using pandas and NumPy
                df_results = pd.DataFrame(results)
                df_results = df_results.sort_values(by="Match Score", ascending=False)
                
                for _, row in df_results.iterrows():
                    match_percentage = int(np.round(row['Match Score']))
                    
                    st.markdown(f"Candidate Name: {row['Candidate Name']}")
                    st.markdown(f"Match Score: {match_percentage}%")
                    st.markdown("")
                    
                    st.markdown("Strengths")
                    for skill in row["Strengths"]:
                        st.markdown(f"* {skill.title()}")
                        
                    st.markdown("")
                    st.markdown("Missing Skills")
                    for skill in row["Missing Skills"]:
                        st.markdown(f"* {skill.title()}")
                        
                    st.markdown("")
                    st.markdown("Recommendation")
                    st.markdown(row["Recommendation"])
                    st.markdown("---")
            
            st.markdown("<br><hr><center>Built by <b>Shadab Jamadar</b></center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
