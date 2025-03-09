import streamlit as st
import os
import nest_asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, Document
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from dotenv import load_dotenv

# Load environment variables
os.environ["MISTRAL_API_KEY"] = "WxuATixGO6kp5LQ2ilW1jLRiD5IFibV8"
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Apply nest_asyncio (for running async tasks in notebooks if needed)
nest_asyncio.apply()

# Streamlit UI
st.title("Agentic RAG with Mistral AI")

# Define policy URLs and content (for indexing purposes)
policy_texts = {
    "Student Conduct Policy": "This policy governs student behavior and disciplinary actions...",
    "Academic Schedule Policy": "This policy outlines the academic calendar and scheduling rules...",
    "Student Attendance Policy": "This policy explains attendance requirements and consequences of absenteeism.Definitions 1.1 The following words and expressions have the meaning hereby assigned to them: 1.1.1 Academic Members : Faculty Members and Instructional Members. 1.1.2 Academic Unit : An entity within the University that delivers Courses for Programs or for entry into Programs. 1.1.3 Attendance : The Student’s physical presence in a Learning Session. 1.1.4 Course : A set of Learning Sessions in a particular subject, with a defined scope and duration, and specific learning outcomes. 1.1.5 Course Requirements : Assessments and/or deliverables that a Student is required to complete, such as assignments, papers, reports and other coursework. 1.1.6 Credit : A unit of measurement assigned to a Course based on the total amount of learning time that counts toward a Program or credential completion, at a particular level of the Qatar National Qualification Framework. 1.1.7 Faculty Members : Members of the teaching and/or research staff, whether on part- or full-time, holding the following ranks: Professor, Associate Professor, Assistant Professor, Senior Lecturer/Senior Technical Instructor, or Lecturer/Technical Instructor. 1.1.8 Instructional Members: Members of the teaching staff, whether on part- or full- time, holding the following titles: Assistant Lecturer/or Workshop/Lab/Clinical Instructor, Assistant Technical Instructor, Trades Technical Instructor, or Teaching Assistant. 1.1.9 Learning Session Classes, labs, placements or work term prescribed by a Course. 1.1.10 Placement : A period spent by Student(s) for the purposes of clinical work or work term. 1.1.11 Program: A prescribed set of Courses leading to a qualification, including a Certificate, Diploma (2 years), Advanced Diploma (3 years), Bachelor, Master, or Doctorate, according to the Qatar National Qualifications Framework. 1.1.12 Punctuality: Arriving on time for scheduled Learning Session. 1.1.13 Student: A person who is presently enrolled at the University in a Credit course or who is designated by the University as a Student. 1.1.14 University: University of Doha for Science and Technology established by Emiri Resolution No. 13 of 2022. 1.2 Where the context requires, words importing the singular shall include the plural and vice-versa. 1.3 Where a word or phrase is given a particular meaning, other parts of speech and grammatical forms of that word or phrase have corresponding meanings. 2.0 Policy Purpose 2.1 The purpose of this policy is to establish Student Course Attendance standards and Student responsibilities. 3.0 Policy Scope 3.1 The policy applies to all Students and Academic Members. 4.0 Policy Statement 4.1 General 4.1.1 The University recognizes that regular Attendance and participation in Class is fundamental to Student success. 4.1.2 Attendance records start on the first day of the Course, and end on the last Learning Session of that Course. 4.2 Attendance Standard 4.2.1 The maximum allowable limit for absenteeism is 15% of Learning Sessions per Course during a semester. 4.2.2 Students exceeding the allowable limit for absenteeism in a Course will receive a failing grade for that Course. 4.2.3 The Academic Units may identify specific Learning Sessions, absenteeism from which will lead to a failing grade in the corresponding Course. 4.2.3.1 Punctuality standards will be set by each Academic Unit. A Student exceeding the set standards will be marked as absent. 4.3 Academic Members’ Responsibilities 4.3.1 With regard to Student Attendance, Academic Members are responsible for: 4.3.1.1 Informing Students of the importance of Attendance at Learning Sessions. 4.3.1.2 Recording Attendance in their Learning Sessions. 4.4 Student Responsibilities 4.4.1 Students are responsible for the regular, Punctual Attendance of all Learning Sessions, and prescribed activities for the Courses in which they are enrolled. 4.5 Admissions and Registration Department Responsibilities 4.5.1 The Admissions and Registration Department is the custodian of all Student Attendance records. 4.6 Course Requirements 4.6.1 Absence from a Learning Session does not relieve Students from completing any missed Course Requirements. 4.6.2 The Academic Member may grant Students an extension for completion of Course Requirements, if substantiating evidence is provided. 4.7 Placements (Clinical and Work Terms) 4.7.1 Students, who are enrolled in a Placement as part of their Program, are responsible for ensuring that they adhere to the attendance standards set by the Placement provider. 4.7.2 Students exceeding the allowable limit for absenteeism in a Course set by the Placement provider, where it is lower than the standard set by this Policy, will receive a failing grade for that Course",
    "Student Appeals Policy": "This policy details the process for students to appeal decisions...",
    "Graduation Policy": "This policy describes graduation requirements and processes...",
    "Academic Standing Policy": "This policy defines academic performance standards...",
    "Transfer Policy": "This policy outlines the transfer of credits and student mobility...",
    "Admissions Policy": "This policy sets the criteria and procedures for student admissions...",
    "Final Grade Policy": "This policy explains the grading system and final grade calculations...",
    "Registration Policy": "This policy provides guidelines for course registration and enrollment...",
}

policy_urls = {
    name: f"https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/{name.lower().replace(' ', '-')}"
    for name in policy_texts.keys()
}

# Set LLM globally
Settings.llm = MistralAI(api_key=MISTRAL_API_KEY)
Settings.embed_model = MistralAIEmbedding(api_key=MISTRAL_API_KEY)

# Create documents for indexing
documents = [Document(text=policy_texts[name], metadata={"name": name}) for name in policy_texts]
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Initialize session state for relevant policies
if "relevant_policies" not in st.session_state:
    st.session_state.relevant_policies = []
    st.session_state.relevant_query_engine = None

st.write("Enter your queries (first question: get relevant policies, subsequent questions: ask about them):")
user_input = st.text_input("Enter your prompt:")

if user_input:
    inputs = user_input.split("\n")
    responses = []
    
    for i, query in enumerate(inputs):
        query = query.strip()
        if not query:
            continue
        
        if i == 0:
            # Step 1: Find relevant policies
            policy_query_lower = query.lower()
            relevant_policies = [name for name in policy_texts.keys() if policy_query_lower in name.lower()]
            st.session_state.relevant_policies = relevant_policies
            
            if relevant_policies:
                policy_list = "\n".join([f"- {policy_name}: {policy_urls[policy_name]}" for policy_name in relevant_policies])
                responses.append(f"**Relevant Policies:**\n{policy_list}")
                
                # Create a new query engine only with relevant documents
                relevant_documents = [Document(text=policy_texts[name], metadata={"name": name}) for name in relevant_policies]
                st.session_state.relevant_query_engine = VectorStoreIndex.from_documents(relevant_documents).as_query_engine()
            else:
                responses.append("No matching policies found.")
        else:
            # Step 2: Answer specific policy-related question using previously retrieved policies
            if st.session_state.relevant_policies and st.session_state.relevant_query_engine:
                response = st.session_state.relevant_query_engine.query(query)
                responses.append(f"**Response:** {response.response}")
            else:
                responses.append("No relevant policies found to answer this question.")
    
    for response in responses:
        st.write(response)
