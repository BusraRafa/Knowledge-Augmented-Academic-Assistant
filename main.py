from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)


def generate_response(input_text: str, chat_history: List[Dict[str, str]], university_database: Dict) -> Dict:
    
    """
    Generate chatbot response using RAG approach.
    Uses database if relevant data found, otherwise uses GPT's knowledge naturally.
    """
    if chat_history is None:
        chat_history = []
    
    query_lower = input_text.lower()
    

    database_context = ""
    database_found = False
    
    for university in university_database.get("universities", []):
        
        uni_mentioned = (
            university["university_name"].lower() in query_lower or
            university["city"].lower() in query_lower
        )
        
        if uni_mentioned:
            database_found = True
            
            for program in university.get("programs", []):
                database_context += f"University: {university['university_name']}\n"
                database_context += f"Location: {university['city']}, {university['country']}\n"
                database_context += f"Program: {program['program_name']} ({program['degree_level']})\n"
                database_context += f"Duration: {program['duration']}\n"
                database_context += f"Language: {program['language']}\n"
                database_context += f"Description: {program['program_overview']['description']}\n\n"
                
                
                if any(word in query_lower for word in ["admission", "requirement", "apply", "how to"]):
                    req = program.get("admission_requirements", {})
                    database_context += "Admission Requirements:\n"
                    database_context += f"- GPA: {req.get('gpa_requirement', 'N/A')}\n"
                    database_context += f"- Academic: {', '.join(req.get('academic', []))}\n"
                    database_context += f"- Test Scores: {', '.join(req.get('test_scores', []))}\n"
                    database_context += f"- Documents: {', '.join(req.get('documents', []))}\n\n"
                    
                    deadlines = program.get("application_deadlines", [])
                    if deadlines:
                        database_context += "Application Deadlines:\n"
                        for d in deadlines:
                            database_context += f"- {d['intake']}: {d['deadline']}\n"
                        database_context += "\n"
                
                if any(word in query_lower for word in ["fee", "tuition", "cost", "price", "scholarship"]):
                    fees = program.get("tuition_fees", {})
                    database_context += f"Tuition Fees: {fees.get('per_year', 'N/A')} per year\n"
                    database_context += f"Application Fee: {program.get('application_fee', 'N/A')}\n"
                    database_context += f"Scholarships: {'Available' if fees.get('scholarships_available') else 'Not mentioned'}\n\n"
                
                if any(word in query_lower for word in ["curriculum", "courses", "subjects", "learn", "syllabus"]):
                    curriculum = program.get("curriculum", {})
                    if curriculum:
                        database_context += "Curriculum:\n"
                        for year, courses in list(curriculum.items())[:2]:
                            database_context += f"- {year.replace('_', ' ').title()}: {', '.join(courses[:4])}\n"
                        database_context += "\n"
                
                if any(word in query_lower for word in ["career", "job", "prospect", "work"]):
                    careers = program.get("career_prospects", [])
                    if careers:
                        database_context += f"Career Prospects: {', '.join(careers)}\n\n"
                
                database_context += "---\n\n"
    

    is_admission_query = any(word in query_lower for word in [
        "admission", "apply", "application", "requirement", "how to apply",
        "admission process", "get admitted", "entry requirement"
    ])

    system_prompt = """You are a highly knowledgeable and professional university admissions assistant chatbot.
You help students find universities and programs that match their needs with comprehensive, detailed guidance.

Your role is to:
- Provide thorough, well-structured, and detailed answers about universities, programs, admissions, fees, and careers
- Break down complex information into clear sections with proper formatting
- Anticipate follow-up questions and address them proactively
- Guide students through their decision-making process with actionable advice

RESPONSE STRUCTURE AND DETAIL REQUIREMENTS:

1. **BE COMPREHENSIVE**: Don't give minimal answers. Provide detailed, multi-faceted responses that cover:
   - All relevant aspects of the question
   - Important context and background information
   - Step-by-step processes when applicable
   - Additional helpful information the student might need

2. **USE CLEAR STRUCTURE**: Organize responses with:
   - Brief introductory statement
   - Main content broken into logical sections
   - Clear headers/labels for different topics (e.g., "Academic Requirements:", "Required Documents:")
   - Numbered lists for sequential steps
   - Bullet points for multiple items
   - Summary or recommendation section when appropriate

3. **PROVIDE DEPTH**: For each topic, elaborate with specifics:
   - Don't just list requirements - explain what they mean
   - Include relevant details like deadlines, processes, tips
   - Mention important considerations or common questions
   - Add context that helps students understand the full picture

4. **BE ACTIONABLE**: Help students know what to do next:
   - Suggest concrete next steps
   - Offer to help with related questions
   - Point out important things to consider or prepare

CRITICAL INSTRUCTIONS:
- When information is provided about universities/programs, treat it as YOUR knowledge, not external data
- NEVER say phrases like "based on the information provided", "according to the database", "the information shows", or "even though the information focuses on"
- Answer naturally as if you inherently know this information
- Don't mention sources, data, or where information comes from
- Be conversational yet professional - like an experienced advisor
- If specific details are missing, acknowledge it naturally and offer to help find more information

WHEN USER ASKS ABOUT ADMISSIONS WITHOUT SPECIFYING A UNIVERSITY:
- Provide a comprehensive response covering general admission guidance
- Explain the typical admission process and timeline
- List common requirements across universities
- Then ask which specific university or program they're interested in for tailored details
- Offer to provide detailed admission requirements, deadlines, and application processes
- Make them feel supported with actionable next steps

# WHEN USER ASKS OFF-TOPIC QUESTIONS (travel, weather, general knowledge, etc.):
# - DO NOT answer the off-topic question
# - FIRST, briefly acknowledge what they're asking about (e.g., "It looks like you're asking about...", "I see you're wondering about...")
# - THEN redirect them by explaining you're specifically designed to help with university admissions and applications
# - Ask if they need any help with university programs, admissions, or applications
# - Keep it friendly, brief, and focused on redirecting to your purpose
# - The acknowledgment shows you understood them before redirecting

RESPONSE LENGTH:
- Aim for 150-300 words for most responses
- Go longer (300-500 words) for complex topics like admission processes or program comparisons
- Keep only greeting responses brief (50-100 words)

Remember: Students benefit from detailed, well-organized information. Don't be brief when thoroughness adds value."""
    

    messages = [{"role": "system", "content": system_prompt}]
    
    
    for chat in chat_history:
        if chat.get("user"):
            messages.append({"role": "user", "content": chat["user"]})
        if chat.get("assistant"):
            messages.append({"role": "assistant", "content": chat["assistant"]})
    
    
    if database_found:
        user_message = f"""Here is information about the universities and programs:

{database_context}

Now answer this question with comprehensive detail: {input_text}

IMPORTANT INSTRUCTIONS:
- Provide a thorough, well-structured response (150-300 words minimum)
- Break down information into clear sections with headers/labels
- Include all relevant details from the information provided
- Add helpful context and explanations
- Suggest next steps or offer additional assistance
- Never mention 'the information provided', 'according to data', or reference any source
- Answer naturally as if this is your own expert knowledge"""
    elif is_admission_query and not database_found:
        user_message = f"""{input_text}

IMPORTANT: The user is asking about admissions but hasn't specified which university (or the university isn't in your knowledge base).

Provide a COMPREHENSIVE response (200-300 words) that includes:
1. A warm greeting acknowledging their interest
2. General overview of admission requirements for this type of program (academic background, typical GPA ranges, common test requirements, standard documents)
3. Explanation of typical application timeline and process
4. Then politely ask which specific university they're interested in
5. Offer specific ways you can help (detailed requirements, deadlines, application strategies)
6. End with encouragement and next steps

Be detailed, structured, and make them feel guided and supported."""
    else:
        user_message = input_text
    
    messages.append({"role": "user", "content": user_message})
    

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.7,
            #max_tokens=1000
        )
        
        result = response.choices[0].message.content.strip()
        
        return {
            "response": result,
            #"database_used": database_found
        }
    
    except Exception as e:
        return {
            "response": f"I apologize, but I encountered an error: {str(e)}",
            "database_used": False
        }


if __name__ == "__main__":
    
    university_data = {
        "universities": [
            {
                "university_id": "UNI_001",
                "university_name": "University of Barcelona",
                "country": "Spain",
                "city": "Barcelona",
                "website": "https://web.ub.edu/en/",
                "ranking": {"world_rank": 350, "subject_rank": 120},
                "programs": [
                    {
                        "program_id": "CS_BSC_001",
                        "program_name": "Bachelor of Computer Science",
                        "degree_level": "Bachelor",
                        "department": "Computer Science",
                        "language": "English",
                        "duration": "4 years",
                        "application_fee": "USD 45",
                        "last_updated": "2024-09-15",
                        "program_overview": {
                            "description": "This program prepares students for careers in software development, data science, artificial intelligence, and systems engineering through a strong foundation in computer science theory and practical application."
                        },
                        "learning_outcomes": [
                            "Understand core computer science concepts including algorithms, data structures, and operating systems",
                            "Apply programming skills to solve real-world problems",
                            "Design and analyze efficient software systems"
                        ],
                        "curriculum": {
                            "first_year": ["Introduction to Programming", "Discrete Mathematics", "Calculus I", "Computer Systems Basics"],
                            "second_year": ["Data Structures", "Algorithms", "Databases", "Operating Systems"],
                            "third_year": ["Software Engineering", "Computer Networks", "Artificial Intelligence", "Web Development"],
                            "fourth_year": ["Final Year Project", "Machine Learning", "Cyber Security", "Cloud Computing"]
                        },
                        "admission_requirements": {
                            "academic": ["High school diploma or equivalent", "Strong background in mathematics"],
                            "gpa_requirement": "Minimum GPA 3.0",
                            "test_scores": ["SAT/ACT (optional)", "IELTS/TOEFL for international students"],
                            "documents": ["Personal statement", "Letters of recommendation", "Academic transcripts"]
                        },
                        "application_deadlines": [
                            {"intake": "Fall 2025", "deadline": "January 1, 2025"},
                            {"intake": "Spring 2026", "deadline": "October 1, 2025"}
                        ],
                        "career_prospects": ["Software Engineer", "Data Analyst", "Machine Learning Engineer", "Systems Analyst"],
                        "tuition_fees": {"per_year": "USD 18,000", "scholarships_available": True}
                    }
                ]
            },
            {
                "university_id": "UNI_002",
                "university_name": "Autonomous University of Barcelona",
                "country": "Spain",
                "city": "Bellaterra",
                "website": "https://www.uab.cat/web/",
                "ranking": {"world_rank": 149, "subject_rank": 12},
                "programs": [
                    {
                        "program_id": "ENG_BSC_001",
                        "program_name": "Bachelor of Engineering",
                        "degree_level": "Bachelor",
                        "department": "Engineering",
                        "language": "English",
                        "duration": "4 years",
                        "application_fee": "EUR 50",
                        "program_overview": {
                            "description": "The program is designed to provide students with the knowledge and skills needed to become leaders in the field of engineering."
                        },
                        "curriculum": {
                            "first_year": ["Introduction to Engineering", "Calculus I", "Physics", "Materials Science"],
                            "second_year": ["Mechanical Engineering", "Circuit Theory", "Engineering Design", "Electronics"],
                            "third_year": ["Fluid Mechanics", "Thermodynamics", "Electrical Machines", "Control Systems"],
                            "fourth_year": ["Capstone Project", "Advanced Topics in Engineering"]
                        },
                        "admission_requirements": {
                            "academic": ["High school diploma or equivalent", "Strong background in mathematics"],
                            "gpa_requirement": "Minimum GPA 3.2",
                            "test_scores": ["SAT/ACT (optional)", "IELTS/TOEFL for international students"],
                            "documents": ["Personal statement", "Letters of recommendation", "Academic transcripts"]
                        },
                        "application_deadlines": [
                            {"intake": "Fall 2025", "deadline": "April 1, 2025"},
                            {"intake": "Spring 2026", "deadline": "November 1, 2025"}
                        ],
                        "career_prospects": ["Mechanical Engineer", "Electrical Engineer", "Civil Engineer", "Systems Engineer"],
                        "tuition_fees": {"per_year": "EUR 15,000", "scholarships_available": True}
                    }
                ]
            },
            {
                "university_id": "UNI_003",
                "university_name": "Pompeu Fabra University",
                "country": "Spain",
                "city": "Barcelona",
                "website": "https://www.upf.edu/en/",
                "ranking": {"world_rank": 265, "subject_rank": 33},
                "programs": [
                    {
                        "program_id": "ARTS_BA_001",
                        "program_name": "Bachelor of Arts",
                        "degree_level": "Bachelor",
                        "department": "Arts",
                        "language": "English",
                        "duration": "3 years",
                        "application_fee": "EUR 40",
                        "program_overview": {
                            "description": "This program offers an interdisciplinary approach to arts and humanities, focusing on critical thinking, communication, and artistic expression."
                        },
                        "curriculum": {
                            "first_year": ["Introduction to Arts", "History of Western Art", "Philosophy of Arts", "Creative Writing"],
                            "second_year": ["Art Theory", "Literature and Culture", "Contemporary Art", "Public Speaking"],
                            "third_year": ["Art Exhibition Project", "Cultural Studies", "Art Management"]
                        },
                        "admission_requirements": {
                            "academic": ["High school diploma or equivalent"],
                            "gpa_requirement": "Minimum GPA 3.0",
                            "test_scores": ["IELTS/TOEFL for international students"],
                            "documents": ["Personal statement", "Portfolio (for art-related applicants)", "Letters of recommendation"]
                        },
                        "application_deadlines": [
                            {"intake": "Fall 2025", "deadline": "June 1, 2025"},
                            {"intake": "Spring 2026", "deadline": "October 15, 2025"}
                        ],
                        "career_prospects": ["Art Curator", "Museum Educator", "Cultural Analyst", "Creative Director"],
                        "tuition_fees": {"per_year": "EUR 12,000", "scholarships_available": True}
                    }
                ]
            }
        ]
    }
    
    
    chat_history = [
        {
            "user": "Hello, how are you?",
            "assistant": "I'm doing well, thank you for asking! How can I help you today with your university search?"
        }
    ]
    
   
    input_text = "What are the admission requirements for computer science at Global Tech University?"
    
    
    result = generate_response(input_text, chat_history, university_data)
    
    json_output = json.dumps(result, indent=2, ensure_ascii=False)
    print(json_output)


# from typing import List, Dict
# from openai import OpenAI
# from dotenv import load_dotenv
# import os
# import json

# load_dotenv()

# openai_api_key = os.getenv("OPENAI_API_KEY")
# openai_client = OpenAI(api_key=openai_api_key)


# def generate_response(input_text: str, chat_history: List[Dict[str, str]], university_database: Dict) -> Dict:
    
#     """
#     Generate chatbot response using RAG approach.
#     Uses database if relevant data found, otherwise uses GPT's knowledge naturally.
#     """
#     if chat_history is None:
#         chat_history = []
    
#     query_lower = input_text.lower()
    

#     database_context = ""
#     database_found = False
    
#     for university in university_database.get("universities", []):
        
#         uni_mentioned = (
#             university.get("univ_name", "").lower() in query_lower or
#             any(loc["address"].lower() in query_lower for loc in university.get("locations_list", []))
#         )
        
#         if uni_mentioned:
#             database_found = True
            
#             for program in university.get("programs", []):
#                 database_context += f"University: {university['univ_name']}\n"
#                 database_context += f"Type: {university['univ_type']}\n"
#                 database_context += f"Founded: {university['year_founded']}\n"
#                 database_context += f"About: {university['about']}\n\n"

#                 database_context += f"Program: {program['title']} ({program['level']})\n"
#                 database_context += f"Duration: {program['duration']}\n"
#                 database_context += f"Language: {program['language']}\n"
#                 database_context += f"Description: {program['description']}\n\n"
                
#                 if any(word in query_lower for word in ["admission", "requirement", "apply", "how to"]):
#                     database_context += "Admission Requirements:\n"
#                     for r in program.get("requirements", []):
#                         database_context += f"- {r}\n"
#                     database_context += "\n"
                    
#                     deadlines = program.get("deadlines", [])
#                     if deadlines:
#                         database_context += "Application Deadlines:\n"
#                         for d in deadlines:
#                             database_context += f"- {d['batch_name']}: {d['start_date']} to {d['end_date']}\n"
#                         database_context += "\n"
                
#                 if any(word in query_lower for word in ["fee", "tuition", "cost", "price", "scholarship"]):
#                     tuition = program.get("tuition", {})
#                     database_context += f"Domestic Tuition: {tuition.get('domestic_tuition')} {tuition.get('currency')}\n"
#                     database_context += f"International Tuition: {tuition.get('international_tuition')} {tuition.get('currency')}\n"
                    
#                     scholarships = program.get("scholarships", [])
#                     if scholarships:
#                         database_context += "Scholarships:\n"
#                         for s in scholarships:
#                             database_context += f"- {s['name']} ({s['amount']}): {s['eligibility']}\n"
#                         database_context += "\n"
                
#                 if any(word in query_lower for word in ["curriculum", "courses", "subjects", "learn", "syllabus"]):
#                     courses = program.get("courses", {})
#                     database_context += "Curriculum Overview:\n"
#                     for year, course_list in courses.items():
#                         database_context += f"- {year.replace('_', ' ').title()}: {', '.join(course_list)}\n"
#                     database_context += "\n"
                
#                 database_context += "---\n\n"
    

#     is_admission_query = any(word in query_lower for word in [
#         "admission", "apply", "application", "requirement", "how to apply",
#         "admission process", "get admitted", "entry requirement"
#     ])

#     system_prompt = """You are a highly knowledgeable and professional university admissions assistant chatbot.
# You help students find universities and programs that match their needs with comprehensive, detailed guidance.

# Your role is to:
# - Provide thorough, well-structured, and detailed answers about universities, programs, admissions, fees, and careers that the student is asking about
# - Break down complex information into clear sections with proper formatting
# - Anticipate follow-up questions and address them proactively
# - Guide students through their decision-making process with actionable advice

# CRITICAL INSTRUCTIONS:
# - Treat all university information as your own knowledge
# - Never mention data sources or databases
# - Be professional, helpful, and supportive
# """

#     messages = [{"role": "system", "content": system_prompt}]
    
#     for chat in chat_history:
#         if chat.get("user"):
#             messages.append({"role": "user", "content": chat["user"]})
#         if chat.get("assistant"):
#             messages.append({"role": "assistant", "content": chat["assistant"]})
    
#     if database_found:
#         user_message = f"""Here is information about the universities and programs:

# {database_context}

# Now answer this question with comprehensive detail: {input_text}
# """
#     elif is_admission_query:
#         user_message = input_text
#     else:
#         user_message = input_text
    
#     messages.append({"role": "user", "content": user_message})
    
#     response = openai_client.chat.completions.create(
#         model="gpt-4-turbo",
#         messages=messages,
#         temperature=0.7,
#     )
    
#     return {"response": response.choices[0].message.content.strip()}


# if __name__ == "__main__":
    
#     university_data = {
#         "universities": [
#             {
#                 "id": 91,
#                 "univ_name": "Dhaka New University",
#                 "tagline": "The Future starts Here",
#                 "univ_type": "public",
#                 "year_founded": 2005,
#                 "total_campuses": 3,
#                 "about": "A leading university in research and innovation.",
#                 "what_makes_us_different": "Dhaka Global University stands out with its world-class research facilities, cutting-edge innovation hubs, and dedicated faculty who provide personalized mentorship.",
#                 "total_faculty": 65,
#                 "total_students": "15,000+",
#                 "total_programs": 100,
#                 "locations_list": [
#                     {"name": "Main Campus", "address": "Dhanmondi, Dhaka"},
#                     {"name": "Branch Office", "address": "Banani, Dhaka"}
#                 ],
#                 "accreditations_list": [
#                     {"name": "UGC Approved", "valid_until": "2030-12-31"}
#                 ],
#                 "rankings_list": [
#                     {"title": "QS World Ranking", "rank": "#801", "year": 2025}
#                 ],
#                 "media": {
#                     "logo_url": "https://example.com/media/logo.jpg",
#                     "banner_video_url": "https://example.com/media/banner.mp4",
#                     "section_video_url": "https://example.com/media/section.mp4"
#                 },
#                 "programs": [
#                     {
#                         "id": 1,
#                         "title": "Bachelor of Computer Science",
#                         "level": "Bachelor",
#                         "duration": "4 Years",
#                         "language": "English",
#                         "modality": "On-Campus",
#                         "credits": 120,
#                         "next_start_date": "2026-09-01",
#                         "status": "Published",
#                         "description": "A comprehensive program in data science.",
#                         "image_base64": "BASE64_ENCODED_IMAGE_HERE",
#                         "curriculum_overview": "Overview of 4 years curriculum",
#                         "courses": {
#                             "first_year": ["Python", "Statistics"],
#                             "second_year": ["Machine Learning", "Databases"],
#                             "third_year": ["Big Data", "AI"],
#                             "fourth_year": ["Thesis", "Internship"]
#                         },
#                         "requirements": [
#                             "High school diploma",
#                             "GPA 3.5"
#                         ],
#                         "learning_outcomes": [
#                             {"outcome_text": "Expertise in Python"},
#                             {"outcome_text": "Data Visualization skills"}
#                         ],
#                         "faculties": [
#                             {
#                                 "name": "Dr. Alan Turing",
#                                 "department": "Computer Science",
#                                 "expertise": "Theory of Computation"
#                             }
#                         ],
#                         "deadlines": [
#                             {
#                                 "batch_name": "Fall 2026",
#                                 "start_date": "2026-01-01",
#                                 "end_date": "2026-05-01"
#                             }
#                         ],
#                         "admission_steps": [
#                             {
#                                 "step_title": "Application",
#                                 "step_description": "Submit online form",
#                                 "order": 1
#                             }
#                         ],
#                         "tuition": {
#                             "domestic_tuition": 5000.0,
#                             "international_tuition": 12000.0,
#                             "currency": "USD"
#                         },
#                         "additional_expenses": [
#                             {
#                                 "expense_name": "Housing",
#                                 "cost_estimate": "$2000/year"
#                             }
#                         ],
#                         "scholarships": [
#                             {
#                                 "name": "Merit Scholarship",
#                                 "amount": "$1000",
#                                 "eligibility": "GPA > 3.8"
#                             }
#                         ],
#                         "financial_aid": {
#                             "description": "Needs based aid",
#                             "email": "aid@univ.edu",
#                             "phone": "12345678"
#                         }
#                     }
#                 ]
#             }
#         ]
#     }


#     chat_history = [
#         {
#             "user": "Hello, how are you?",
#             "assistant": "I'm doing well, thank you for asking! How can I help you today with your university search?"
#         }
#     ]
    
   
#     input_text = "What are the admission requirements for Dhaka New University? which programs can i apply for?"
    
    
#     result = generate_response(input_text, chat_history, university_data)
    
#     json_output = json.dumps(result, indent=2, ensure_ascii=False)
#     print(json_output)
