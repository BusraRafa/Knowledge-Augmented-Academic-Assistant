from typing import List, Dict
from openai import OpenAI
import openai
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
            university["univ_name"].lower() in query_lower or
            any(loc["name"].lower() in query_lower for loc in university.get("locations_list", []))
        )

        if uni_mentioned:
            database_found = True

            
            database_context += f"University: {university['univ_name']} ({university.get('univ_type', 'N/A')})\n"
            database_context += f"Tagline: {university.get('tagline', '')}\n"
            database_context += f"Founded: {university.get('year_founded', 'N/A')}, Total Campuses: {university.get('total_campuses', 'N/A')}\n"
            database_context += f"Total Faculty: {university.get('total_faculty', 'N/A')}, Total Students: {university.get('total_students', 'N/A')}, Total Programs: {university.get('total_programs', 'N/A')}\n"
            database_context += f"About: {university.get('about', '')}\n"
            database_context += f"Unique Features: {university.get('what_makes_us_different', '')}\n"
            database_context += f"Locations: {', '.join([loc['name'] + ' (' + loc['address'] + ')' for loc in university.get('locations_list', [])])}\n"
            database_context += f"Accreditations: {', '.join([acc['name'] + ' (valid until ' + acc['valid_until'] + ')' for acc in university.get('accreditations_list', [])])}\n"
            database_context += f"Rankings: {', '.join([r['title'] + ' ' + str(r['rank']) + ' (' + str(r['year']) + ')' for r in university.get('rankings_list', [])])}\n\n"

            
            for program in university.get("programs", []):
                database_context += f"Program: {program['title']} ({program['level']})\n"
                database_context += f"Status: {program.get('status', 'N/A')}, Next Start Date: {program.get('next_start_date', 'N/A')}\n"
                database_context += f"Duration: {program.get('duration', 'N/A')}\n"
                database_context += f"Language: {program.get('language', 'N/A')}, Modality: {program.get('modality', 'N/A')}, Credits: {program.get('credits', 'N/A')}\n"
                database_context += f"Program Description: {program.get('description', '')}\n"
                database_context += f"Curriculum Overview: {program.get('curriculum_overview', '')}\n"
                database_context += f"Program Image (Base64): {program.get('image_base64', 'N/A')}\n"

                
                faculties = program.get("faculties", [])
                if faculties:
                    database_context += "Faculty Members:\n"
                    for f in faculties:
                        database_context += f"- {f['name']} ({f['department']}), Expertise: {f['expertise']}\n"

                
                if any(word in query_lower for word in ["curriculum", "courses", "subjects", "learn", "syllabus"]):
                    courses = program.get("courses", {})
                    if courses:
                        database_context += "Curriculum by Year:\n"
                        for year, course_list in courses.items():
                            database_context += f"- {year.replace('_', ' ').title()}: {', '.join(course_list)}\n"
                        database_context += "\n"

                
                if any(word in query_lower for word in ["admission", "requirement", "apply", "how to"]):
                    reqs = program.get("requirements", [])
                    steps = program.get("admission_steps", [])
                    database_context += "Admission Requirements:\n"
                    for r in reqs:
                        database_context += f"- {r}\n"
                    if steps:
                        database_context += "Admission Steps:\n"
                        for step in sorted(steps, key=lambda x: x.get("order", 0)):
                            database_context += f"{step['order']}. {step['step_title']}: {step['step_description']}\n"
                    deadlines = program.get("deadlines", [])
                    if deadlines:
                        database_context += "Application Deadlines:\n"
                        for d in deadlines:
                            database_context += f"- {d['batch_name']}: {d['start_date']} to {d['end_date']}\n"
                    database_context += "\n"

                
                if any(word in query_lower for word in ["fee", "tuition", "cost", "price", "scholarship"]):
                    tuition = program.get("tuition", {})
                    database_context += f"Tuition Fees: Domestic {tuition.get('domestic_tuition', 'N/A')} {tuition.get('currency', '')}, International {tuition.get('international_tuition', 'N/A')} {tuition.get('currency', '')}\n"
                    additional_expenses = program.get("additional_expenses", [])
                    if additional_expenses:
                        database_context += "Additional Expenses:\n"
                        for expense in additional_expenses:
                            database_context += f"- {expense['expense_name']}: {expense['cost_estimate']}\n"
                    scholarships = program.get("scholarships", [])
                    if scholarships:
                        database_context += "Scholarships:\n"
                        for sch in scholarships:
                            database_context += f"- {sch['name']}: {sch['amount']} (Eligibility: {sch['eligibility']})\n"
                    financial_aid = program.get("financial_aid", {})
                    if financial_aid:
                        database_context += f"Financial Aid: {financial_aid.get('description', '')}, Contact: {financial_aid.get('email', '')}, {financial_aid.get('phone', '')}\n\n"

                
                if any(word in query_lower for word in ["career", "job", "prospect", "work"]):
                    outcomes = program.get("learning_outcomes", [])
                    if outcomes:
                        database_context += "Learning Outcomes / Career Prospects:\n"
                        for out in outcomes:
                            database_context += f"- {out.get('outcome_text', '')}\n"
                        database_context += "\n"

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
"""

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
- Provide a thorough, well-structured response
- Break down information into clear sections with headers/labels
- Include all relevant details from the information provided
- Add helpful context and explanations
- Suggest next steps or offer additional assistance
- Never mention 'the information provided', 'according to data', or reference any source
- Answer naturally as if this is your own expert knowledge"""
    elif is_admission_query and not database_found:
        user_message = f"""{input_text}

IMPORTANT: The user is asking about admissions but hasn't specified which university (or the university isn't in your knowledge base).

Provide a COMPREHENSIVE response that includes:
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
        )

        result = response.choices[0].message.content.strip()

        return {
            "response": result,
        }
    except openai.AuthenticationError:
        
        return {
            "response": " Invalid API key. Please check your API key and try again."
        }
    except Exception as e:
        return {

            #"response": f"We are facing an error: {str(e)}"
            "response": "I'm sorry, I'm unable to process your request at the moment. Please try again later."
            #"database_used": False
        }



if __name__ == "__main__":
    
    university_data = {
        "universities": [
            {
                "id": 91,
                "univ_name": "Dhaka New University",
                "tagline": "The Future starts Here",
                "univ_type": "public",

                "year_founded": 2005,
                "total_campuses": 3,

                "about": "A leading university in research and innovation.",
                "what_makes_us_different": "Dhaka Global University stands out with its world-class research facilities, cutting-edge innovation hubs, and dedicated faculty who provide personalized mentorship.",
                
                "total_faculty": 65,
                "total_students": "15,000+",
                "total_programs": 100,
                
                "locations_list": [
                    {
                        "name": "Main Campus", 
                        "address": "Dhanmondi, Dhaka"
                    },
                    {
                        "name": "Branch Office", 
                        "address": "Banani, Dhaka"
                    }
                ],
                
                "accreditations_list": [
                    {
                        "name": "UGC Approved", 
                        "valid_until": "2030-12-31"
                        }
                ],
                "rankings_list": [
                    {
                        "title": "QS World Ranking", 
                        "rank": "#801", 
                        "year": 2025
                    }
                ],

                "programs": [
                    {
                        "id": 1,
                        "title": "Bachelor of Computer Science",
                        "level": "Bachelor",
                        "duration": "4 Years",
                        "language": "English",
                        "modality": "On-Campus",
                        "credits": 120,
                        "next_start_date": "2026-09-01",
                        "status": "Published",
                        
                        "description": "A comprehensive program in data science.",
                        "image_base64": "BASE64_ENCODED_IMAGE_HERE",
                        
                        "curriculum_overview": "Overview of 4 years curriculum",
                        
                        "courses": {
                            "first_year": ["Python", "Statistics"],
                            "second_year": ["Machine Learning", "Databases"],
                            "third_year": ["Big Data", "AI"],
                            "fourth_year": ["Thesis", "Internship"]
                        },

                        "requirements": [
                            "High school diploma", 
                            "GPA 3.5"
                        ],

                        "learning_outcomes": [
                            {"outcome_text": "Expertise in Python"},
                            {"outcome_text": "Data Visualization skills"}
                        ],
                        "faculties": [
                            {
                                "name": "Dr. Alan Turing", 
                                "department": "Computer Science", 
                                "expertise": "Theory of Computation"
                            }
                        ],
                        "deadlines": [
                         {
                               "batch_name": "Fall 2026", 
                               "start_date": "2026-01-01", 
                               "end_date": "2026-05-01"
                            }
                        ],

                        "admission_steps": [
                            {
                                "step_title": "Application", 
                                "step_description": "Submit online form", 
                                "order": 1
                            }
                        ],
                        "tuition": {
                            "domestic_tuition": 5000.0, 
                            "international_tuition": 12000.0, 
                            "currency": "USD"
                            },
                        "additional_expenses": [
                            {
                                "expense_name": "Housing", 
                                "cost_estimate": "$2000/year"
                            }
                        ],
                        "scholarships": [
                            {
                                "name": "Merit Scholarship", 
                                "amount": "$1000", "eligibility": 
                                "GPA > 3.8"
                            }
                        ],
                        "financial_aid": {
                            "description": "Needs based aid", 
                            "email": "aid@univ.edu", 
                            "phone": "12345678"
                        }
                    }
                ]
            },
            {
                "id": 92,
                "univ_name": "Blue Valley University",
                "tagline": "Empowering the Future",
                "univ_type": "private",

                "year_founded": 2010,
                "total_campuses": 2,

                "about": "Blue Valley University is dedicated to fostering creativity and innovation through a multidisciplinary approach.",
                "what_makes_us_different": "With a focus on interdisciplinary studies, Blue Valley University provides students with cutting-edge research facilities, a global network of alumni, and a strong emphasis on entrepreneurial spirit.",

                "total_faculty": 80,
                "total_students": "12,000+",
                "total_programs": 120,

                "locations_list": [
                    {
                    "name": "Main Campus",
                    "address": "Gulshan, Dhaka"
                    },
                    {
                    "name": "Research Hub",
                    "address": "Banani, Dhaka"
                    }
                ],

                "accreditations_list": [
                    {
                    "name": "UGC Approved",
                    "valid_until": "2035-12-31"
                    }
                ],

                "rankings_list": [
                    {
                    "title": "QS World Ranking",
                    "rank": "#701",
                    "year": 2025
                    }
                ],

                "programs": [
                    { 
                    "id": 2,
                    "title": "Bachelor of Business Administration",
                    "level": "Bachelor",
                    "duration": "3 Years",
                    "language": "English",
                    "modality": "On-Campus",
                    "credits": 90,
                    "next_start_date": "2026-08-01",
                    "status": "Published",

                    "description": "A comprehensive program designed to prepare students for leadership in the business world.",
                    "image_base64": "BASE64_ENCODED_IMAGE_HERE",

                    "curriculum_overview": "3 years of business curriculum with a focus on management, marketing, and entrepreneurship.",

                    "courses": {
                        "first_year": ["Business Communication", "Microeconomics"],
                        "second_year": ["Marketing Principles", "Financial Accounting"],
                        "third_year": ["Strategic Management", "Entrepreneurship"]
                    },

                    "requirements": [
                        "High school diploma",
                        "GPA 3.2"
                    ],

                    "learning_outcomes": [
                        { "outcome_text": "Leadership skills in business" },
                        { "outcome_text": "Marketing and finance knowledge" }
                    ],

                    "faculties": [
                        {
                        "name": "Prof. John Smith",
                        "department": "Business Administration",
                        "expertise": "Organizational Behavior"
                        }
                    ],

                    "deadlines": [
                        {
                        "batch_name": "Fall 2026",
                        "start_date": "2026-04-01",
                        "end_date": "2026-07-01"
                        }
                    ],

                    "admission_steps": [
                        {
                        "step_title": "Online Application",
                        "step_description": "Complete the online form and upload documents.",
                        "order": 1
                        }
                    ],

                    "tuition": {
                        "domestic_tuition": 4500.0,
                        "international_tuition": 10500.0,
                        "currency": "USD"
                    },

                    "additional_expenses": [
                        {
                        "expense_name": "Books & Supplies",
                        "cost_estimate": "$1000/year"
                        }
                    ],

                    "scholarships": [
                        {
                        "name": "Blue Valley Excellence Scholarship",
                        "amount": "$1500",
                        "eligibility": "Top 10% of applicants"
                        }
                    ],

                    "financial_aid": {
                        "description": "Merit and need-based scholarships available",
                        "email": "financialaid@bluevalley.edu",
                        "phone": "98765432"
                        }
                    }
                ]
            }           
        ]
    }
    

    chat_history = [
        {
            "user": "Hello, how are you?", 
            "assistant": "I'm doing well! How can I help you with your university search today?"
        }

    ]

    input_text = "What tution fees for computer science at Dhaka New University? what scholarships are available?"


    result = generate_response(input_text, chat_history, university_data)

    json_output = json.dumps(result, indent=2, ensure_ascii=False)
    print(json_output)
