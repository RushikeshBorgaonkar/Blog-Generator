from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class BlogGenerator(BaseModel):
    title: str
    sections: list  

def get_llm():
    try:      
        return ChatGroq(
            model_name="llama3-8b-8192",        
        )
    except Exception as e:
        print(f"Exception occurred while initializing LLM: {e}")
        return None

def outline_creater(topic): 
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI bot that generates a blog outline based on the given topic."),
        ("human", "Create a blog outline for the topic: {topic}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"topic": topic})
    return response.content
   

def content_creater(outline):
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in creating detailed blog content based on an outline."),
        ("human", "Create content for the outline: {outline}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"outline": outline})
    return response.content
   

def formatter(content):
    llm = get_llm()
    
    output_parser = JsonOutputParser(pydantic_object=BlogGenerator)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI bot specializing in formatting blog content."),
        ("human", """Please return the content in valid JSON format strictly following these instructions:
        - Content: {content}
        - Format Instructions: {format_instructions}""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
            "content": content, 
            "format_instructions": output_parser.get_format_instructions()
        })
    return output_parser.parse(response.content)
  

def chatbot():
    print("Welcome to the Blog Generator Chatbot!")
    while True:
        user_input = input("You Type something to generate a blog on : ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        print("\nGenerating blog outline...")
        outline = outline_creater(user_input)
        print(f"Outline Generated : {outline}")
        print("=============================================================")
        print("=============================================================")
        print("\nGenerating blog content...")
        content = content_creater(outline)
        print(f"Content Generated: {content}")
        print("=============================================================")
        print("=============================================================")
        print("\nFormatting the blog content...")
        formatted_blog = formatter(content)
        print("\nHere is your generated blog in JSON format:\n")
        print(formatted_blog)
        print("=============================================================")
        print("=============================================================")
        print("\nAsk for another blog topic, or type 'exit' to quit.\n")

if __name__ == "__main__":
    chatbot()