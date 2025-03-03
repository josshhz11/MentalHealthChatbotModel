import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import time
import sqlite3
import pandas as pd
import os
import sys

general_question_bank = [
    "Did you partake in physical activity today?",
    "Did you leave the house today?",
    "Did you eat your medication regularly?",
    "How much did you sleep last night?",
    "How would you describe your mood today?",
    "How did your interactions with your family or friends go today?",
    "What did you do that you enjoyed today? (like a hobby)"
]

model = OllamaLLM(model="llama3.2")
template_str = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer: 
"""

def get_refined_questions(question_bank):
    print("🔄 Starting question refinement process...")
    
    prompt = """You are an empathetic mental health professional who specializes in patient engagement and follow-up care. 
    
                Task: Rewrite each question to be more conversational, empathetic, and engaging for mental health patients who have been discharged from care.

                Guidelines:
                - Make questions warm and supportive
                - Use casual, friendly language
                - Avoid clinical or formal tone
                - Include gentle encouragement
                - Make questions open-ended to encourage dialogue
                - Maintain a positive, upbeat tone while being sensitive
                - Each response should be a single question
                - Keep the core meaning of each original question

                Input Questions:
                {questions}

                Format your response as a numbered list with ONLY the refined questions, one per line, starting with 1.
                Do not include any other text or explanations."""


    # Convert question bank to numbered string format
    print("📝 Formatting input questions...")
    questions_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(question_bank)])
    
    chat_prompt = ChatPromptTemplate.from_template(template=prompt)
    chain = chat_prompt | model
    
    # Get response from model
    response = chain.invoke({"questions": questions_str})

    # Get response from model
    print("🤖 Generating refined questions using LLM...")
    response = chain.invoke({"questions": questions_str})
    print("✨ Successfully received model response!")
    
    # Parse the response into a list of questions
    print("📋 Parsing refined questions...")
    refined_questions = []
    for line in response.split('\n'):
        # Skip empty lines and extract questions
        if line.strip() and any(char.isdigit() for char in line):
            # Remove numbering and whitespace
            question = line.split('.', 1)[1].strip()
            refined_questions.append(question)
    
    print(f"✅ Successfully refined {len(refined_questions)} questions!")
    return refined_questions

def generate_refined_questions():
    # Use the function
    refined_question_bank = get_refined_questions(general_question_bank)

    # Save to file
    print("\n💾 Saving refined questions to file...")
    with open('refined_questions1.txt', 'w') as f:
        for q in refined_question_bank:
            f.write(q + '\n')
    print("📁 Questions saved to 'refined_questions.txt'")

    # Print results
    print("\n==== Results ====")
    print("\nOriginal questions:")
    for q in general_question_bank:
        print(f"- {q}")

    print("\nRefined questions:")
    for q in refined_question_bank:
        print(f"- {q}")

def generate_responses(num_responses=50):
    print("🚀 Starting response generation process...")

    # Check if database exists
    db_exists = os.path.exists('mental_health_data.db')
    
    if not db_exists:
        print("📁 Database not found. Initializing new database...")
        if not initialize_db():
            return

    # Connect to database
    conn = sqlite3.connect('mental_health_data.db')
    cursor = conn.cursor()

    # Read refined questions from file
    print("📖 Reading refined questions from file...")
    with open('refined_questions.txt', 'r') as f:
        questions = [line.strip() for line in f.readlines() if line.strip()]

    # Insert questions
    print("📝 Inserting questions into database...")
    for question in questions:
        cursor.execute('INSERT INTO questions (question_text) VALUES (?)', (question,))

    # Check which questions are new
    existing_questions = pd.read_sql_query("SELECT question_text FROM questions", conn)
    new_questions = [q for q in questions if q not in existing_questions['question_text'].values]
    
    if new_questions:
        print(f"📝 Inserting {len(new_questions)} new questions into database...")
        for question in new_questions:
            cursor.execute('INSERT INTO questions (question_text) VALUES (?)', (question,))

    # Get all question IDs
    question_df = pd.read_sql_query("SELECT question_id, question_text FROM questions", conn)

    # Generate responses using LLM
    print(f"🤖 Generating {num_responses} sample responses per question using Llama3.2...")
    
    response_prompt = """You are a mental health patient who has been discharged from care. 
    Generate a realistic, varied response to this follow-up question. 
    The response should reflect different possible mental states and situations.
    Mix positive, neutral, and negative responses to create a diverse dataset.
    Keep responses between 1-3 sentences.

    Question: {question}

    Generate a single response:"""
    
    chat_prompt = ChatPromptTemplate.from_template(template=response_prompt)
    chain = chat_prompt | model
    
    # Generate num_responses (default 50) for each question
    for _, row in question_df.iterrows():
        question_id = row['question_id']
        question = row['question_text']
        print(f"\n💭 Generating responses for question {question_id}: {question}")

        for i in range(num_responses):
            if i % 10 == 0:
                print(f"Progress: {i}/{num_responses} responses")
            
            response = chain.invoke({"question": question})
            
            # Insert response into database
            cursor.execute('''
                INSERT INTO responses (question_id, response_text)
                VALUES (?, ?)
            ''', (question_id, response.strip()))
    
    # Commit changes
    conn.commit()

    # Update CSV files
    update_csv_files(conn)
    
    conn.close()
    
    print("\n✅ Process completed successfully!")
    print(f"📊 Generated {len(questions) * 50} total responses")
    print("💾 Data saved to:")
    print("   - mental_health_data.db (SQLite database)")
    print("   - questions.csv")
    print("   - responses.csv")

def update_csv_files(conn):
    """Update CSV files with current database content"""
    print("\n📊 Updating CSV files...")
    
    # Export questions
    questions_df = pd.read_sql_query("SELECT * FROM questions", conn)
    questions_df.to_csv('questions.csv', index=False)
    
    # Export responses
    responses_df = pd.read_sql_query("""
        SELECT r.response_id, q.question_text, r.response_text, r.sentiment_score
        FROM responses r 
        JOIN questions q ON r.question_id = q.question_id
    """, conn)
    responses_df.to_csv('responses.csv', index=False)
    print("✅ CSV files updated successfully!")

def initialize_db():
    try:
        # Initialize SQLite database
        print("🗄️ Setting up SQLite database...")
        conn = sqlite3.connect('mental_health_data.db')
        cursor = conn.cursor()

        # Drop existing tables if needed
        cursor.execute('DROP TABLE IF EXISTS responses')
        cursor.execute('DROP TABLE IF EXISTS questions')

        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            question_id INTEGER PRIMARY KEY,
            question_text TEXT NOT NULL
        )''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS responses (
            response_id INTEGER PRIMARY KEY,
            question_id INTEGER,
            response_text TEXT NOT NULL,
            sentiment_score INTEGER DEFAULT NULL,
            FOREIGN KEY (question_id) REFERENCES questions (question_id)
            )''')

        conn.commit()
        conn.close()
        print("✅ Database setup complete!")
        return True
    except sqlite3.Error as e:
        print("❌ Error initializing database:", e)
        return False
    
def label_sentiments():
    # Read the responses CSV
    df = pd.read_csv('responses.csv')
    
    # Create a copy for labeled data
    labeled_df = df.copy()
    labeled_df['sentiment_score'] = None
    
    print("🏷️ Starting sentiment labeling process...")
    print("Enter a score from 1-100 for each response:")
    print("Guidelines:")
    print("• 1-20:   Very negative/concerning responses")
    print("• 21-40:  Moderately negative responses")
    print("• 41-60:  Neutral responses")
    print("• 61-80:  Moderately positive responses")
    print("• 81-100: Very positive/encouraging responses")
    print("\nUse any number within these ranges for fine-grained scoring")
    print("Type 'q' to save and quit\n")
    
    try:
        for idx, row in df.iterrows():
            # Clear screen for Windows
            os.system('cls')
            
            print(f"Progress: {idx + 1}/{len(df)} responses")
            print(f"\nQuestion: {row['question_text']}")
            print(f"Response: {row['response_text']}")
            
            while True:
                score = input("\nEnter sentiment score (1-100) or 'q' to quit: ")
                if score.lower() == 'q':
                    raise KeyboardInterrupt
                try:
                    score = int(score)
                    if 1 <= score <= 100:
                        labeled_df.at[idx, 'sentiment_score'] = score
                        break
                    else:
                        print("Please enter a number between 1 and 100")
                except ValueError:
                    print("Please enter a valid number")
                    
            # Save progress every 10 responses
            if (idx + 1) % 10 == 0:
                labeled_df.to_csv('labeled_responses.csv', index=False)
                print("\nProgress saved! ✅")
                
    except KeyboardInterrupt:
        print("\n\nLabeling interrupted. Saving progress...")
    
    # Save final results
    labeled_df.to_csv('labeled_responses.csv', index=False)
    
    # Update database with new labels
    conn = sqlite3.connect('mental_health_data.db')
    for idx, row in labeled_df.iterrows():
        if pd.notna(row['sentiment_score']):
            conn.execute('''
                UPDATE responses 
                SET sentiment_score = ? 
                WHERE response_id = ?
            ''', (int(row['sentiment_score']), row['response_id']))
    conn.commit()
    conn.close()
    
    print("\n✅ Labeling process completed and saved!")

def query_db():
    try:
        conn = sqlite3.connect('mental_health_data.db')
        cursor = conn.cursor()

        while True:
            print("\n📊 Database Query Options:")
            print("1. View all questions")
            print("2. View responses for a specific question")
            print("3. View responses by sentiment range")
            print("4. View summary statistics")
            print("5. Custom SQL query")
            print("6. Delete DB entries")
            print("7. Exit")
            
            choice = input("\nEnter your choice (1-7): ")

            match(choice):
                case '1':
                    questions_df = pd.read_sql_query(
                        "SELECT question_id, question_text FROM questions", conn)
                    print("\nAll Questions:")
                    for _, row in questions_df.iterrows():
                        print(f"{row['question_id']}: {row['question_text']}")
                case '2':
                    question_id = input("\nEnter question ID: ")
                    responses_df = pd.read_sql_query("""
                        SELECT r.response_id, q.question_id, q.question_text, r.response_text, r.sentiment_score
                        FROM responses r
                        JOIN questions q ON r.question_id = q.question_id
                        WHERE q.question_id = ?
                        ORDER BY r.sentiment_score DESC
                    """, conn, params=[question_id])
                    
                    if responses_df.empty:
                        print("No responses found for this question ID")
                    else:
                        print("\nResponses:")
                        for _, row in responses_df.iterrows():
                            sentiment = row['sentiment_score'] if pd.notna(row['sentiment_score']) else 'Not labeled'
                            print(f"Question ID: {row['question_id']}, Response ID: {row['response_id']}, Score: {sentiment}")
                            print(f"Question: {row['question_text']}")
                            print(f"Response: {row['response_text']}")
                            print("-" * 50)

                case '3':
                    min_score = input("Enter minimum sentiment score (1-100): ")
                    max_score = input("Enter maximum sentiment score (1-100): ")
                    
                    responses_df = pd.read_sql_query("""
                        SELECT r.response_id, q.question_id, q.question_text, r.response_text, r.sentiment_score
                        FROM responses r
                        JOIN questions q ON r.question_id = q.question_id
                        WHERE r.sentiment_score BETWEEN ? AND ?
                        ORDER BY r.sentiment_score DESC
                    """, conn, params=[min_score, max_score])
                    
                    print(f"\nResponses with sentiment score between {min_score} and {max_score}:")
                    for _, row in responses_df.iterrows():
                        print(f"Question ID: {row['question_id']}, Response ID: {row['response_id']}, Score: {row['sentiment_score']}")
                        print(f"Question: {row['question_text']}")
                        print(f"Response: {row['response_text']}")
                        print("-" * 50)
            
                case '4':
                    stats_df = pd.read_sql_query("""
                        SELECT 
                            q.question_text,
                            COUNT(r.response_id) as total_responses,
                            AVG(r.sentiment_score) as avg_sentiment,
                            MIN(r.sentiment_score) as min_sentiment,
                            MAX(r.sentiment_score) as max_sentiment
                        FROM questions q
                        LEFT JOIN responses r ON q.question_id = r.question_id
                        GROUP BY q.question_id
                    """, conn)
                    
                    print("\nSummary Statistics:")
                    for _, row in stats_df.iterrows():
                        print(f"\nQuestion: {row['question_text']}")
                        print(f"Total Responses: {row['total_responses']}")
                        print(f"Average Sentiment: {row['avg_sentiment']:.2f}")
                        print(f"Range: {row['min_sentiment']} - {row['max_sentiment']}")
                        print("-" * 50)

                case '5':
                    print("\n📝 Custom SQL Query")
                    print("Available tables: 'questions' and 'responses'")
                    print("Example: SELECT * FROM questions WHERE question_id = 1")
                    
                    try:
                        query = input("\nEnter your SQL query: ").strip()
                        
                        if query.lower().startswith(('insert', 'update', 'delete', 'drop', 'alter')):
                            print("❌ Error: Only SELECT queries are allowed for safety")
                            continue
                            
                        if not query.lower().startswith('select'):
                            print("❌ Error: Query must start with SELECT")
                            continue
                        
                        # Execute query and fetch results
                        result_df = pd.read_sql_query(query, conn)
                        
                        if result_df.empty:
                            print("\nNo results found.")
                        else:
                            print("\n📊 Query Results:")
                            print(f"Found {len(result_df)} rows")
                            print("\nColumns:", ", ".join(result_df.columns))
                            print("\nResults:")
                            pd.set_option('display.max_columns', None)
                            pd.set_option('display.width', None)
                            print(result_df.to_string())
                            
                            # Option to save results
                            save = input("\nWould you like to save these results to CSV? (y/n): ")
                            if save.lower() == 'y':
                                filename = f"custom_query_results_{int(time.time())}.csv"
                                result_df.to_csv(filename, index=False)
                                print(f"✅ Results saved to {filename}")
                                
                    except pd.io.sql.DatabaseError as e:
                        print(f"❌ SQL Error: {str(e)}")
                    except Exception as e:
                        print(f"❌ Unexpected error: {str(e)}")
                        
                case '6':
                    delete_db_entries()
                case '7':
                    print("Exiting query tool...")
                    break
                case _:
                    print("Invalid choice. Please try again.")
                
            input("\nPress Enter to continue...")
            os.system('cls')  # Clear screen for Windows
            
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    finally:
        conn.close()

def delete_db_entries():
    try:
        conn = sqlite3.connect('mental_health_data.db')
        cursor = conn.cursor()

        print("\n🗑️ Delete Database Entries")
        print("1. Delete from responses table")
        print("2. Delete from questions table")
        print("3. Back to main menu")

        table_choice = input("\nSelect table: ")
        
        if table_choice not in ['1', '2']:
            return

        table_name = 'responses' if table_choice == '1' else 'questions'
        id_column = 'response_id' if table_choice == '1' else 'question_id'

        # Show current entries
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        print(f"\nCurrent {table_name}:")
        print(df)

        # Get IDs to delete
        print("\nEnter IDs to delete (comma-separated, e.g., '1,2,3' or single ID):")
        ids_input = input().strip()
        
        try:
            # Convert input to list of integers
            ids_to_delete = [int(id.strip()) for id in ids_input.split(',')]
            
            # Confirm deletion
            print(f"\nYou are about to delete {len(ids_to_delete)} entries with IDs: {ids_to_delete}")
            confirm = input("Are you sure? (y/n): ")
            
            if confirm.lower() == 'y':
                # Convert list to tuple for SQL IN clause
                ids_tuple = tuple(ids_to_delete)
                
                if len(ids_tuple) == 1:
                    # Special case for single ID
                    cursor.execute(f"DELETE FROM {table_name} WHERE {id_column} = ?", (ids_tuple[0],))
                else:
                    cursor.execute(f"DELETE FROM {table_name} WHERE {id_column} IN {ids_tuple}")
                
                conn.commit()
                print(f"✅ Successfully deleted {cursor.rowcount} entries")
                
                # Show updated entries
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                print(f"\nUpdated {table_name}:")
                print(df)
            else:
                print("Deletion cancelled")
                
        except ValueError:
            print("❌ Invalid ID format. Please enter numbers separated by commas")
        except sqlite3.Error as e:
            print(f"❌ Database error: {e}")
            conn.rollback()
            
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    finally:
        conn.close()
    

def display_menu():
    print("\n=== Mental Health Question & Response Generator ===")
    print("1. Generate Refined Questions")
    print("2. Generate Responses")
    print("3. Query Database")
    print("4. Label Sentiments")
    print("0. Exit")
    return input("\nSelect an option: ")

def main():
    while True:
        choice = display_menu()

        match choice:
            case '1':
                generate_refined_questions()
            case '2':
                try:
                    num_responses = int(input("Enter number of responses to generate per question: "))
                    if num_responses <= 0:
                        print("❌ Please enter a positive number")
                        continue
                    generate_responses(num_responses)
                except ValueError:
                    print("❌ Please enter a valid number")
            case '3':
                query_db()
            case '4':
                label_sentiments()
            case '0':
                print("Exiting...")
                sys.exit()
            case _:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()