from agent import CinematographyAgent

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize agent
    agent = CinematographyAgent(model="gpt-4o-mini")
    
    print("🎬 Cinematography AI Agent")
    print("=" * 60)
    print("Commands:")
    print("  - Type your question about movies/cinematography")
    print("  - 'history' - View conversation history")
    print("  - 'movies' - See movies discussed")
    print("  - 'save' - Save conversation")
    print("  - 'load' - Load previous conversation")
    print("  - 'quit' - Exit")
    print("=" * 60)
    print()
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye! 🎬")
            break
        
        if user_input.lower() == 'history':
            print("\n--- Conversation History ---")
            for msg in agent.get_conversation_history():
                preview = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
                print(f"\n{msg['role'].upper()}: {preview}")
            print()
            continue
        
        if user_input.lower() == 'movies':
            movies = agent.get_discussed_movies()
            print(f"\n🎬 Movies discussed: {', '.join(movies) if movies else 'None yet'}\n")
            continue
        
        if user_input.lower() == 'save':
            agent.save_conversation()
            print("✅ Conversation saved to conversation.json\n")
            continue
        
        if user_input.lower() == 'load':
            agent.load_conversation()
            print("✅ Conversation loaded from conversation.json\n")
            continue
        
        if not user_input:
            continue
        
        # Run agent
        response = agent.run(user_input, verbose=True)
        print(f"\n💬 Assistant: {response}\n")
        print("-" * 60)