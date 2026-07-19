import os
from openai import OpenAI

# 1. Safely get the API key (prevents crashes if it's missing)
api_key = os.environ.get("MOONSHOT_API_KEY")

if not api_key:
    print("❌ ERROR: 'MOONSHOT_API_KEY' environment variable is not set!")
    print("💡 Fix: Make sure you ran 'set MOONSHOT_API_KEY=your_key' or it's in your .env file.")
else:
    print("✅ API Key found. Testing connection to Moonshot API...")
    
    # 2. Initialize client with the official base URL
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1",
    )

    try:
        # 3. Use the correct, publicly available model name
        completion = client.chat.completions.create(
            model="moonshot-v1-8k", 
            messages=[
                {"role": "system", "content": "You are Kimi, an AI assistant provided by Moonshot AI."},
                {"role": "user", "content": "Hello, my name is Li Lei. What is 1+1?"}
            ],
        )
        print("\n🎉 SUCCESS! The API is working perfectly.")
        print("🤖 Kimi's Response:", completion.choices[0].message.content)
        
    except Exception as e:
        print("\n❌ FAILED:", e)
        print("💡 If this says '401 Invalid Authentication', your API key is incorrect, revoked, or you need to add a small balance to your Moonshot account.")