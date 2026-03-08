from groq import Groq

client = Groq(api_key="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")

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "user",
            "content": "Write a SQL query to get the top 3 highest paid employees."
        }
    ]
)

print(response.choices[0].message.content)