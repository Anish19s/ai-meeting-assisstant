from google import genai
from dotenv import load_dotenv
import time
from google.genai.errors import APIError


def generate_summary(result,client):

    prompt=f'''
    You are an AI Meeting Intelligence Assistant.

    Analyze the following meeting transcript and provide:

    1. Executive Summary
    - 5-10 concise bullet points covering the main discussion.

    2. Action Items
    - Task
    - Owner (if mentioned)
    - Deadline (if mentioned)

    3. Key Decisions
    - List all decisions that were finalized during the meeting.

    4. Risks and Concerns
    - Potential risks, blockers, concerns, or unresolved issues discussed.

    5. Open Questions
    - Questions that were raised but not answered or finalized.

    Instructions:
    - Use only information present in the transcript.
    - Do not invent details.
    - If a section has no relevant information, write "Not mentioned."
    - Keep the output clear and structured.

    {result}
    '''


    # 2. Wrapped in a retry loop to instantly bypass the 503 and 429 errors
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                # 3. Upgraded to 2.5-flash for the fastest response and lowest failure rate
                model="gemini-2.5-flash",
                contents=prompt
            )
            break # Success! Break out of the retry loop.

        except APIError as e:
            if e.code in [429, 503] and attempt < 2:
                print(f"Server busy or rate limited ({e.code}). Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"An unexpected error occurred: {e}")
                raise e
            
    return response.text