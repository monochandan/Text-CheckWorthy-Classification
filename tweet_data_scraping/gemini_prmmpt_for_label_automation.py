import base64
import os
import google.generativeai as genai
import warnings
# from google.genai import types
from dotenv import load_dotenv, find_dotenv
import pandas as pd

# load env file for api key
_ = load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")

api_key = os.getenv('GEMINI_API_KEY')
# model = 'gemini-1.5-flash'
# MODEL = os.getenv("model", "default_model_name")  # Replace 'default_model_name' with an actual model name

# configure genai with the api key
genai.configure(api_key=api_key)
# model = genai.GenerativeModel(MODEL)

# print(f"Model: {MODEL}")


def get_lebel(text, prompt):#
    '''
    receiving Text from the dataframe and provided prompt to the model to generare the labels
    '''
    value = 0
    # https://googleapis.github.io/python-genai/#with-uploaded-file-gemini-developer-api-only
    prompt = prompt.format(text = text)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)

    # if response.text == "Yes":
    #     print(f"processing")
    #     value = 1
    # else:
    #     print(f"processing")
    #     value = 0
    # return value
    return response.text
    #print(i)


few_shot_prompt = """
Read the following statement carefully.
Your task is to determine whether the statement contains claims or information that require fact-checking.
- If the statement present factual assertions, statistics or claims that can be verified for accuracy, respond with 'Yes'.
- If the statement is purely opinion-based, general knowledge, trivial or lacks verifiable claims, respond with 'No'.

Examples:
Statement: "I served for eight years in the House of Representatives and I served on the Intelligence Committee, specialized in looking at arms control."
Response: Yes

Statement: "But I do make a pledge that in the next ten days when we're asking the American people to make one of the most important decisions in their lifetime, because I think this election is one of the mast vital in the history of America, that uh - we do together what we can to stimulate voter participation."
Response: No

Statement: "They're the ones who have the challenges and they are people out there making predictions that it's not going to be the same."
Response: No

Statement: "And, as a matter of fact, it just so happens that in the quirks of administering these taxes, those above $50,000 actually did not get quite as big a tax cut percentage-wise as did those from 50,000 down."
Response: Yes

Statement: "The platform to which the President refers, in fact, calls for a religious test in the selection of judges."
Response: No

Statement: "And we're in real trouble on that."
Response: No

Statement: "Third thing I think we need is more forums like this, which is one of the reasons I have so strongly supported campaign finance reform."
Response: No

Statement: "Help me pass these programs."
Response: No

Statement: "I know a lot of wealthy people that have never been audited."
Response: No

Statement: "But when you talk about apology, I think the one that you should really be apologizing for and the thing that you should be apologizing for are the 33,000 e-mails that you deleted, and that you acid washed, and then the two boxes of e-mails and other things last week that were taken from an office and are now missing."
Response: Yes 

Statement: "There won't be 10 millionaires and 14 lawyers in the cabinet."
Response: No

Statement: "And we better be awfully careful."
Response: No

Statement: "Well, I think we have to go to work on the problem of Third World debt and we've got to assist those Third World countries in dealing with this massive debt which they currently-which they have incurred and which is burdening them and which if we don't do something about it and assist them along with other nations around the world, we'll destroy their economies, destroy their future."
Response: No

Statement: "And I made some tough decisions."
Response: No 

Statement: "If we're $4 trillion down, we should have everything perfect, but we don't."
Response: Yes

Statement: "But the Russians, I think we can deal with them but they've got to understand that they're facing a very firm and determined United States of America that will defend our interests and that of other countries in the world."
Response: No 

Statement: "One of the reasons I'm such a strong believer in legal reform is so that people aren't afraid of producing a product that is necessary for the health of our citizens and then end up getting sued in a court of law."
Response: No

Statement: I don't know how you vote "present" on some of that."
Response: No

Statement: "And yet, because of the hard work of the American people and good policies, this economy is growing."
Response: Yes

Now, Classify the following statement:
Statement: "{text}"
Response: 
"""

data_df = pd.read_csv('C:/Users/looka/OneDrive/Documents/Thesis/tweeter_data/final_tweet_data.csv')
# print(data_df['Text'].iloc[0])

data_df['label'] = data_df['Text'].apply(lambda text: get_lebel(text, few_shot_prompt))

# saving in a new csv file 
data_df.to_csv('C:/Users/looka/OneDrive/Documents/Thesis/tweeter_data/gold_tweet_test_gemini.csv', index = False)
