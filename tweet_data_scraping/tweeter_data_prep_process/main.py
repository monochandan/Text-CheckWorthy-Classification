import csv
from twikit import Client, TooManyRequests
import time
from datetime import datetime
from configparser import ConfigParser
from random import randint
import asyncio

MINIMUM_TWEETS = 100
# QUERY = 'chatgpt'
# QUERY = "(COVID OR vaccine OR pandemic OR lockdown OR #COVID19 OR #VaccineMandate) OR " \
#         "(elections OR Biden OR Trump OR Joe Biden OR DOGE OR FBI OR Donald Trump OR Ukraine OR " \
#         "Russia OR Middle East Crisis OR South Asia Crisis OR UN meeting OR US Congress OR US Republic OR geopolitics OR War OR" \
#         " #Politics OR #Election2025  OR #UkraineRussiaWar OR #Trump OR #DOGE OR #Trump2025 #MiddleEastCrisis OR" \
#         "#Geopolitics OR #USRepublic OR #USCongress OR #Bangladesh)"

QUERY = "COVID OR vaccine OR pandemic OR lockdown OR #COVID19 OR " \
        "#VaccineMandate elections OR Biden OR Trump OR Joe Biden OR " \
        "DOGE OR FBI OR Donald Trump OR Ukraine OR Russia OR " \
        "Middle East Crisis OR South Asia Crisis OR UN meeting OR" \
        " US Congress OR US Republic OR geopolitics OR War OR" \
        " #Politics OR #Election2025  OR #UkraineRussiaWar OR" \
        " #Trump OR #DOGE OR #Trump2025 #MiddleEastCrisis OR" \
        " #Geopolitics OR #USRepublic OR #USCongress OR #Bangladesh (COVID OR" \
        " OR OR vaccine OR OR OR pandemic OR OR OR lockdown OR OR OR" \
        " #COVID19 OR OR OR #VaccineMandate OR elections OR OR OR" \
        " Biden OR OR OR Trump OR OR OR Joe OR Biden OR OR OR" \
        " DOGE OR OR OR FBI OR OR OR Donald OR Trump OR OR OR" \
        " Ukraine OR OR OR Russia OR OR OR Middle OR " \
        "East OR Crisis OR OR OR South OR Asia OR Crisis OR OR OR" \
        " UN OR meeting OR OR OR US OR Congress OR OR OR US OR" \
        " Republic OR OR OR geopolitics OR OR OR War OR OR OR" \
        " #Politics OR OR OR #Election2025 OR OR OR #UkraineRussiaWar OR OR OR" \
        " #Trump OR OR OR #DOGE OR OR OR #Trump2025 OR #MiddleEastCrisis OR OR OR" \
        " #Geopolitics OR OR OR #USRepublic OR OR OR #USCongress OR OR OR #Bangladesh)" \
        " lang:en until:2022-12-31 since:2020-01-01"

## login cridentials

config = ConfigParser()
config.read('config.ini')

username = config['X']['username']
email = config['X']['email']
password = config['X']['password']

# create csv file
with open('tweets.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['tweet_count','tweet_id', 'user_name', 'post_text', 'tweet_created_at', 'tweet_retweet_count', 'tweet_favorite_count'])
    


## authenticate to X.com
#! 1) use the login cridentials 2) use cookies
# https://twikit.readthedocs.io/en/latest/twikit.html#twikit-twitter-api-wrapper
client = Client(language = 'en-US')
# async def main():
#     await client.login(
#         auth_info_1=username, 
#         auth_info_2=email, 
#         password = password,
#         cookies_file='cookies.json'
#         )
    
# asyncio.run(main())
# # save the cookies - skip the loging procedure by saving the cookies
# client.save_cookies('my_cookies.json')

# loading the save cookies
client.load_cookies('cookies.json')

async def get_tweets(tweets):
    if tweets is None:
        print(f"{datetime.now()} - Getting tweets......")
        tweets = await client.search_tweet(query= QUERY, product = 'Top')
    else:
        # creating delay  to not to banned
        wait_time = randint(5, 10)
        print(f"{datetime.now()}- Gettng more tweets after {wait_time} seconds......")
        time.sleep(wait_time)
        tweets = await tweets.next()
    return tweets

# ## get tweets
async def fetch_tweet():
    tweet_count = 0
    tweets = None
    
    while tweet_count < MINIMUM_TWEETS:
        '''if we want  more tweets, lets say 1000, that means we ahve call the api so many times. 
        It may bann our account also
        ratelimitexception - so we need to handle this exception'''
        try:
            tweets = await get_tweets(tweets)
        except TooManyRequests as e:
            rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
            print(f"{datetime.now()} - Rate limit reached. Waiting until {rate_limit_reset}")
            wait_time = rate_limit_reset - datetime.now()
            time.sleep(wait_time.total_seconds())
            continue

        # tweets = await get_tweets(tweets)
        # if tweets is None:
        #     print(f"{datetime.now()} - Getting tweets......")
        #     tweets = await client.search_tweet(query= QUERY, product = 'Top')
        # else:
        #     print(f"{datetime.now()}- Gettng more tweets......")
        #     tweets = await tweets.next()

        if not tweets: # some time there is no tweets
            print(f"{datetime.now()} - No more tweets found")
            break

        for tweet in tweets:
            tweet_count += 1
            tweet_data = [tweet_count, tweet.id, tweet.user.name, tweet.text, tweet.created_at, tweet.retweet_count, tweet.favorite_count]
            # tweet_data = {
            #                 'tweet_id': tweet_count, 
            #                 'user_name': tweet.user.name, 
            #                 'post_text': tweet.text, 
            #                 'tweet_created_at':tweet.created_at,
            #                 'tweet_retweet_count':tweet.retweet_count, 
            #                 'tweet_favorite_count': tweet.favorite_count
            #                 }
            with open('tweets.csv', 'a', newline = '', encoding = 'utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(tweet_data)
            #print(tweet_data)
        #break
        print(f"{datetime.now()}- Total tweets got: {tweet_count}")

asyncio.run(fetch_tweet())

