'''
#   Project: SesameAI Labs CSM Demo
#   Description: Demo showing how to use the newly released SeasemeLabs CSM AI
#   Date: 3/17/2025
#   Developer: Robert Burkhall
#
#   Installation:
#       CPU) Run pip install -r requirements.txt
#       GPU) Run pip install -r requirements-cuda.txt
# See Pytorch Notes:
#   https://pytorch.org/get-started/locally/#windows-package-manager
#
#
'''

from huggingface_hub import hf_hub_download
from generator import load_csm_1b

import torchaudio
import torch
import subprocess
import warnings
import os
import sys

import pygame
import pygame_menu

import newspaper
import feedparser

import time

#
#   Variables
#
# HuggingFace Token
FHTOKEN='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'

# Model for Voice
SESAMECSM="sesame/csm-1b"

# File output
file_path=".\\audio.wav"  # Replace with your file path

# Load Expected path
csmmodel='~\\.cache\\huggingface\\hub\\models--sesame--csm-1b\\'

# Output File
OUTPUTWAV='.\\audio.wav'


# Short Text to ouput
helloMsg="Hello, Sesame here.  It's the weekend, Did you get your groove on and have some fun, Or, did you spend time working with the new Sesame AI Labs to generate this message?"

# hello Mike
helloBugs="Hello Bugs Bunny! Friday is a great day to get funded, Yeah!  We, are closing in on getting our insurance to begin operations for a summer release!"

# Long Text to output
newsMsg="""Hello. Let's get to some news.  Hydropower dam walls restrict the flow of rivers and turn them into pools of stagnant water. As these reservoirs age, organic matter like algal biomass and aquatic plants accumulates and eventually decomposes and sinks. That oxygen-poor environment stimulates methane production. 
Reservoir surfaces and turbines then release methane into the atmosphere."""

ryderzMsg="""The website, www.ryderz.us, is the online hub for Ryderz Platform.  Pronounced riders, 
a rideshare platform designed to connect drivers with passengers seeking convenient, 
affordable, and reliable transportation. Ryderz likely operates through a user-friendly 
interface, accessible via its website and possibly a mobile app, allowing customers to 
book rides on-demand or in advance. """

# Substitution feeds, Cal getNews(feeds[x])
feeds = ("https://podnews.net/rss","https://podnews.net/rss/articles","https://podcastfeeds.nbcnews.com/msnbc-rachel-maddow"
         ,"https://podnews.net/prebuilt/rss-press-releases.xml","https://podcastfeeds.nbcnews.com/dateline-nbc")

MAXARTICLES=25
generator=None

#########################################

#
#   Functions
#

def play_file_with_ffplay(file_path):
    """Plays a media file using ffplay.

    Args:
        file_path: The path to the media file.
    """
    command = ['ffplay', '-nodisp', '-autoexit', '-hide_banner', file_path]
    try:
        subprocess.run(command)
    except subprocess.CalledProcessError as e:
        print(f"Error playing file: {e}")
    except FileNotFoundError:
        print("ffplay not found. Ensure FFmpeg is installed and in your PATH.")

def checkCUDAStatus():
    """ Checks the CUDA Installation status
    """
    device=''
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of CUDA devices: {num_gpus}")

        # Get the name of the current CUDA device
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {torch.cuda.get_device_name(current_device)}")

        # Create a tensor on the GPU
        device = torch.device("cuda")
        a = torch.tensor([1.0, 2.0, 3.0]).to(device)
        print(f"Tensor on GPU: {a}")
    else:
        print("CUDA is not available. Please check your installation.")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Found Device:", device)    
    return device

def loginHF():
    print("Attempting Login to HF..")
    command=['huggingface-cli', 'login', '--add-to-git-credential', '--token', FHTOKEN]
    try:
        subprocess.run(command)
    except:
        print('HF Login Failed!')

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

def playContextDemo(tts):
    start=time.time()
    print('Starting Context Demo ...',start)

    from generator import Segment
    speakers = [0, 1, 0, 0]
    transcripts = [
        "Hey how are you doing.",
        "Pretty good, pretty good.",
        "I'm great.",
        "So happy to be speaking to you.",
    ]
    audio_paths = [
        ".\\female-audio.wav",
        ".\\male-audio.wav",
        ".\\female-audio.wav",
        ".\\male-audio.wav",
    ]

    print('Sample Rate:', generator.sample_rate)

    segments = [
        Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
        for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
    ]
    audio = generator.generate(
        text=tts,
        speaker=0,
        context=segments,
        max_audio_length_ms=90_000,
    )

    torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
    end=time.time() 
    print('Elapsed: ', end - start)    

    if (os.path.exists(OUTPUTWAV)):
        play_file_with_ffplay(OUTPUTWAV)

def playNoContextDemo(tts=newsMsg):
    print('Starting No Context Demo...')

    # Select which message to create
    msg=helloBugs
      
    # Login to Hugging Face
    loginHF()

    # Turn off Warnings
    warnings.filterwarnings("ignore")

    # Get Device
    device=checkCUDAStatus()

    #
    # Check for existing model
    # 
    try:
        hf_hub_download.model_info(SESAMECSM)
    except:
        print('No Model found when info was requested!')

    if (os.path.exists(csmmodel)):
        print('Found Existing Path. ',csmmodel)
        model_path=csmmodel
    else:
        print('Fetching Sesame CSM Model from HuggingFace...')
        model_path = hf_hub_download(repo_id=SESAMECSM, filename="ckpt.pt")
        print('Model Fetch Completed.')

    print('Model Path: ', model_path)


    # Use model to Generate Text
    try:
        print('Loading CSM Model for Audio Generation...')
        #generator = load_csm_1b(device)
        audio = generator.generate(
            text=tts,
            speaker=0,
            context=[],
            max_audio_length_ms=90_000,
        )
        # Save generated audio
        print('Saving Audio output for listening in ',OUTPUTWAV,' file.')
        torchaudio.save(OUTPUTWAV, audio.unsqueeze(0).cpu(), generator.sample_rate)
    except Exception as e:
        print("Permissions to llama3b is required, No Audio. Error: ", e)

    if (os.path.exists(file_path)):
        play_file_with_ffplay(file_path)
    else:
        print('No Audio wav file generated!')

def playDemo1():
    start=time.time()
    print('Starting Demo 1...',start)
    playContextDemo(helloMsg)
    end=time.time()
    print('Elapsed: ', end - start)    

def playDemo2():
    start=time.time()
    print('Starting Demo 2...',start)
    playContextDemo(helloBugs)
    end=time.time()
    print('Elapsed: ', end - start)    

def playDemo3():
    start=time.time()
    print('Starting Demo 3...',start)
    playContextDemo(ryderzMsg)
    end=time.time()
    print('Elapsed: ', end - start)    

def getNews(url=feeds[0]):
    #tts=''
    start=time.time()    
    print('Fetching News...',start)
    articles = scrape_news_from_feed(url)
    print('Article Count: ',len(articles))
    if len(articles) > 0:
        for article in articles:
            if len(article) != 0:
                print('Title:', article['title'])
                print('Author:', article['author'])
                print('Publish Date:', article['publish_date'])
                print('Content:', article['content'])
                print()
                playContextDemo(article['content'])
                time.sleep(5)
    else:
        print('Fetch from News RSS Feed failed!')
    end=time.time()
    print('Elapsed: ', end - start)    

def scrape_news_from_feed(feed_url):
    articles = []
    feed = feedparser.parse(feed_url)
    articlesRemaining=MAXARTICLES
    for entry in feed.entries:
        # create a newspaper article object
        try:
            article = newspaper.Article(entry.link)
            # download and parse the article
            article.download()
            article.parse()
            # extract relevant information
            articles.append({
                'title': article.title,
                'author': article.authors,
                'publish_date': article.publish_date,
                'content': article.text
            })
            articlesRemaining-=1
            if articlesRemaining == 0:
                return articles
        except:
            print('Failed to build Newspaper Article.')

    return articles

def whisperLive():
    print('Todo: Needing more information to complete feature')

def showMenu():
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    menu = pygame_menu.Menu('SesameAI Demo', 600, 400, theme=pygame_menu.themes.THEME_BLUE)
    menu.add.text_input('Name: ', default=os.getlogin())
    menu.add.button('Context Voice Demo 1', playDemo1)
    menu.add.button('Context Voice Demo 2', playDemo2)
    menu.add.button('Context Voice Demo 3', playDemo3)
    menu.add.button('No Context Demo 1',playNoContextDemo)
    menu.add.button('Get RSS News', getNews)

    menu.add.button('Quit', pygame_menu.events.EXIT)

    menu.mainloop(screen)
 
def cleanRecording():
    try:
        os.remove('.\\audio.wav')
    except:
        print('Continuing as no wav file found.')

#
#######    Start Main Routine Here   ###############################################
#
#
#   Remove Previous wav generated
cleanRecording()

#   Create Generator Object
generator = load_csm_1b(checkCUDAStatus())
generator._model = torch.compile(generator._model)

#   Show Menu to User
showMenu()
