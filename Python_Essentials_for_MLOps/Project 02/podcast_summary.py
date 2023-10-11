"""All Imports"""
import logging
import os
import json
import requests
import xmltodict
import pendulum

from airflow.decorators import dag, task
from airflow.providers.sqlite.operators.sqlite import SqliteOperator
from airflow.providers.sqlite.hooks.sqlite import SqliteHook
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

logging.basicConfig(filename='errors.log', level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def log_errors(func):
    """
    Decorator para capturar erros e escrever no log
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as val_err:
            logging.error('Erro na função %s : %s', {func.__name__},  {val_err}, exc_info=True)
        except NameError as name_err:
            logging.error('Erro na função %s : %s', {func.__name__},  {name_err}, exc_info=True)

    return wrapper

# URL of the podcast feed and other constants
PODCAST_URL = "https://www.marketplace.org/feed/podcast/marketplace/"
EPISODE_FOLDER = "episodes"
FRAME_RATE = 16000
logging.debug("Defining parameters: \n" +
              "PODCAST_URL = {PODCAST_URL} \n" + 
              "EPISODE_FOLDER = {EPISODE_FOLDER} \n FRAME_RATE = {FRAME_RATE}")

# Define the Airflow Directed Acyclic Graph (DAG) with the given properties
@dag(
    dag_id='podcast_summary',
    schedule_interval="@daily",
    start_date=pendulum.datetime(2022, 5, 30),
    catchup=False,
)
@log_errors
def podcast_summary():
    """
    The principal function, creates a database to store podcast episodes, 
    retrieves episodes from a specified URL, loads new episodes into the 
    database, downloads audio files, performs speech-to-text transcription,
    and updates the database with transcripts.
    """
    # Task to create the SQLite database table if it doesn't exist
    create_database = SqliteOperator(
        task_id='create_table_sqlite',
        sql=r"""
        CREATE TABLE IF NOT EXISTS episodes (
            link TEXT PRIMARY KEY,
            title TEXT,
            filename TEXT,
            published TEXT,
            description TEXT,
            transcript TEXT
        );
        """,
        sqlite_conn_id="podcasts"
    )

    # Task to retrieve podcast episodes from the specified URL
    @log_errors
    @task()
    def get_episodes():
        """
        Retrieves podcast episodes from the specified URL and returns them.
        """
        data = requests.get(PODCAST_URL, timeout=30)
        feed = xmltodict.parse(data.text)
        episodes = feed["rss"]["channel"]["item"]
        logging.debug("Found %d episodes.", {len(episodes)})
        return episodes

    # Task to load new podcast episodes into the SQLite database
    podcast_episodes = get_episodes()
    create_database.set_downstream(podcast_episodes)

    @log_errors
    @task()
    def load_episodes(episodes):
        """
        Task to filter and load new episodes into the database
        """
        hook = SqliteHook(sqlite_conn_id="podcasts")
        stored_episodes = hook.get_pandas_df("SELECT * from episodes;")
        new_episodes = []
        for episode in episodes:
            if episode["link"] not in stored_episodes["link"].values:
                filename = f"{episode['link'].split('/')[-1]}.mp3"
                new_episodes.append([episode["link"],
                                     episode["title"],
                                     episode["pubDate"], episode["description"], filename])

        hook.insert_rows(table='episodes',
                         rows=new_episodes,
                        target_fields=["link", "title", "published", "description", "filename"])
        return new_episodes

    #new_episodes = load_episodes(podcast_episodes)

    # Task to download audio files of new episodes
    @log_errors
    @task()
    def download_episodes(episodes):
        """
        Downloads audio files of new episodes and returns their information.
        """
        audio_files = []
        for episode in episodes:
            name_end = episode["link"].split('/')[-1]
            filename = f"{name_end}.mp3"
            audio_path = os.path.join(EPISODE_FOLDER, filename)
            if not os.path.exists(audio_path):
                audio = requests.get(episode["enclosure"]["@url"], timeout=30)
                with open(audio_path, "wb+") as f:
                    f.write(audio.content)
            audio_files.append({
                "link": episode["link"],
                "filename": filename
            })
            logging.debug("Adding a new audio file, %s", {filename})
        return audio_files

    #audio_files = download_episodes(podcast_episodes)

    # Task to perform speech-to-text transcription on audio files and update the database
    @log_errors
    @task()
    def speech_to_text():#audio_files, new_episodes):
        """
        Performs speech-to-text transcription on audio files and updates 
        the database with the transcripts.
        """
        hook = SqliteHook(sqlite_conn_id="podcasts")
        untranscribed_episodes = hook.get_pandas_df(
            "SELECT * from episodes WHERE transcript IS NULL;")

        model = Model(model_name="vosk-model-en-us-0.22-lgraph")
        rec = KaldiRecognizer(model, FRAME_RATE)
        rec.SetWords(True)

        for _, row in untranscribed_episodes.iterrows():
            logging.debug("Transcribing %s", {row['filename']})
            filepath = os.path.join(EPISODE_FOLDER, row["filename"])
            mp3 = AudioSegment.from_mp3(filepath)
            mp3 = mp3.set_channels(1)
            mp3 = mp3.set_frame_rate(FRAME_RATE)
            step = 20000
            transcript = ""
            for i in range(0, len(mp3), step):
                segment = mp3[i:i+step]
                rec.AcceptWaveform(segment.raw_data)
                result = rec.Result()
                text = json.loads(result)["text"]
                transcript += text
            hook.insert_rows(table='episodes',
                             rows=[[row["link"], transcript]],
                             target_fields=["link", "transcript"], replace=True)

    # Uncomment this to try speech to text (may not work)
    speech_to_text()

#summary = podcast_summary()
