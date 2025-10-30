# Key Phrase detection MLaaS

> The user enters a phrase. The app trains my neural network model to detect that phrase in audio. 
> 
> The user receives the trained model wrapped around a clean library/API to allow phrase detection in live audio.

## Model

1. A **log-mel** spectrogram of the audio is made.

2. **CNN** extracts some local features for further processing

3. **RNN** extracts information for each time step, describing the audio up to that point. Here, dropout is applied to avoid overfitting.

4. **MLP** turns per-time-step information into scores indicating likelihood of key phrase occuring up to the point of each time step

5. **Temporal pool** turns this score into one number that indicates likelihood of key phrase occuring in the whole clip.

## Data

- All data is augmented - pitch shifting, adding noise

- Negatives are clips containing **near confusers** (similar-sounding phrases), **modifications of the phrase**, and random clips of audio not containing the phrase. There's also some small number of pure noise clips.

- Positives are more self-explanatory

- Multiple voices are used, both male and female

- More positives than negatives in dataset for each query

## Generating data

- There's a **Sqlite3** datatabase of already existing samples, so that they can be reused for many queries

- **TTS tools** (Coqui, Piper, ElevenLabs) and **public speech datasets** (AMI Meeting Corpus, The People’s Speech). In the future, the user will be able to send own voice clips

- For each query, we look at each category of test (negatives, positives, and specific types of negatives, different sources) to see how many tests we already have in each category. We add these to the dataset. 
  Then, we generate additional tests and add them to the DB and the dataset. 

- It's important to avoid adding too many tests of a given category.

# 




