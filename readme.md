# Multi-Modal Psychology Data

This repository is the official data of [My Paper Title-arxivlink](./)  

The dataset contains **1275 real-students data** including mental scale score, therapy interview audio recordings and text.   

```
data_structure
.
├── data
│   ├── np_data
│   │   ├── user_*
│   │   │   ├── sentence_embeddings
│   │   │   │   └── *.npy
│   │   │   └── mel-spectrogram
│   │   │       └── *.npy
│   │   └── np_data.json
│   └── wav_select.json
├── model
|   └── chinese-bert-wwm-ext
│       └── *
├── data_generation.py
└── README.md              
```

## Overview

Our ongoing project aim to build an Artificicial Intelligence(AI) system using therapy conversation information as multimodal input to help automatically detecting psychological distress status, including depression and anxiety levels. 
Our system takes a student's multimodal signals, such as speech and language, as the input signals, then uses advanced AI model to map these inputs onto quantitative indicators of student distress status. 
To achieve our ultimate goal, real students' data was collected with special designed therapy conversation and mental sacles using online service. 
Specifically, we calculated students' mental scale score and recorded their audio with a computer-based system.  

## Data Source

Our data was collected from over 15 thousand middle school students in the Guangdong Province of China. 
The Deparment of Education of the local government requires every middle school student to have a mental health evaluation using standard psychological scales such GAD-7(General Anxiety Disorder-7) and MHT(Mental Health Test) at the beginning of each school year. 
We collaborate with the Department for Education, the schools to build a computer-based system administrating psychological scales and therapy conversation to each student. 
Under the authorization of Department of Education, all the collected data are anonymized to remove the sensitive identify information.  
**We share 1000 students' anonymized data in this respository.**

## Data Description

Our data could be divided into 2 parts, one is standard psychological health scale socre, which would be indicator of mental level, and the other is multimodal data including audio recording and text, which would be input signals. 

### Scale

The standard level indicator is composed by three different scales: PHQ-9(Patient Health Questionnaire-9), GAD-7(General Anxiety Disorder-7) and MHT(Mental Health Test).  

- **PHQ-9**: Standard measurement of depression with 9 questions. 
The answer score varies from 0 to 3 representing `not at all(0)`, `several days(1)`,`more than half the days(2)`,`nearly every day(3)` respectively, and the total questionaire scores 0-27.  
    - Score 0-4: Minimal depression
    - Score 5-9: Mild depression
    - Score 10-14: Moderate depression
    - Score 15-19: Moderately severe depression
    - Score 20-27: Severe depression  
    > To make the final score be reliable ground truth, we modified the scale by adding 2 additional question, one is a promision and the other is polygraph question. Further detail would be illustrate latter.  
    

- **GAD-7**: Standard measurement of anxiety with 7 questions. 
The answer score varies from 0 to 3 representing `not at all(0)`, `several days(1)`,`more than half the days(2)`,`nearly every day(3)` respectively, and the total questionaire scores 0-21.  
    - Score 0-4: Minimal anxiety
    - Score 5-9: Mild anxiety
    - Score 10-14: Moderate anxiety
    - Score 15-21: Severe anxiety  
    > To make the final score be reliable ground truth, we modified the scale by adding 2 additional question, one is a promision and the other is polygraph question. Further detail would be illustrate latter.  
- **MHT**: 100 questions generally used in primary school and middle to evaluate 10 mental health dimensions. 
All the answers are `yes(1)` or `no(0)`, and the sum of these questions varies from 0 to 100.
    1. Social Anxiety
    2. Academic Anxiety
    3. Tendency toward Loneliness
    4. Tendency toward Self-blame
    5. Tendency toward Hypersensitivity
    6. Physical Symptoms
    7. Tendency toward Terror
    8. Tendency toward Impulsivity
    9. Substance use or abuse(To check whether student answers carefully)  
    > To make the final score be reliable ground truth, we trained a random forest machine learning classification model based on other modified scales data(like PHQ-9 and GAD-7). 
    With over 0.9 overall f1-score, our trained model filtered several MHT answers which could be considered as reliable ground truth.

### Audio Recordings

To collecte multi-turn conversation recordings, we designed a question system contains 7-10 different topics concentrating on student's recent school life. Such as *Do you sleep well in recent 2 weeks, and could you share your experience?* or *In recent 2 weeks, do you have any study problem and how do you deal with it?* Students were asked question with a unfixed path, which means what current question each student would be asked depends on the previous answer.  
Due to the concerns about privacy confidentiality restricted by Department of Education, we release only extreated mel-spectrogram feature from audio as open-source data. `./data_generation.py` contains the code of our feature extracting method. We firstly concatente each student's audio together, combining individual audio segments into one cohesive audio stream. After that, we proceed to extract text features by converting the audio content into text representations. A pre-trained BERT model is utilized to encode text into a numerical tensor. Simulataneously with text feature extraction, mel-spectrograms is computed by involving `Short-Time Fourier Transform` to audio signal in order to approximate the human ear's frequency sensitivity. Following the generation of mel-spectrograms, we segment the data into smaller frames or segments. This segmentation process involvues dividing the continuous audio stream and associated mel-spectrogram into smaller, overlapping segments. Each segment typically corresponds to a 60 secondss duration, with a 10 seconds certain overlap between consecutive segments.

## Data Fields

[List and describe each field (column) in the dataset. Include data type, description, and any possible values.]  
**Scale Data**: 

## Data Examples

[Provide a few examples of data entries to give users a sense of what the dataset looks like.]

## Usage

[Offer guidance on how to use the dataset, including any code examples, tutorials, or analysis scripts.]

## License

[Specify the license under which the dataset is released. This could be an open-source license like MIT, GPL, or Creative Commons, or a custom license.]

## Citation

[Provide a citation format for users to reference when citing your dataset.]

## Contributing

[Explain how others can contribute to the dataset, whether it's through adding new data, improving documentation, or fixing issues.]

## Acknowledgements

[Give credit to any individuals or organizations that contributed to the creation of the dataset.]

## Contact

[Provide contact information for users to reach out with questions, feedback, or collaboration opportunities.]

---

3344 MB

