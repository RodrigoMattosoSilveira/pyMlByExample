# Introduction

#Installation
Steps I used to install the environment required by the book:
* Created a [python virtual environment|https://docs.python.org/3/tutorial/venv.html]: `python3 -m venv ~/projects/pyMlBook`;
* Activated the visual environment: `source ~/projects/pyMlExample/bin/activate`;
* [Installed scikit-learn|https://scikit-learn.org/stable/install.html]: `pip install -U scikit-learn`;
* [Installed matplotlib|https://matplotlib.org/users/installing.html]: `pip install -U matplotlib`;
* Installed numpy: `pip install numpy`;
* Installed Natural Language Toolkit [LTK|http://www.nltk.org/]: `pip install -U nltk`;
  * In trying to run [LTK|http://www.nltk.org/] I found out I had to `install python3@3.7`
  * In trying to run `install python3@3.7` I found out that Xcode (10.2 => /Applications/Xcode-beta.app/Contents/Developer) is too outdated, and had
update to Xcode 11.3 (or delete it).
  * Installed `Xcode`
  * Still got error: `DEPRECATION WARNING: The system version of Tk is deprecated and may be removed in a future release. Please don't rely on it. Set TK_SILENCE_DEPRECATION=1 to suppress this warning.``
  * Had to install `python@3.8`, due to incompatibilities with `Tk`
  * Had to recreate the virtual environment!

# Terminology
* `active learning` or human-in-the-loop, which advocates combining the efforts of machine learners and humans;
* `bias`: is the error stemming from incorrect assumptions in the learning algorithm;
* `bias, low`: happens when we extract too much information from the training sets forcing our models to work well with them;
* `bias, high`: happens when we use small training sets, although the model's variance is low as performance in training and test sets are pretty consistent, in a bad way;
* `bias–variance tradeoff`: effort to avoid cases where any of `bias` or `variance` is getting high; since decreasing one increases the other, we should always make both bias and variance as low as possible;
* `chatbots`: see Conversational agents
* `Conversational agents`: computers having a conversation with us has reshaped the way we run businesses;
* `cost function`, which measures how well the models are learning;
* `labeled data`: when learning data comes with description, targets or desired outputs besides indicative signals, the learning goal becomes to find a general rule that maps inputs to outputs;
* learning types:
  * `reinforced`: learning data provides feedback so that the machine adapts to dynamic conditions in order to achieve a certain goalas for instance self-driving cars and chess master AlphaGo.;
  * `supervised`: when learning data contains only indicative signals without any description attached, it is up to us to find structure of the data underneath, to discover hidden information, or to determine how to describe the data; commonly used in daily applications, such as face and speech recognition, products or movie recommendations, and sales forecasting;
  * `supervised classification`:  attempts to find the appropriate class label, such as analyzing positive/negative sentiment and prediction loan defaults;
  * `supervised, regression`: trains on and predicts a continuous-valued response, for example predicting house prices;
  * `supervised, semi`: uses unlabeled data (typically a large amount) for training, in conjunction a small amount of labeled data.
  * `unsupervised`: when learning data contains only indicative signals without any description attached, it is up to us to find structure of the data underneath, to discover hidden information, or to determine how to describe the data;  used to detect anomalies, such as fraud or defective equipment, or to group customers with similar online behaviors for a marketing campaign.
* `learning data`
* `loss function`, see `cost function`;
* `machine learning` replaces instructing machines thru rules that process well defined data in a predictable manner, with enabling machines thru trained models which command machines to learn and extract patterns, and to figure things out themselves from abundant data; the main task of machine learning is to explore and construct algorithms that can learn from historical data and make predictions on new input data;
* `machine learning algorithms`:
    * `logic-based learning`: used basic rules specified by human experts, and with these rules, systems tried to reason using formal logic, background knowledge, and hypotheses.
    * `artificial neural networks`: imitate animal brains, and consist of interconnected neurons that are also an imitation of biological neurons: modeling complex relationships between inputs and outputs and to capture patterns in data;
    * `genetic algorithms`: popular in the 1990s, mimic the biological process of evolution and try to find the optimal solutions using methods such as mutation and crossover;
    * `deep learning`: believed to resemble the way humans learn, emerged in 2007, based on multi-layered neural networks supported by graphical processing units (GPU) to massively speed up computation, particularly matrix and vector algebra required by these networks;
    * `statistical learning`:
* `Natural language processing (NLP)` deals with the interactions between machine (computer) and human (natural) languages, including conversation, speech, sign languages, etc.
* `ontology`: the way we organize knowledge;
* `overfitting`: is the production of an analysis that corresponds too closely or exactly to a particular set of data, and may therefore fail to fit additional data or predict future observations reliably".
* `part of speech (POS)`: is a grammatical word category such as a noun or verb;
* `stop word`: word that occurs very frequently in a text and are to be ignored when tokenizing the text;
* `tokenization`: the task of breaking language into fragments separated with whitespaces (`tokens`), while removing certain punctuations, digits, emoticons. These fragments are the so-called tokens used for further processing. Moreover, tokens composed of one word are also called unigrams in computational linguistics; bigrams are composed of two consecutive words, trigrams of three consecutive words, and n-grams of n consecutive words. Here is an example of tokenization
* `token`: labels elements of language, identified by a NLP algorithm;
* `training sample`: see `training data`;
* `training sets`: equivalent to the training question used to prepare for a test, this data is used to fine tune models;
* `turing's test (1950)`; a human, communicating with a machine,  asserts they are communicating with another human;
* `validation sample`: see `validation data`;
* `validation sets`: help us verify how well the models will perform in a simulated setting then we fine-tune the models accordingly in order to achieve greater hits;
* `underfitting`: occurs when we train models with not enough data;
* `unlabeled data`:
* `variance`: measures how sensitive the model prediction is to variations in the datasets
* `variance, high`: results from models working with `low bias` data;

# NLP
Remember to run `source ~/projects/pyMlBook/bin/activate` whenever you enter the `~/projects/pyMlBook` project;
* The text instructed me to: `print names.words()[:10]`; this fails, the correct syntax is `(print names.words()[:10])`;
* Had to install `pip`, so that I can install the new libraries, see [this site|https://ahmadawais.com/install-pip-macos-os-x-python/]
* Installed [gemsim|https://radimrehurek.com/gensim/auto_examples/index.html]: `pip install --upgrade genism`
* Installed [TextBlob|https://textblob.readthedocs.io/en/dev/]: `pip install -U textblob`;

# Chapter 2. Exploring the 20 Newsgroups Dataset with Text Analysis Algorithms
## What is NLP?
Deals with the interactions between machine (computer) and human (natural) languages, including conversation, speech, sign languages, translations, etc.
* `turing's test (1950)`; a human, communicating with a machine,  asserts they are communicating with another human;
* `Georgetown–IBM experiment (1954)`: scientists claimed that machine translation would be solved within three to five years; although we've seen much progress, the problem has not been fully solved;
* `ontology`: the way we organize knowledge;
* `Conversational agents`: computers having a conversation with us has reshaped the way we run businesses;
* `part of speech (POS)`: is a grammatical word category such as a noun or verb; it tries to determine the appropriate tag for each word in a sentence or a larger document. The following table gives examples of English POS:
* Part of speech Examples
  * `Noun`: David, machine;
  * `Pronoun`: Them, her;
  * `Adjective`: Awesome, amazing;
  * `Verb`: Read, write;
  * `Adverb`: Very, quite;
  * `Preposition`: Out, at;
  * `Conjunction`: And, but;
  * `Interjection`: : Unfortunately, luckily;
  * `Article`: : A, the;

## Touring powerful NLP libraries in Python
* `Tokenization`: Given a text sequence, tokenization is the task of breaking it into fragments separated with whitespaces. Meanwhile, certain characters are usually removed, such as punctuations, digits, emoticons. These fragments are the so-called tokens used for further processing. Moreover, tokens composed of one word are also called unigrams in computational linguistics; bigrams are composed of two consecutive words, trigrams of three consecutive words, and n-grams of n consecutive words. Here is an example of tokenization;
* `POS tagging`: We can apply an off-the-shelf tagger or combine multiple `NLTK taggers` to customize the tagging process. It is easy to directly use the built-in tagging function pos_tag, as in pos_tag(input_tokens) for instance. But behind the scene, it is actually a prediction from a prebuilt supervised learning model. The model is trained based on a large corpus composed of words that are correctly tagged.

## The newsgroups data
