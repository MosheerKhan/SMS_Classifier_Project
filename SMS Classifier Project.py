#!/usr/bin/env python
# coding: utf-8

# # Company : Bharat Intern

# ***Position : Data Science Intern***
# 

# ***Auther : Mosheer Khan***

# ***Project Description : SMS Classifier Project***

#  Develop a text classification model to classify SMS as either **spam or non-spam** using data science techiques in python

# ***Stages to perform this project:***

# ***1.Data Gathering***

# ***2.Data Cleaning***

# ***3.Exploratory Data Analysis (EDA)***

# ***4.Text preprocessing***

# ***5.Model Building***

# ***6.Evaluation of Model***

# ***7.Website***

# # **Step- 1 DATA GATHERING**

# In[5]:


import numpy as np


# In[6]:


import pandas as pd


# In[15]:


import seaborn as sns


# In[19]:


import matplotlib.pyplot as plt


# In[26]:


df=pd.read_csv(r"C:\Users\Mosheer\Downloads\spam.csv",encoding='latin1')
df


# In[27]:


#Check the row and columns:
df.shape


# # Step- 2 DATA CLEANING

# In[28]:


#Check the cloumns name and their datatype:
df.info


# In[31]:


#Drop the unncessary columns i.e last 3 columns:
df.drop(columns=['Unnamed: 2' , 'Unnamed: 3' , 'Unnamed: 4'],inplace = True)


# In[32]:


df.head()


# In[33]:


#Rename the columns name
df.rename(columns={'v1' : 'Target' , 'v2' : 'Text'} , inplace=True)


# In[34]:


df.head()


# In[37]:


#Applying label Encoder for the Target column:
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[39]:


df['Target'] = encoder.fit_transform(df['Target'])


# In[40]:


df.head()


# In[41]:


df.isnull().sum()

#There is no missing values:


# In[43]:


#Check the duplicate values:
df.duplicated().sum()

#There are 403 duplicate values


# In[44]:


#Remove the all duplicate values:
df = df.drop_duplicates(keep = "first")


# In[45]:


#Recheck the column after drop the duplicate column:
df.duplicated().sum()


#There is no any nulls values


# # Step- 3 EXPLORATORY DATA ANALYSIS (EDA)

# In[47]:


df.head()


# In[51]:


#Check how many message are spam or ham:
df['Target'].value_counts()


# In[63]:


#Represent the no. of messages which are spam or ham using MATPLOTLIB
plt.pie(df['Target'].value_counts(), colors = [ 'Green', 'red'], labels=['ham','spam'], autopct="%0.1f")
plt.title('Distribution of ham and spam in percentage')

plt.show()


# # Step- 4 TEXT PREPROCESSING USING NLTK LIBRARY

# ***To summarize the task using the NLTK library***

# 1.Counting character: Use NLTK to count the no. of alphabets in the SMS text.

# 2.Counting words: Utilize NLTK to count the no. of words in the SMS text.

# 3.Counting sentance: Employ NLTK to count the no. of sentences in the SMS text.

# AND DO ANALYSIS

# In[64]:


#NLTK - Natural Language Toolkit
get_ipython().system('pip install nltk')


# In[66]:


get_ipython().system('pip install nltk')


# In[67]:


import nltk


# In[68]:


import nltk
nltk.download('punkt')


# # Check Number of Characters:

# In[70]:


df['num_characters'] = df['Text'].apply(len)
df.head()


# In[71]:


# Use of nltk library
df['num_words'] = df['Text'].apply(nltk.word_tokenize)
df


# In[72]:


# Count number of words in each SMS text
df['num_words'] = df['num_words'].apply(len)
df


# # Check Number of Sentences:

# In[73]:


df['num_sentences'] = df['Text'].apply(nltk.sent_tokenize)
df


# In[74]:


# Count number of sentence in each sms text

df['num_sentences'] = df['num_sentences'].apply(len)
df.head()


# In[75]:


# See the descriptive statistics data:
df[['num_characters', 'num_words', 'num_sentences']].describe()


# In[77]:


# See ham messages:
df.loc[df['Target'] == 0][['num_characters', 'num_words', 'num_sentences']].describe()


# In[78]:


# See spam messages:
df.loc[df['Target'] == 1][['num_characters', 'num_words', 'num_sentences']].describe()


# # Note:

# The Descriptive Statistics of ham and spam messages indicate that, on average, spam messages tend to be slightly larger in size compared to ham messages.

# # Characters:

# In[82]:


# See the histplot of ham messages and spam messages:
plt.figure(figsize=(12,5))
ham_messages = df.loc[df['Target']==0][['num_characters']]            # ham messages
spam_messages = df.loc[df['Target']==1][['num_characters']]           # spam messages
sns.histplot(spam_messages['num_characters'], color='Red')
sns.histplot(ham_messages['num_characters'], color='yellow')
plt.title('Distribution of ham messages and spam messages character')
plt.show()


# In[85]:


# See the histplot of ham messages and spam messages:
plt.figure(figsize=(12,5))
ham_messages = df.loc[df['Target']==0][['num_characters']]            # ham messages
spam_messages = df.loc[df['Target']==1][['num_characters']]           # spam messages
sns.histplot(spam_messages['num_characters'], color='Red')
sns.histplot(ham_messages['num_characters'], color='Yellow')
plt.title('Distribution of ham messages and spam messages words')
plt.show()


# # Finding relation between Number of characters and Number of words:

# In[86]:


sns.pairplot(df, hue='Target')
plt.show()


# By seeing the above pairplot we can say that there are some outlier in the dataset.

# # Correlation Coefficient

# In[87]:


df[['Target', 'num_characters', 'num_words', 'num_sentences']].corr()


# In[90]:


## Use heatmap for better understanding:
sns.heatmap(df[['Target', 'num_characters', 'num_words', 'num_sentences']].corr(), annot=True)    
plt.show()


# NOTE: annot=True in a heatmap function adds numerical annotations to each cell of the heatmap for improved data interpretation.

# # Analyze by the above Heatmap:

# The correlation analysis indicates that there is a positive correlation of 𝟬.𝟯𝟴 between the number of characters in a message and the likelihood of it being classified as spam. This suggests that as the number of characters in a message increases, there is a tendency for it to be classified as spam, potentially due to certain characteristics or patterns associated with longer messages in the context of spam classification.

# 𝗗𝗘𝗖𝗜𝗦𝗜𝗢𝗡 - The decision to use the "num_characters" column in model creation is driven by its correlation coefficient of 𝟬.𝟯𝟴 with the target variable, indicating a stronger association compared to "num_words" and "num_sentences."

# # Data Preprocessing:

# The below code is performing data preprocessing on the DataFrame. It includes steps like removing punctuation, converting text to lowercase, splitting text into words, and lemmatizing the words.
# 
# ‣ Lower case - 𝐶𝑜𝑛𝑣𝑒𝑟𝑡 𝑣𝑎𝑙𝑢𝑒𝑠 𝑖𝑛 𝑙𝑜𝑤𝑒𝑟 𝑐𝑎𝑠𝑒.
# ‣ Tokenization - 𝘱𝘳𝘰𝘤𝘦𝘴𝘴 𝘰𝘧 𝘣𝘳𝘦𝘢𝘬𝘪𝘯𝘨 𝘥𝘰𝘸𝘯 𝘵𝘦𝘹𝘵 𝘪𝘯𝘵𝘰 𝘴𝘮𝘢𝘭𝘭𝘦𝘳 𝘶𝘯𝘪𝘵𝘴⸴ 𝘴𝘶𝘤𝘩 𝘢𝘴 𝘸𝘰𝘳𝘥𝘴
# ‣ Removing special characters -
# 
#  Eg: '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'  etc.
#  
# ‣ Removing stopwords and punctuation -
#  Eg: (a, an, the, and, but, or, in, on, at, by, with, from, to, of, for, I, you, he, she, it, we, they, me, him, her,
#        us, them, is, are, am, was, were, be, being, been, have, has, had, do, does, did, will, would, shall, should,
#        can, could, may, might, must) etc.

# ‣ Stemming - Involves 𝙇𝙚𝙢𝙢𝙞𝙯𝙖𝙩𝙞𝙤𝙣
#  To reduce words to their base or root form, ensuring consistency and simplification for analysis, such as converting variations like "dancing," "dance," and "danced" to their common base form "dance" to streamline text processing and analysis.

# In[91]:


from nltk.corpus import stopwords       # 𝙨𝙩𝙤𝙥𝙬𝙤𝙧𝙙𝙨 module is used to access a list of common stopwords in various languages
from nltk.stem.porter import PorterStemmer
import string                           # Import string
ps = PorterStemmer()


# In[92]:


def transform_text(text):
    text = text.lower()                 # Converting text to lowercase
    text = nltk.word_tokenize(text)     # Doing tokenize to break down text into small words of list
    
    y = []                    
    for i in text:
        if i.isalnum():                 # Removing special characters
            y.append(i)
              
    text = y[:]                    # Assign text to y                                  
    y.clear()                      # Clear the y
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:     # Removing stopwords and punctuation
            y.append(i)
            
            
    text = y[:]                    # Again, Assign text to y
    y.clear()                      # And clear the y
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)             # Joining of list of data into a single string


# In[98]:


transform_text("I loved the YT lectures on Machine Learning. How about you?")


# In[102]:


df['transformed_text'] = df['Text'].apply(transform_text)


# In[103]:


df.head()


# # Word Cloud

# 𝐍𝐨𝐰 𝐰𝐞 𝐜𝐚𝐧 𝐰𝐨𝐫𝐤𝐢𝐧𝐠 𝐭𝐨 𝐟𝐢𝐧𝐝 𝐓𝐨𝐩 𝐰𝐨𝐫𝐝𝐬 𝐰𝐡𝐢𝐜𝐡 𝐚𝐫𝐞 𝐮𝐬𝐞𝐝 𝐢𝐧 𝐡𝐚𝐦 𝐚𝐧𝐝 𝐬𝐩𝐚𝐦 𝐦𝐞𝐬𝐬𝐚𝐠𝐞𝐬 𝐚𝐧𝐝 𝐮𝐬𝐞 𝐰𝐨𝐫𝐝 𝐜𝐥𝐨𝐮𝐝 𝐭𝐨 𝐬𝐡𝐨𝐰 𝐭𝐡𝐞 𝐫𝐞𝐩𝐫𝐞𝐬𝐞𝐧𝐭𝐚𝐭𝐢𝐨𝐧:

# In[105]:


get_ipython().system(' pip install wordCloud')


# # Top words in spam messages:

# In[111]:


plt.figure(figsize=(8,15))
from wordcloud import WordCloud         # Use wordcloud module to show the top text word used in ham and spam messages
wc = WordCloud(width=1000, height=1000, min_font_size=10, background_color='Black')
spam_wc = wc.generate(df[df['Target'] == 1]['transformed_text'].str.cat(sep=" "))    # Use generate function, and 1 for "spam"
plt.imshow(spam_wc) 
plt.show()
# To show the word cloud chart


# In[110]:


plt.figure(figsize=(8, 15))
wc = WordCloud(width=1000, height=1000, min_font_size=10, background_color='white')
ham_wc = wc.generate(df[df['Target']==0]['transformed_text'].str.cat(sep=" "))
plt.imshow(ham_wc) 
plt.show()
# To show the word cloud chart


# # 𝐓𝐨𝐩 40 𝐰𝐨𝐫𝐝𝐬 𝐮𝐬𝐞 𝐢𝐧 𝐬𝐩𝐚𝐦 𝐦𝐞𝐬𝐬𝐚𝐠𝐞:

# In[112]:


df.loc[df['Target']==1]['transformed_text']


# In[116]:


plt.figure(figsize=(15,5))
from collections import Counter      # Use Counter module
spam_words = []
for message in df.loc[df['Target']==1]['transformed_text'].tolist():
    for words in message.split():
        spam_words.append(words)
        
spam_data = pd.DataFrame(Counter(spam_words).most_common(40), columns=['VALUE', 'COUNT'])    # Most 30 common words
ax = sns.barplot(x='VALUE', y='COUNT', data=spam_data)
plt.title('Top 40 Most common words used in spam messages')
plt.xticks(rotation=90)
for bars in ax.containers:
    ax.bar_label(bars)


# # 𝐓𝐨𝐩 40 𝐰𝐨𝐫𝐝𝐬 𝐮𝐬𝐞 𝐢𝐧 𝐡𝐚𝐦 𝐦𝐞𝐬𝐬𝐚𝐠𝐞:

# In[119]:


df.loc[df['Target']==0]['transformed_text']


# In[121]:


plt.figure(figsize=(15,5))
from collections import Counter
ham_words = []
for messages in df.loc[df['Target']==0]['transformed_text'].tolist():
    for words in messages.split():
        ham_words.append(words)

ham_data = pd.DataFrame(Counter(ham_words).most_common(40), columns=['VALUE', 'COUNT'])    # Most 30 common words
ax = sns.barplot(x='VALUE', y='COUNT', data=ham_data)
plt.title('Top 40 Most common words used in ham messages')
plt.xticks(rotation=90)
for bars in ax.containers:
    ax.bar_label(bars)


# # Step- 5 Model Building

# ***Naive Bayes Algorithmn***

# 
# For an SMS classifier project, Naive Bayes algorithms is suitable choice due to the nature of textual data and the simplicity of the algorithm.

# 𝐈𝐧𝐩𝐮𝐭: transformed_text (𝙏𝙖𝙨𝙠 𝙩𝙤 𝙘𝙤𝙣𝙫𝙚𝙧𝙩 𝙞𝙣𝙩𝙤 𝙣𝙪𝙢𝙚𝙧𝙞𝙘𝙖𝙡 𝙫𝙖𝙡𝙪𝙚)
# 
# 𝐎𝐮𝐭𝐩𝐮𝐭: target

# ***Nᴏᴛᴇ:***
# For building machine learning model the input data should be numerical/ vector, but here 'transformed_text' is in string format so we have deal with it using "bag of words".

# # 𝗖𝗼𝘂𝗻𝘁𝗩𝗲𝗰𝘁𝗼𝗿𝗶𝘇𝗲𝗿
# 

# ***𝐖𝐡𝐚𝐭 𝐢𝐬 𝐛𝐚𝐠 𝐨𝐟 𝐰𝐨𝐫𝐝𝐬 (𝗖𝗼𝘂𝗻𝘁𝗩𝗲𝗰𝘁𝗼𝗿𝗶𝘇𝗲𝗿)?***

# The bag of words model simplifies text into numbers by counting how often words appear, making it easier for computers to analyze and process text data for tasks like classification and analysis.

# In[122]:


# Use of 𝗖𝗼𝘂𝗻𝘁𝗩𝗲𝗰𝘁𝗼𝗿𝗶𝘇𝗲𝗿 class from 𝘀𝗸𝗹𝗲𝗮𝗿𝗻.𝗳𝗲𝗮𝘁𝘂𝗿𝗲_𝗲𝘅𝘁𝗿𝗮𝗰𝘁𝗶𝗼𝗻.𝘁𝗲𝘅𝘁 module for converting collection of text documents into a
# matrix of token counts.

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
cv = CountVectorizer()

X1 = cv.fit_transform(df['transformed_text']).toarray()

print("SMS:", X1.shape[0])
print("Total Words:", X1.shape[1])


# In[123]:


print(X1)


# In[125]:


y1 = df['Target'].values
y1


# # Now we apply train-test split()

# In[126]:


from sklearn.model_selection import train_test_split


# # Why we use?

# Naive Bayes classifiers, such as MultinomialNB and BernoulliNB, are commonly used for SMS classifier projects because they are efficient, easy to implement, effective for text classification tasks, handle sparse data well, scale to large datasets, and achieve good performance in practice.

# 𝐃𝐢𝐟𝐟𝐞𝐫𝐞𝐧𝐜𝐞 𝐛𝐞𝐭𝐰𝐞𝐞𝐧 𝐭𝐡𝐞𝐬𝐞 𝐍𝐚𝐢𝐲𝐞 𝐁𝐚𝐲𝐞𝐬 𝐀𝐥𝐠𝐨𝐫𝐢𝐭𝐡𝐦𝐧𝐬❓
# 
# • 𝗚𝗮𝘂𝘀𝘀𝗶𝗮𝗻𝗡𝗕 works with data that looks like a bell curve, 𝗺𝗲𝗮𝗻𝘀: it is typically used for features that are continuous or real-valued.
# 
# • 𝗠𝘂𝗹𝘁𝗶𝗻𝗼𝗺𝗶𝗮𝗹𝗡𝗕 works with counting occurrences, like words in text.
# 
# • 𝗕𝗲𝗿𝗻𝗼𝘂𝗹𝗹𝗶𝗡𝗕 works with situations where you're just interested in whether something happens or not, like yes or no questions.

# # 𝗪𝗵𝘆 𝘄𝗲 𝘂𝘀𝗲 𝗽𝗿𝗲𝗰𝗶𝘀𝗶𝗼𝗻_𝘀𝗰𝗼𝗿𝗲❓

# 𝗥𝗘𝗔𝗦𝗢𝗡: Precision help to make sure that SMS classifier is accurate in catching spam while avoiding mistakes that could upset users or miss business goals.

# In[128]:


# Import all the Naive Bayes Algorithmn because I don't know the distribution of data.

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=2)   # 20% data for test
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

gnb1 = GaussianNB()                
mnb1 = MultinomialNB()
bnb1 = BernoulliNB()


# # Gaussian Naive Bayes (GaussianNB):

# In[129]:


gnb1.fit(X1_train, y1_train)                                         # train the model
y1_predict1 = gnb1.predict(X1_test)
print("accuracy score:", accuracy_score(y1_test, y1_predict1))      # The accuracy is 88%
print("confusion_matrix:", confusion_matrix(y1_test, y1_predict1))
print("precision_score:", precision_score(y1_test, y1_predict1))    # The precision score is 53%


# 𝗗𝗘𝗖𝗜𝗦𝗜𝗢𝗡 𝗧𝗔𝗞𝗘𝗡- The 𝗚𝗮𝘂𝘀𝘀𝗶𝗮𝗻 𝗡𝗮𝗶𝘃𝗲 𝗕𝗮𝘆𝗲𝘀 classifier achieved an 𝗮𝗰𝗰𝘂𝗿𝗮𝗰𝘆 of 𝟖𝟖%, indicating reasonable overall performance. However, its 𝗽𝗿𝗲𝗰𝗶𝘀𝗶𝗼𝗻 𝘀𝗰𝗼𝗿𝗲 of 𝟬.𝟱𝟯 suggests it struggles to accurately identify spam, potentially leading to a significant number of false positives. Given the importance of precision in spam detection to maintain user satisfaction, I choose not to use the 𝗚𝗮𝘂𝘀𝘀𝗶𝗮𝗻 𝗡𝗮𝗶𝘃𝗲 𝗕𝗮𝘆𝗲𝘀 classifier.

# # Multinomial Naive Bayes (MultinomialNB):

# In[130]:


mnb1.fit(X1_train, y1_train)
y1_predict2 = mnb1.predict(X1_test)
print("accuracy score:", accuracy_score(y1_test, y1_predict2))      # The accuracy is 96%
print("confusion_matrix:", confusion_matrix(y1_test, y1_predict2))  
print("precision_score:", precision_score(y1_test, y1_predict2))    # The precision score is 83%


# 𝗡𝗼𝘁𝗲: I need to focus on 𝗽𝗿𝗲𝗰𝗶𝘀𝗶𝗼𝗻_𝘀𝗰𝗼𝗿𝗲 first rather than accuracy because the data are imbalanced.
# 
# 𝗗𝗘𝗖𝗜𝗦𝗜𝗢𝗡 𝗧𝗔𝗞𝗘𝗡- The 𝗠𝘂𝗹𝘁𝗶𝗻𝗼𝗺𝗶𝗮𝗹 𝗡𝗮𝗶𝘃𝗲 𝗕𝗮𝘆𝗲𝘀 classifier achieved an 𝗮𝗰𝗰𝘂𝗿𝗮𝗰𝘆 of 𝟵𝟲%, indicating reasonable overall performance. However, its 𝗽𝗿𝗲𝗰𝗶𝘀𝗶𝗼𝗻 𝘀𝗰𝗼𝗿𝗲 of 𝟬.𝟴𝟯 suggests it struggles to accurately identify spam, potentially leading to a significant number of false positives. Given the importance of precision in spam detection to maintain user satisfaction, I choose not to use the 𝗠𝘂𝗹𝘁𝗶𝗻𝗼𝗺𝗶𝗮𝗹 𝗡𝗮𝗶𝘃𝗲 𝗕𝗮𝘆𝗲𝘀 classifier.

# # Bernoulli Naive Bayes (BernoulliNB)

# In[131]:


bnb1.fit(X1_train, y1_train)
y1_predict3 = bnb1.predict(X1_test)
print("accuracy score:", accuracy_score(y1_test, y1_predict3))      # The accuracy is 97%
print("confusion_matrix:", confusion_matrix(y1_test, y1_predict3))
print("precision_score:", precision_score(y1_test, y1_predict3))    # The precision score is 97%


# 𝗗𝗘𝗖𝗜𝗦𝗜𝗢𝗡 𝗧𝗔𝗞𝗘𝗡- The 𝗕𝗲𝗿𝗻𝗼𝘂𝗹𝗹𝗶 𝗡𝗮𝗶𝘃𝗲 𝗕𝗮𝘆𝗲𝘀 classifier achieved an 𝗮𝗰𝗰𝘂𝗿𝗮𝗰𝘆 of 𝟵𝟳%, indicating reasonable overall performance and the 𝗽𝗿𝗲𝗰𝗶𝘀𝗶𝗼𝗻 𝘀𝗰𝗼𝗿𝗲 of 𝟬.𝟵𝟳 indicates that 𝗕𝗲𝗿𝗻𝗼𝘂𝗹𝗹𝗶 𝗡𝗮𝗶𝘃𝗲 𝗕𝗮𝘆𝗲𝘀 is better than rest of the two algorithmn.

# # 𝐓𝐟𝐢𝐝𝐟𝐕𝐞𝐜𝐭𝐨𝐫𝐢𝐳𝐞𝐫

# What is 𝐓𝐟𝐢𝐝𝐟𝐕𝐞𝐜𝐭𝐨𝐫𝐢𝐳𝐞𝐫?
# 
# 𝐓𝐟𝐢𝐝𝐟𝐕𝐞𝐜𝐭𝐨𝐫𝐢𝐳𝐞𝐫 is a tool in natural language processing that transforms text data into numbers for machine learning. 𝗜𝘁 𝗮𝘀𝘀𝗶𝗴𝗻𝘀 𝘀𝗰𝗼𝗿𝗲𝘀 𝘁𝗼 𝘄𝗼𝗿𝗱𝘀 𝗯𝗮𝘀𝗲𝗱 𝗼𝗻 𝘁𝗵𝗲𝗶𝗿 𝗶𝗺𝗽𝗼𝗿𝘁𝗮𝗻𝗰𝗲 𝗶𝗻 𝗲𝗮𝗰𝗵 𝗱𝗼𝗰𝘂𝗺𝗲𝗻𝘁 𝗮𝗻𝗱 𝗮𝗰𝗿𝗼𝘀𝘀 𝘁𝗵𝗲 𝗲𝗻𝘁𝗶𝗿𝗲 𝗱𝗮𝘁𝗮𝘀𝗲𝘁. This helps algorithms understand the context and significance of words. In short, TfidfVectorizer makes text data understandable for machine learning algorithms by giving numerical importance to words.

# In[132]:


# Use of 𝐓𝐟𝐢𝐝𝐟𝐕𝐞𝐜𝐭𝐨𝐫𝐢𝐳𝐞𝐫 class from 𝘀𝗸𝗹𝗲𝗮𝗿𝗻.𝗳𝗲𝗮𝘁𝘂𝗿𝗲_𝗲𝘅𝘁𝗿𝗮𝗰𝘁𝗶𝗼𝗻.𝘁𝗲𝘅𝘁 

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000)                     # Use max 3000 words

X = tfidf.fit_transform(df['transformed_text']).toarray()

print(X.shape)


# In[134]:


y = df['Target'].values
y


# # Now we apply train-test split()

# In[135]:


from sklearn.model_selection import train_test_split


# In[136]:


# Import all the Naive Bayes Algorithmn because I don't know the distribution of data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)   # 20% data for test
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score


# Why we use precision_score?
# Rᴇᴀsᴏɴ: Precision help to make sure that SMS classifier is accurate in catching spam while avoiding mistakes that could 
# upset users or miss business goals.


gnb = GaussianNB()                
mnb = MultinomialNB()
bnb = BernoulliNB()


# # Gaussian Naive Bayes (GaussianNB):

# In[137]:


gnb.fit(X_train, y_train)                                         # train the model
y_pred1 = gnb.predict(X_test)
print("accuracy score:", accuracy_score(y_test, y_pred1))      # The accuracy is 86%
print("confusion_matrix:", confusion_matrix(y_test, y_pred1))
print("precision_score:", precision_score(y_test, y_pred1))    # The precision score is 50%


# 𝗗𝗘𝗖𝗜𝗦𝗜𝗢𝗡 𝗧𝗔𝗞𝗘𝗡- The 𝗚𝗮𝘂𝘀𝘀𝗶𝗮𝗻 𝗡𝗮𝗶𝘃𝗲 𝗕𝗮𝘆𝗲𝘀 classifier achieved an 𝗮𝗰𝗰𝘂𝗿𝗮𝗰𝘆 of 𝟴𝟳%, indicating reasonable overall performance. However, its 𝗽𝗿𝗲𝗰𝗶𝘀𝗶𝗼𝗻 𝘀𝗰𝗼𝗿𝗲 of 𝟬.𝟱𝟬 suggests it struggles to accurately identify spam, potentially leading to a significant number of false positives. Given the importance of precision in spam detection to maintain user satisfaction, I choose not to use the 𝗚𝗮𝘂𝘀𝘀𝗶𝗮𝗻 𝗡𝗮𝗶𝘃𝗲 𝗕𝗮𝘆𝗲𝘀 classifier.

# # Multinomial Naive Bayes (MultinomialNB):

# In[138]:


mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print("accuracy score:", accuracy_score(y_test, y_pred2))      # The accuracy is 97%
print("confusion_matrix:", confusion_matrix(y_test, y_pred2))  
print("precision_score:", precision_score(y_test, y_pred2))    # The precision score is 1.0%


# 𝗗𝗘𝗖𝗜𝗦𝗜𝗢𝗡 𝗧𝗔𝗞𝗘𝗡- The 𝗠𝘂𝗹𝘁𝗶𝗻𝗼𝗺𝗶𝗮𝗹 𝗡𝗮𝗶𝘃𝗲 𝗕𝗮𝘆𝗲𝘀 classifier achieved an 𝗮𝗰𝗰𝘂𝗿𝗮𝗰𝘆 of 97%, indicating reasonable overall performance and 𝗽𝗿𝗲𝗰𝗶𝘀𝗶𝗼𝗻 𝘀𝗰𝗼𝗿𝗲 of 𝟭.𝟬 which means that 𝘁𝗵𝗶𝘀 𝗺𝗼𝗱𝗲𝗹 𝗱𝗼𝗻❜𝘁 𝗴𝗶𝘃𝗲 𝗮𝗻𝘆 𝗙𝗮𝗹𝘀𝗲 𝗽𝗼𝘀𝗶𝘁𝗶𝘃𝗲 and it is suitable also.

# # Bernoulli Naive Bayes (BernoulliNB)

# In[139]:


bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print("accuracy score:", accuracy_score(y_test, y_pred3))      # The accuracy is 98%
print("confusion_matrix:", confusion_matrix(y_test, y_pred3))
print("precision_score:", precision_score(y_test, y_pred3))    # The precision score is 99%


# 𝗗𝗘𝗖𝗜𝗦𝗜𝗢𝗡 𝗧𝗔𝗞𝗘𝗡- The 𝗕𝗲𝗿𝗻𝗼𝘂𝗹𝗹𝗶 𝗡𝗮𝗶𝘃𝗲 𝗕𝗮𝘆𝗲𝘀 classifier achieved an 𝗮𝗰𝗰𝘂𝗿𝗮𝗰𝘆 of 98%, indicating reasonable overall performance and the 𝗽𝗿𝗲𝗰𝗶𝘀𝗶𝗼𝗻 𝘀𝗰𝗼𝗿𝗲 of 𝟬.𝟵𝟵 indicates that 𝗕𝗲𝗿𝗻𝗼𝘂𝗹𝗹𝗶 𝗡𝗮𝗶𝘃𝗲 𝗕𝗮𝘆𝗲𝘀 is better than rest of the two algorithmn.

# # 𝗥𝗜𝗚𝗛𝗧 𝗡𝗢𝗪 𝗗𝗘𝗖𝗜𝗦𝗜𝗢𝗡

# The 𝗠𝘂𝗹𝘁𝗶𝗻𝗼𝗺𝗶𝗮𝗹 𝗡𝗮𝗶𝘃𝗲 𝗕𝗮𝘆𝗲𝘀 (𝗠𝘂𝗹𝘁𝗶𝗻𝗼𝗺𝗶𝗮𝗹𝗡𝗕) with 𝐓𝐟𝐢𝐝𝐟𝐕𝐞𝐜𝐭𝐨𝐫𝐢𝐳𝐞𝐫 are best fit for the 𝗦𝗠𝗦-𝗖𝗹𝗮𝘀𝘀𝗶𝗳𝗶𝗲𝗿 𝗣𝗿𝗼𝗷𝗲𝗰𝘁 due to its 𝗽𝗿𝗲𝗰𝗶𝘀𝗶𝗼𝗻 𝘀𝗰𝗼𝗿𝗲 of 𝟭.𝟬 and 𝗮𝗰𝗰𝘂𝗿𝗮𝗰𝘆 𝗶𝘀 𝟵𝟱%, indicating that 𝘁𝗵𝗶𝘀 𝗺𝗼𝗱𝗲𝗹 𝗱𝗼𝗻❜𝘁 𝗴𝗶𝘃𝗲 𝗮𝗻𝘆 𝗙𝗮𝗹𝘀𝗲 𝗣𝗼𝘀𝗶𝘁𝗶𝘃𝗲.
# 
# However, I'm exploring other algorithms to determine if any of them might be suitable for the 𝗦𝗠𝗦 𝗖𝗹𝗮𝘀𝘀𝗶𝗳𝗶𝗲𝗿 𝗣𝗿𝗼𝗷𝗲𝗰𝘁.
# 

# # 𝗘𝗫𝗣𝗟𝗢𝗥𝗘:
# # 𝐋𝐨𝐠𝐢𝐬𝐭𝐢𝐜𝐑𝐞𝐠𝐫𝐞𝐬𝐬𝐢𝐨𝐧
# # 𝐃𝐞𝐜𝐢𝐬𝐢𝐨𝐧𝐓𝐫𝐞𝐞𝐂𝐥𝐚𝐬𝐬𝐢𝐟𝐢𝐞𝐫
# # 𝐊𝐍𝐞𝐢𝐠𝐡𝐛𝐨𝐫𝐬𝐂𝐥𝐚𝐬𝐬𝐢𝐟𝐢𝐞𝐫
# # 𝐑𝐚𝐧𝐝𝐨𝐦𝐅𝐨𝐫𝐞𝐬𝐭𝐂𝐥𝐚𝐬𝐬𝐢𝐟𝐢𝐞𝐫

# In[140]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[141]:


# Creating object:
lrc = LogisticRegression(solver='liblinear', penalty='l1')
dtc = DecisionTreeClassifier(max_depth=5)
knc = KNeighborsClassifier()
rfc = RandomForestClassifier(n_estimators=50, random_state=2)


# In[142]:


# Creating a dictionary to hold the classifier:
# keys   -> Algorithmn name
# values -> object name
classifiers = {
    'LR' : lrc,
    'DT' : dtc,
    'KN' : knc,
    'RF' : rfc
}


# In[143]:


def train_classifier(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, precision 


# In[144]:


train_classifier(lrc, X_train, y_train, X_test, y_test)


# In[145]:


accuracy_scores = []
precision_scores = []

for name, classifier in classifiers.items():
    current_accuracy, current_precision = train_classifier(classifier, X_train, y_train, X_test, y_test)
    
    print("For:", name)
    print("Accuracy:", current_accuracy)
    print("Precision:", current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
    


# In[146]:


performance_df = pd.DataFrame({'algorithm' : classifiers.keys(), 'accuracy': accuracy_scores, 'Precision': precision_scores})
performance_df.sort_values('Precision', ascending=False)


# # 𝐅𝐈𝐍𝐀𝐋 𝐃𝐄𝐂𝐈𝐒𝐈𝐎𝐍

# 𝗕𝘆 𝗲𝘅𝗮𝗺𝗶𝗻𝗲 𝗮𝗻𝗱 𝗰𝗼𝗺𝗽𝗮𝗿𝗲 𝘁𝗵𝗲 𝗽𝗲𝗿𝗳𝗼𝗿𝗺𝗮𝗻𝗰𝗲 𝗼𝗳 𝗱𝗶𝗳𝗳𝗲𝗿𝗲𝗻𝘁 𝗮𝗹𝗴𝗼𝗿𝗶𝘁𝗵𝗺𝘀
# 
# I can say that only the 𝗠𝘂𝗹𝘁𝗶𝗻𝗼𝗺𝗶𝗮𝗹 𝗡𝗮𝗶𝘃𝗲 𝗕𝗮𝘆𝗲𝘀 (𝗠𝘂𝗹𝘁𝗶𝗻𝗼𝗺𝗶𝗮𝗹𝗡𝗕) with 𝐓𝐟𝐢𝐝𝐟𝐕𝐞𝐜𝐭𝐨𝐫𝐢𝐳𝐞𝐫 have 𝗽𝗿𝗲𝗰𝗶𝘀𝗶𝗼𝗻 𝘀𝗰𝗼𝗿𝗲 of 𝟭.𝟬 and 𝗮𝗰𝗰𝘂𝗿𝗮𝗰𝘆 𝗶𝘀 𝟵𝟱% which is best for 𝗦𝗠𝗦-𝗖𝗹𝗮𝘀𝘀𝗶𝗳𝗶𝗲𝗿 𝗣𝗿𝗼𝗷𝗲𝗰𝘁, while 𝗞𝗡𝗲𝗮𝗿𝗲𝘀𝘁 𝗡𝗲𝗶𝗴𝗵𝗯𝗼𝗿 (𝗞𝗡𝗡) 𝗰𝗹𝗮𝘀𝘀𝗶𝗳𝗶𝗲𝗿 also achieves a 𝗽𝗿𝗲𝗰𝗶𝘀𝗶𝗼𝗻 𝘀𝗰𝗼𝗿𝗲 of 𝟭.𝟬, but its 𝗮𝗰𝗰𝘂𝗿𝗮𝗰𝘆 𝗳𝗮𝗹𝗹𝘀 𝘀𝗵𝗼𝗿𝘁 𝗰𝗼𝗺𝗽𝗮𝗿𝗲𝗱 𝘁𝗼 𝗠𝘂𝗹𝘁𝗶𝗻𝗼𝗺𝗶𝗮𝗹𝗡𝗕 𝘄𝗶𝘁𝗵 𝗧𝗳𝗶𝗱𝗳𝗩𝗲𝗰𝘁𝗼𝗿𝗶𝘇𝗲𝗿, rendering it less suitable for 𝗦𝗠𝗦-𝗖𝗹𝗮𝘀𝘀𝗶𝗳𝗶𝗲𝗿 𝗣𝗿𝗼𝗷𝗲𝗰𝘁.

# # PICKLE

# Pickle in Python is primarily used in serialzing and deserializing a Python Object structure. In other words, it's the process of converting a Python object into a byte stream (0,1) to store it in a file/database, maintain program state acrros session, or transport data over the network.
# 
# 
# 𝗦𝗲𝗿𝗶𝗮𝗹𝗶𝘇𝗶𝗻𝗴: Converting a Python object into a byte stream. (𝗣𝗶𝗰𝗸𝗹𝗲)

# In[147]:


import pickle  # Import the pickle module for serialization

# Serialize and save the 'tfidf' object to a file named 'vectorizer.pkl'
# 'wb'(write binary) mode is used for writing the file in binary mode
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))  # Serialize tfidf and save to 'vectorizer.pkl'

# Serialize and save the 'mnb' object to a file named 'model.pkl'
# 'wb' mode is used for writing the file in binary mode
pickle.dump(mnb, open('model.pkl', 'wb'))  # Serialize mnb and save to 'model.pkl'


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




